"""
Comprehensive Automated Test Suite for Marswalk Scientific AI Engine.
Validates all 11 mandatory edge-case test scenarios and anti-hallucination invariants.
"""
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.services.terrain_service import TerrainService
from app.services.route_service import RouteService
from app.services.eva_service import EVAService
from app.services.rad_service import RadiationService
from app.services.space_weather_service import SpaceWeatherService
from app.services.env_service import EnvironmentalService

@pytest.fixture
def anyio_backend():
    return "asyncio"

# ----------------- TEST-01: Flat Terrain ----------------- #
def test_flat_terrain():
    service = TerrainService(cell_size_m=20.0)
    elevations = [-2500.0, -2500.1, -2500.2, -2500.1]
    distances = [0.0, 100.0, 200.0, 300.0]
    res = service.analyze_profile(elevations, distances, max_slope_limit_deg=15.0)
    
    assert res["measurements"]["slope_avg_deg"] < 2.0
    assert res["classification"]["terrain_difficulty"] == "Flat / Easy"
    assert res["classification"]["is_within_limits"] is True
    assert res["calculations"]["steep_segments_count"] == 0

# ----------------- TEST-02: Steep Terrain (>15°) ----------------- #
def test_steep_terrain_hazard():
    service = TerrainService(cell_size_m=20.0)
    # 40m rise over 100m distance = ~21.8 degrees
    elevations = [-2500.0, -2460.0]
    distances = [0.0, 100.0]
    res = service.analyze_profile(elevations, distances, max_slope_limit_deg=15.0)
    
    assert res["measurements"]["slope_max_deg"] > 15.0
    assert res["classification"]["is_within_limits"] is False
    assert "Exceeds Slope Limit" in res["classification"]["terrain_difficulty"]
    assert res["calculations"]["steep_segments_count"] == 1

# ----------------- TEST-03: Invalid Profile Points ----------------- #
def test_invalid_profile_handling():
    service = TerrainService()
    with pytest.raises(ValueError, match="At least two elevation points"):
        service.analyze_profile([-2500.0], [0.0])

# ----------------- TEST-04: Route Exceeding EVA Budget ----------------- #
def test_eva_budget_exceeded():
    eva = EVAService(default_k_suit=0.60)
    # Long route: 12 km with steep 10° slope and multiple stops on a 4.0h life-support budget
    res = eva.calculate_eva_budget(
        segment_lengths_km=[6.0, 6.0],
        segment_slopes_deg=[10.0, 10.0],
        stop_count=4,
        stop_duration_minutes=30.0,
        suit_budget_hours=4.0,
        reserve_percent=20.0
    )
    
    calc = res["calculations"]
    assert calc["total_eva_duration_hours"] > calc["effective_budget_hours"]
    assert res["classification"]["budget_exceeded"] is True
    assert res["classification"]["severity"] in ["Warning", "Critical"]

# ----------------- TEST-05: Route Cost Comparison ----------------- #
def test_route_cost_comparison():
    service = RouteService(max_slope_limit_deg=15.0)
    
    # Route A (Gentle 3° traverse)
    route_a = service.evaluate_route_segments([3.0, 3.0, 3.0], [100.0, 100.0, 100.0])
    # Route B (Steep 12° traverse)
    route_b = service.evaluate_route_segments([12.0, 12.0, 12.0], [100.0, 100.0, 100.0])
    
    assert route_a["weighted_terrain_cost"] < route_b["weighted_terrain_cost"]
    assert route_a["exceeds_slope_threshold"] is False
    assert route_b["exceeds_slope_threshold"] is False

# ----------------- TEST-06: Missing MEDA Observations ----------------- #
def test_missing_meda_sol():
    env = EnvironmentalService()
    # Sol 99999 does not exist in historical archive
    res = env.analyze_environmental_conditions(sol=99999)
    
    assert res["measurements"] == {}
    assert res["classification"]["environmental_status"] == "Unavailable"
    assert any("refuses to fabricate" in lim for lim in res["limitations"])

# ----------------- TEST-07: DONKI Cache Resilience ----------------- #
def test_donki_cache_loading():
    sw = SpaceWeatherService()
    cached = sw.load_cached_data()
    
    assert "sep_events" in cached
    assert "flares" in cached
    assert cached["source"] == "local_resilience_cache"
    
    res = sw.evaluate_space_weather_hazard(cached, sep_threshold_pfu=10.0)
    assert "space_weather_status" in res["classification"]

# ----------------- TEST-08: Radiation Calculation Verification ----------------- #
def test_radiation_exact_math():
    rad = RadiationService(baseline_mgy_per_day=0.21, quality_factor_q=3.05)
    
    # 4.0 hours EVA: dose = 0.21 * (4.0 / 24.0) = 0.035 mGy
    res = rad.estimate_exposure(eva_duration_hours=4.0)
    calc = res["calculations"]
    
    expected_mgy = 0.21 * (4.0 / 24.0)
    assert abs(calc["estimated_absorbed_dose_mgy"] - expected_mgy) < 1e-5
    assert calc["estimated_equivalent_dose_msv"] == round(expected_mgy * 3.05, 5)

# ----------------- TEST-09: API Route Integration ----------------- #
@pytest.mark.asyncio
async def test_api_terrain_analysis_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "coordinates": [[77.45, 18.38], [77.46, 18.39]],
            "radius_meters": 100.0,
            "max_slope_threshold_deg": 15.0
        }
        response = await client.post("/api/intelligence/terrain-analysis", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "slope_max_deg" in data["measurements"]
        assert len(data["evidence"]) > 0
        assert "interpretation" in data

# ----------------- TEST-10: Radiation Provenance Disambiguation ----------------- #
@pytest.mark.asyncio
async def test_api_radiation_endpoint_provenance():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"eva_duration_hours": 3.0}
        response = await client.post("/api/intelligence/radiation-estimation", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        # Verify Gale Crater reference disambiguation
        prov = data["data_provenance"]["primary_dataset"]
        assert "Gale Reference" in prov
        assert any("Gale Crater" in lim for lim in data["limitations"])

# ----------------- TEST-11: Master Mission Aggregate Endpoint ----------------- #
@pytest.mark.asyncio
async def test_api_master_mission_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "route_coordinates": [[77.44, 18.41], [77.45, 18.42], [77.46, 18.43]],
            "stop_count": 2,
            "stop_duration_minutes": 20.0,
            "suit_time_budget_hours": 4.0,
            "sol": 950
        }
        response = await client.post("/api/intelligence/mission-analysis", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["analysis_type"] == "mission_aggregate_analysis"
        assert "mission_go_status" in data["classification"]
        assert len(data["evidence"]) > 0
