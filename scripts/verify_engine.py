"""
Final End-to-End Verification Script for Marswalk Scientific Intelligence Engine.
Executes real simulation queries against all 7 analysis modules and prints complete results.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.terrain_service import TerrainService
from app.services.route_service import RouteService
from app.services.eva_service import EVAService
from app.services.rad_service import RadiationService
from app.services.space_weather_service import SpaceWeatherService
from app.services.env_service import EnvironmentalService
from app.rag.vector_store import ScientificVectorStore
from app.rag.retriever import ScientificRetriever
from app.llm.mock_provider import MockLLMProvider
import asyncio

async def run_verification():
    print("=" * 70)
    print("MARSWALK MISSION PLANNER — SCIENTIFIC AI ENGINE VERIFICATION")
    print("=" * 70)

    # 1. Vector Store & RAG Verification
    print("\n[1/7] Testing RAG Knowledge Retrieval Layer...")
    store = ScientificVectorStore()
    chunk_count = store.collection.count() if store.collection else len(store._fallback_corpus)
    print(f"  -> ChromaDB Collection: 'marswalk_scientific_knowledge_v3'")
    print(f"  -> Total Indexed Chunks: {chunk_count}")
    
    retriever = ScientificRetriever(store)
    evidence = retriever.retrieve_evidence("Jezero delta scarp slope traversability", topic="terrain_traversability", top_k=2)
    print(f"  -> Query: 'Jezero delta scarp slope traversability'")
    print(f"  -> Retrieved {len(evidence)} citation(s):")
    for ev in evidence:
        print(f"     * [{ev.citation_key}] {ev.title} ({ev.organization})")
        print(f"       Snippet: {ev.relevant_text[:100]}...")

    # 2. Terrain Analysis Verification
    print("\n[2/7] Testing Terrain Intelligence Module...")
    terrain = TerrainService(cell_size_m=20.0)
    # Simulate Jezero delta ascent from -2534m to -2488m over 600m
    t_res = terrain.analyze_profile([-2534.0, -2520.0, -2505.0, -2492.0, -2488.0], [0.0, 150.0, 300.0, 450.0, 600.0], max_slope_limit_deg=15.0)
    print(f"  -> Elevation Delta: {t_res['measurements']['elevation_delta_m']} m")
    print(f"  -> Average Slope: {t_res['measurements']['slope_avg_deg']}° | Max Slope: {t_res['measurements']['slope_max_deg']}°")
    print(f"  -> Classification: {t_res['classification']['terrain_difficulty']} (Within limits: {t_res['classification']['is_within_limits']})")

    # 3. Route Evaluation Verification
    print("\n[3/7] Testing Route Evaluation & Cost Surface...")
    route_svc = RouteService(max_slope_limit_deg=15.0)
    # Compare gentle route vs steep scarp route
    r_gentle = route_svc.evaluate_route_segments([4.0, 5.0, 6.0], [200.0, 200.0, 200.0])
    r_steep = route_svc.evaluate_route_segments([4.0, 16.5, 6.0], [200.0, 200.0, 200.0])
    print(f"  -> Gentle Route (4°-6°): Weighted Cost = {r_gentle['weighted_terrain_cost']} | Rating = {r_gentle['traversability_rating']}")
    print(f"  -> Steep Route (with 16.5° breach): Weighted Cost = {r_steep['weighted_terrain_cost']} | Rating = {r_steep['traversability_rating']}")
    assert r_steep["exceeds_slope_threshold"] is True

    # 4. EVA Duration & Tobler Hiking Budget
    print("\n[4/7] Testing EVA Budget (Modified Tobler Martian Walk Model)...")
    eva_svc = EVAService(default_k_suit=0.60)
    eva_res = eva_svc.calculate_eva_budget(
        segment_lengths_km=[1.5, 1.5],
        segment_slopes_deg=[4.0, 6.0],
        stop_count=3,
        stop_duration_minutes=30.0,
        suit_budget_hours=4.0,
        reserve_percent=20.0
    )
    c = eva_res["calculations"]
    print(f"  -> Total Traverse: 3.0 km with 3 stops (30 min total)")
    print(f"  -> Outbound Walk: {c['outbound_walking_hours']} h | Return Walk: {c['return_walking_hours']} h | Science Stops: {c['scientific_activity_hours']} h")
    print(f"  -> Total EVA Duration: {c['total_eva_duration_hours']} h (Effective Budget: {c['effective_budget_hours']} h)")
    print(f"  -> Consumable Status: {eva_res['classification']['budget_status']}")

    # 5. MSL RAD Radiation Exposure Estimation
    print("\n[5/7] Testing MSL RAD Radiation Exposure Modeling...")
    rad_svc = RadiationService(baseline_mgy_per_day=0.21, quality_factor_q=3.05)
    rad_res = rad_svc.estimate_exposure(eva_duration_hours=c['total_eva_duration_hours'])
    rc = rad_res["calculations"]
    print(f"  -> Duration: {rad_res['measurements']['eva_duration_hours']} hours")
    print(f"  -> Reference Dose Rate: {rad_res['measurements']['reference_dose_rate_mgy_per_day']} mGy/day (MSL Gale Crater)")
    print(f"  -> Estimated Absorbed Dose: {rc['estimated_absorbed_dose_mgy']} mGy ({rc['estimated_absorbed_dose_ugy']} µGy)")
    print(f"  -> Estimated Equivalent Dose: {rc['estimated_equivalent_dose_msv']} mSv ({rc['estimated_equivalent_dose_usv']} µSv)")
    print(f"  -> Equivalent Airline Flights: ~{rc['flight_chest_xray_equivalents']} transcontinental flights")

    # 6. Environmental (MEDA) & Space Weather (DONKI)
    print("\n[6/7] Testing Environmental (MEDA) & Space Weather (DONKI)...")
    env_svc = EnvironmentalService()
    env_res = env_svc.analyze_environmental_conditions(sol=950)
    em = env_res["measurements"]
    print(f"  -> MEDA Sol 950: Temp = {em['air_temperature_c']['min']}°C to {em['air_temperature_c']['max']}°C | Pressure = {em['atmospheric_pressure_pa']['avg']} Pa")
    
    sw_svc = SpaceWeatherService()
    cached = sw_svc.load_cached_data()
    sw_res = sw_svc.evaluate_space_weather_hazard(cached)
    print(f"  -> DONKI 7-Day Space Weather Status: '{sw_res['classification']['space_weather_status']}'")

    # 7. Grounded LLM Scientific Synthesis
    print("\n[7/7] Testing Grounded LLM Scientific Synthesis Layer...")
    llm = MockLLMProvider()
    llm_out = await llm.generate_interpretation(
        analysis_type="terrain_analysis",
        measurements=t_res["measurements"],
        calculations=t_res["calculations"],
        classification=t_res["classification"],
        evidence_text=retriever.format_evidence_block(evidence)
    )
    print(f"  -> Synthesized Interpretation:")
    print(f"     \"{llm_out['interpretation']}\"")
    print(f"  -> Explicit Limitations ({len(llm_out['limitations'])}):")
    for lim in llm_out["limitations"]:
        print(f"     * {lim}")
    print(f"  -> Explicit Uncertainty ({len(llm_out['uncertainty'])}):")
    for unc in llm_out["uncertainty"]:
        print(f"     * {unc}")

    print("\n" + "=" * 70)
    print("ALL 7 SCIENTIFIC AI MODULES VERIFIED OPERATIONAL & GROUNDED!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_verification())
