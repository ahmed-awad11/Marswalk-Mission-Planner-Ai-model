"""
Unified Pydantic Models and Data Contracts for Marswalk Scientific AI Engine.
Strictly separates:
- raw inputs
- deterministic measurements (from data rasters/sources)
- deterministic calculations (from Python math)
- classification flags
- grounded scientific interpretation (from LLM)
- evidence citations (from RAG)
- limitations & uncertainty
- data provenance
"""
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class AnalysisType(str, Enum):
    TERRAIN = "terrain_analysis"
    ROUTE = "route_evaluation"
    EVA = "eva_estimation"
    ENVIRONMENTAL = "environmental_analysis"
    RADIATION = "radiation_estimation"
    SPACE_WEATHER = "space_weather_analysis"
    MISSION_AGGREGATE = "mission_aggregate_analysis"

class EvidenceItem(BaseModel):
    citation_key: str = Field(..., description="Unique reference key, e.g. REF-1")
    title: str = Field(..., description="Official title of the NASA document or paper")
    source_url: str = Field(..., description="DOI or official NASA PDS / NTRS URL")
    organization: str = Field("NASA / USGS / JPL", description="Publishing body")
    relevant_text: str = Field(..., description="Verbatim or summarized relevant scientific evidence")

class DataProvenance(BaseModel):
    primary_dataset: str
    sensor_or_mission: str
    spatial_resolution: Optional[str] = None
    crs: Optional[str] = None
    observation_date_or_sol: Optional[str] = None
    processing_timestamp: str

class BaseIntelligenceResponse(BaseModel):
    analysis_id: str
    analysis_type: AnalysisType
    status: str = "success"
    inputs: Dict[str, Any] = Field(default_factory=dict)
    measurements: Dict[str, Any] = Field(default_factory=dict, description="Raw physical measurements extracted from rasters or sensors")
    calculations: Dict[str, Any] = Field(default_factory=dict, description="Purely deterministic Python calculations")
    classification: Dict[str, Any] = Field(default_factory=dict, description="Objective rule-based hazard classifications")
    interpretation: str = Field(..., description="Grounded scientific interpretation synthesized by LLM")
    evidence: List[EvidenceItem] = Field(default_factory=list, description="Citations retrieved from authoritative RAG corpus")
    limitations: List[str] = Field(default_factory=list, description="Known physical and data limitations")
    uncertainty: List[str] = Field(default_factory=list, description="Sources of measurement and modeling uncertainty")
    data_provenance: DataProvenance
    timestamp: str

# ----------------- Specific Request Schemas ----------------- #

class TerrainAnalysisRequest(BaseModel):
    coordinates: List[List[float]] = Field(..., description="List of [lon, lat] pairs or single point")
    radius_meters: Optional[float] = Field(100.0, description="Sampling radius around point")
    max_slope_threshold_deg: Optional[float] = Field(15.0, description="Project-defined slope limit")

class RouteEvaluationRequest(BaseModel):
    route_coordinates: List[List[float]] = Field(..., description="Array of [lon, lat] coordinates defining proposed route")
    max_slope_threshold_deg: Optional[float] = Field(15.0, description="Project-defined maximum traversable slope")
    compute_alternative: Optional[bool] = Field(True, description="Whether to compute least-cost alternative path")

class EVABudgetRequest(BaseModel):
    route_distance_km: float = Field(..., description="Total route distance in kilometers")
    cumulative_elevation_gain_m: float = Field(0.0, description="Ascent meters")
    cumulative_elevation_loss_m: float = Field(0.0, description="Descent meters")
    average_slope_deg: float = Field(0.0, description="Route average slope angle")
    stop_count: int = Field(0, description="Number of scientific waypoints / sample stations")
    stop_duration_minutes: float = Field(0.0, description="Total minutes allocated for stops")
    suit_time_budget_hours: float = Field(4.0, description="Total life support consumable budget")
    reserve_margin_percent: float = Field(20.0, description="Mandatory safety return margin percentage")
    suit_factor_k: Optional[float] = Field(0.60, description="Spacesuit penalty factor for Tobler hiking formula")

class EnvironmentalAnalysisRequest(BaseModel):
    sol: Optional[int] = Field(950, description="Perseverance Sol number")
    include_historical_ranges: Optional[bool] = Field(True, description="Compare against seasonal baseline")

class RadiationEstimationRequest(BaseModel):
    eva_duration_hours: float = Field(..., description="Planned duration of extravehicular activity in hours")
    reference_dose_mgy_day: Optional[float] = Field(0.21, description="MSL RAD Gale Crater baseline dose rate")

class SpaceWeatherRequest(BaseModel):
    lookback_days: Optional[int] = Field(7, description="Number of recent days to inspect for solar events")

class MissionAnalysisRequest(BaseModel):
    route_coordinates: List[List[float]]
    stop_count: int = 2
    stop_duration_minutes: float = 30.0
    suit_time_budget_hours: float = 4.0
    sol: Optional[int] = 950
