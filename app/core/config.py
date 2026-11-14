"""
Core Configuration and Scientific Constants for Marswalk Scientific AI Engine.
"""
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "documents"
RASTERS_DIR = DATA_DIR / "rasters"
CACHE_DIR = DATA_DIR / "cache"
CHROMA_DIR = DATA_DIR / "chroma_db"

class Settings(BaseSettings):
    # Mission Terrain & Pathfinding Parameters
    MAX_SLOPE_DEG: float = 15.0  # Slope limit in degrees before cell is classified impassable
    CAUTION_SLOPE_DEG: float = 10.0  # Slope threshold triggering caution flag
    
    # Martian Astronaut Mobility Parameters (Modified Tobler Hiking Model)
    TOBLER_BASE_VELOCITY_KMH: float = 6.0
    TOBLER_EXP_FACTOR: float = 3.5
    TOBLER_SLOPE_OFFSET: float = 0.05
    DEFAULT_K_SUIT: float = 0.60  # Spacesuit encumbrance & 0.38g stride dynamic penalty factor
    DEFAULT_RESERVE_PERCENT: float = 20.0  # Percentage of life-support budget reserved for safety
    
    # Radiation Parameters (MSL Curiosity RAD Reference)
    RAD_GALE_BASELINE_MGY_DAY: float = 0.21  # Measured surface GCR absorbed dose rate at Gale Crater
    RAD_QUALITY_FACTOR_Q: float = 3.0  # Mean radiation quality factor for Martian surface GCR
    
    # Space Weather Parameters (NASA CCMC DONKI)
    DONKI_API_URL: str = "https://api.nasa.gov/DONKI"
    DONKI_LOOKBACK_DAYS: int = 7
    DONKI_SEP_FLUX_THRESHOLD_PFU: float = 10.0  # Proton flux units at >10 MeV triggering caution
    
    # Jezero Crater Geographic Boundaries (IAU 2000 Mars Coordinate Frame)
    JEZERO_LAT_MIN: float = 18.20
    JEZERO_LAT_MAX: float = 18.60
    JEZERO_LON_MIN: float = 77.20
    JEZERO_LON_MAX: float = 77.80
    
    # Vector DB & Embeddings
    CHROMA_PERSIST_DIR: str = str(CHROMA_DIR)
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    RAG_TOP_K: int = 4
    RAG_SIMILARITY_THRESHOLD: float = 0.60
    
    # LLM Settings
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    FALLBACK_TO_MOCK_LLM: bool = True  # Allows offline/mock testing without live API keys
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
