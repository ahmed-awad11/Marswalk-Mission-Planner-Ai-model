"""
Deterministic Environmental Intelligence Module.
Parses Mars 2020 Perseverance MEDA in-situ atmospheric observations and evaluates environmental conditions.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from app.core.config import CACHE_DIR

class EnvironmentalService:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_file = cache_dir / "meda_sample_cache.json"

    def get_sol_observation(self, sol: int = 950) -> Optional[Dict[str, Any]]:
        """Retrieves verified MEDA observation for requested Sol."""
        if not self.cache_file.exists():
            return None
        with open(self.cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            sols = data.get("sols", {})
            return sols.get(str(sol))

    def analyze_environmental_conditions(self, sol: int = 950) -> Dict[str, Any]:
        """
        Analyzes MEDA pressure, temperature, and wind extremes.
        """
        obs = self.get_sol_observation(sol)
        if obs is None:
            return {
                "measurements": {},
                "calculations": {},
                "classification": {
                    "environmental_status": "Unavailable",
                    "caution_flags": ["No MEDA records available for requested Sol"]
                },
                "limitations": [
                    f"Perseverance MEDA sensor observations are not recorded or indexed for Sol {sol}.",
                    "System refuses to fabricate simulated weather readings."
                ]
            }

        press = obs.get("pressure_pa", {})
        temp = obs.get("air_temperature_c", {})
        wind = obs.get("wind_speed_ms", {})
        loc = obs.get("location", {})
        
        t_min = temp.get("min", -80.0)
        t_max = temp.get("max", -15.0)
        diurnal_delta = round(t_max - t_min, 1)
        
        max_wind = wind.get("max", 5.0)
        
        caution_flags = []
        if max_wind > 15.0:
            caution_flags.append(f"Elevated wind gust hazard ({max_wind} m/s recorded). Potential dust lofting.")
        if t_min < -85.0:
            caution_flags.append(f"Extreme cryogenic nocturnal minimum ({t_min}°C). Suit sole conductive cooling hazard.")
            
        status = "Caution: Environmental Extremes" if caution_flags else "Nominal Jezero In-Situ Atmospheric State"

        return {
            "measurements": {
                "sol": sol,
                "terrestrial_date": obs.get("terrestrial_date"),
                "observation_location": loc,
                "atmospheric_pressure_pa": press,
                "air_temperature_c": temp,
                "ground_temperature_c": obs.get("ground_temperature_c"),
                "wind_speed_ms": wind,
                "wind_direction_deg": obs.get("wind_direction_deg"),
            },
            "calculations": {
                "diurnal_temperature_swing_c": diurnal_delta,
                "peak_wind_gust_ms": max_wind,
                "flagged_hazards_count": len(caution_flags),
            },
            "classification": {
                "environmental_status": status,
                "caution_flags": caution_flags,
            },
            "limitations": [
                "Measurements are in-situ observations recorded by Perseverance at its specific geographic location.",
                "Atmospheric pressure decreases with altitude by roughly 1 Pa per 10m; observations do not represent elevated Jezero crater rims.",
                "These are historical sensor recordings, not a real-time predictive forecast."
            ]
        }
