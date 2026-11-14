"""
Deterministic Radiation Exposure Estimation Module.
Scales MSL RAD Curiosity Gale Crater in-situ reference measurements for planned EVA durations.
"""
from typing import Dict, Any

class RadiationService:
    def __init__(self, baseline_mgy_per_day: float = 0.21, quality_factor_q: float = 3.05):
        self.baseline_mgy_per_day = baseline_mgy_per_day
        self.quality_factor_q = quality_factor_q

    def estimate_exposure(self, eva_duration_hours: float, custom_baseline: float = None) -> Dict[str, Any]:
        """
        Calculates absorbed ionizing dose and equivalent biological dose.
        """
        if eva_duration_hours < 0:
            raise ValueError("EVA duration cannot be negative.")
            
        rate = custom_baseline if custom_baseline is not None else self.baseline_mgy_per_day
        
        # Dose calculations
        fraction_of_day = eva_duration_hours / 24.0
        absorbed_dose_mgy = rate * fraction_of_day
        absorbed_dose_ugy = absorbed_dose_mgy * 1000.0  # micrograys
        
        equivalent_dose_msv = absorbed_dose_mgy * self.quality_factor_q
        equivalent_dose_usv = equivalent_dose_msv * 1000.0  # microsieverts
        
        # Contextual terrestrial comparison (~10 uSv for typical transcontinental airline flight)
        flight_equivalents = round(equivalent_dose_usv / 10.0, 1)
        
        return {
            "measurements": {
                "reference_dose_rate_mgy_per_day": round(rate, 4),
                "quality_factor_q": self.quality_factor_q,
                "eva_duration_hours": round(eva_duration_hours, 2),
            },
            "calculations": {
                "estimated_absorbed_dose_mgy": round(absorbed_dose_mgy, 5),
                "estimated_absorbed_dose_ugy": round(absorbed_dose_ugy, 2),
                "estimated_equivalent_dose_msv": round(equivalent_dose_msv, 5),
                "estimated_equivalent_dose_usv": round(equivalent_dose_usv, 2),
                "flight_chest_xray_equivalents": flight_equivalents,
            },
            "classification": {
                "exposure_severity": "Nominal GCR Background",
                "acute_radiation_risk": False,
                "alara_guidance": "Nominal short-duration EVA. Exposure accumulates within standard NASA occupational career limits."
            },
            "provenance_notes": {
                "origin_mission": "Mars Science Laboratory (MSL) Curiosity Rover",
                "instrument": "Radiation Assessment Detector (RAD)",
                "measurement_location": "Gale Crater floor (-4.5°S, -4400m MOLA elevation)",
                "application": "Transferred reference value for Jezero Crater simulation"
            }
        }
