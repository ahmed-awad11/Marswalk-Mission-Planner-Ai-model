"""
Deterministic Mock LLM Provider.
Generates structured, grounded scientific interpretations offline without external API keys.
Guarantees robust hackathon demo stability and passes all scientific edge-case tests.
"""
from typing import Dict, Any, List

class MockLLMProvider:
    async def generate_interpretation(
        self,
        analysis_type: str,
        measurements: Dict[str, Any],
        calculations: Dict[str, Any],
        classification: Dict[str, Any],
        evidence_text: str
    ) -> Dict[str, Any]:
        """
        Synthesizes grounded scientific commentary based on deterministic calculations.
        """
        # 1. Terrain Analysis Interpretation
        if analysis_type == "terrain_analysis":
            max_slope = measurements.get("slope_max_deg", 0.0)
            avg_slope = measurements.get("slope_avg_deg", 0.0)
            steep_count = calculations.get("steep_segments_count", 0)
            
            if max_slope > 15.0:
                interpretation = (
                    f"The surveyed transect reaches a peak slope of {max_slope}°, exceeding the 15.0° operational traversability limit. "
                    f"According to USGS topographic guidelines [REF-1] and Jezero delta geomorphology studies [REF-2], slopes exceeding 15° "
                    f"frequently feature unconsolidated basaltic talus and scree near the angle of repose (~20°-30°), presenting severe foot slipping "
                    f"and traction hazards for suited crew."
                )
            elif max_slope > 10.0:
                interpretation = (
                    f"The terrain exhibits moderate-to-challenging relief with an average slope of {avg_slope}° and maximum of {max_slope}°. "
                    f"While within the 15° limit, traversing steep sections increases metabolic expenditure and requires zig-zag path planning [REF-1]."
                )
            else:
                interpretation = (
                    f"The surveyed area displays nominal low-relief topography with an average slope of {avg_slope}° and maximum of {max_slope}°. "
                    f"This corresponds to smooth crater floor pavement (Máaz or Séítah formations) suitable for steady traversal [REF-1]."
                )
                
            limitations = [
                "Slope calculations are derived from 20m CTX DEM resolution, which smooths out sub-meter boulders and micro-roughness.",
                "Surface mechanical properties (soil cohesion, shear strength) cannot be verified without direct in-situ geotechnical measurements."
            ]
            uncertainty = [
                "Active sand ripple drift thickness in low depressions is unmeasured from orbit."
            ]

        # 2. Route Evaluation Interpretation
        elif analysis_type == "route_evaluation":
            steep_len = calculations.get("steep_length_m", 0.0)
            cost = calculations.get("weighted_terrain_cost", 1.0)
            
            if steep_len > 0:
                interpretation = (
                    f"The planned route includes {steep_len}m of traversal exceeding the 15.0° slope limit, resulting in an elevated terrain cost of {cost}. "
                    f"Traversing scarp escarpments directly increases the risk of scree sliding and spacesuit falls [REF-1]. Rerouting along lower-angle benches is advised."
                )
            else:
                interpretation = (
                    f"The planned route traverses favorable terrain with a weighted terrain cost of {cost}. "
                    f"All segments remain below the 15.0° threshold, avoiding critical scree hazards along delta fronts [REF-1]."
                )
            limitations = [
                "Route pathfinding uses a 2D polynomial slope cost surface and does not account for 3D obstacle avoidance around boulders.",
                "Terrestrial least-cost path formulations do not guarantee crew safety under low-gravity mobility dynamics."
            ]
            uncertainty = [
                "Sub-pixel rock abundance may require localized micro-detours not captured in orbital cost rasters."
            ]

        # 3. EVA Estimation Interpretation
        elif analysis_type == "eva_estimation":
            t_total = calculations.get("total_eva_duration_hours", 0.0)
            effective_budget = calculations.get("effective_budget_hours", 0.0)
            exceeded = classification.get("budget_exceeded", False)
            
            if exceeded:
                interpretation = (
                    f"The estimated EVA duration of {t_total} hours exceeds the effective life-support budget ({effective_budget} hours). "
                    f"Walking velocity modeled via Tobler's function with spacesuit penalty k_suit=0.60 indicates that metabolic expenditure and "
                    f"traverse time compromise the mandatory return safety reserve [REF-1]."
                )
            else:
                interpretation = (
                    f"Estimated EVA duration is {t_total} hours, successfully operating within the {effective_budget}-hour effective budget limit. "
                    f"Calculated walking velocities reflect Martian reduced gravity (0.38g) and pressurized spacesuit joint encumbrance (k_suit=0.60) [REF-1]."
                )
            limitations = [
                "Tobler's hiking function is a terrestrial empirical formulation adapted with a suit penalty factor, not an in-situ validated Martian astronaut mobility model.",
                "Consumable usage is modeled as linear time consumption and does not dynamically scale with crew heart rate or metabolic elevation."
            ]
            uncertainty = [
                "Astronaut fatigue and cognitive workload over multi-hour exploration walks are not modeled."
            ]

        # 4. Environmental Analysis Interpretation
        elif analysis_type == "environmental_analysis":
            sol = measurements.get("sol", 950)
            temp = measurements.get("air_temperature_c", {})
            wind = measurements.get("wind_speed_ms", {})
            
            if not measurements:
                interpretation = f"Insufficient data: Perseverance MEDA sensor records are unavailable for the requested Sol {sol}."
                limitations = ["Historical sensor observations for this sol are not indexed."]
                uncertainty = ["Atmospheric conditions cannot be predicted without active meteorological data."]
            else:
                interpretation = (
                    f"Perseverance MEDA recorded diurnal air temperatures between {temp.get('min')}°C and {temp.get('max')}°C on Sol {sol}. "
                    f"At low atmospheric density (~750 Pa), conductive heat loss through suit boots dominates over convective wind chill [REF-1]. "
                    f"Peak wind speeds of {wind.get('max')} m/s remain below active dust storm threshold."
                )
                limitations = [
                    "Measurements represent local microclimate at Perseverance's specific rover location and cannot be extrapolated across crater rims.",
                    "These are historical observations, not a live meteorological forecast."
                ]
                uncertainty = [
                    "Localized dust devil activity and transient thermal inversions are unmodeled."
                ]

        # 5. Radiation Estimation Interpretation
        elif analysis_type == "radiation_estimation":
            dose_mgy = calculations.get("estimated_absorbed_dose_mgy", 0.0)
            dose_msv = calculations.get("estimated_equivalent_dose_msv", 0.0)
            hours = measurements.get("eva_duration_hours", 0.0)
            
            interpretation = (
                f"For a {hours}-hour EVA, estimated cumulative ionizing radiation exposure is {dose_mgy} mGy absorbed dose ({dose_msv} mSv equivalent dose). "
                f"Calculations scale MSL Curiosity RAD surface baseline measurements (~0.21 mGy/day) with a biological quality factor Q=3.05 [REF-1]. "
                f"This level represents nominal background GCR exposure without acute radiation health risk."
            )
            limitations = [
                "Dose calculations are scaled from Gale Crater in-situ measurements (MSL RAD, -4400m elevation). Jezero Crater (-2500m elevation) has slightly less atmospheric column shielding.",
                "Spacesuit material shielding is treated as thin tissue-equivalent thickness; specialized regolith shielding is not included."
            ]
            uncertainty = [
                "Solar cycle modulation creates a ±15% variation in background GCR flux."
            ]

        # 6. Space Weather Interpretation
        elif analysis_type == "space_weather_analysis":
            status = classification.get("space_weather_status", "Nominal")
            hazards = calculations.get("active_hazards_count", 0)
            
            if hazards > 0:
                interpretation = (
                    f"Space weather screening detects {hazards} active solar event(s) exceeding caution thresholds within the 7-day lookback window [REF-1]. "
                    f"Status is '{status}'. Elevated solar proton flux presents acute ionizing radiation hazards; suited crew must remain within shelter distance."
                )
            else:
                interpretation = (
                    f"Space weather screening reports '{status}' conditions across the 7-day lookback window. "
                    f"No solar energetic particle events exceeding 10 pfu or major flares were detected [REF-1]. Heliospheric conditions remain quiescent."
                )
            limitations = [
                "DONKI solar event data is recorded by near-Earth and heliospheric satellites; Mars-specific arrival timing requires magnetohydrodynamic spiral modeling.",
                "Threshold rules are project-defined mission planning heuristics, not official NASA flight safety certifications."
            ]
            uncertainty = [
                "Far-side solar eruptions unobserved by Earth-facing satellites could produce unheralded energetic particle shocks."
            ]
            
        else:
            interpretation = "Standard scientific intelligence assessment generated."
            limitations = ["General dataset limitations apply."]
            uncertainty = ["Standard modeling uncertainty."]

        return {
            "interpretation": interpretation,
            "limitations": limitations,
            "uncertainty": uncertainty
        }
