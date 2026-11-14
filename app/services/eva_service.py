"""
Deterministic EVA Duration and Mission Budget Engine.
Implements the Martian-adapted Tobler Hiking Model and life-support margin budgeting.
"""
from typing import Dict, Any, List
import numpy as np

class EVAService:
    def __init__(
        self,
        base_velocity_kmh: float = 6.0,
        exp_factor: float = 3.5,
        slope_offset: float = 0.05,
        default_k_suit: float = 0.60
    ):
        self.base_velocity_kmh = base_velocity_kmh
        self.exp_factor = exp_factor
        self.slope_offset = slope_offset
        self.default_k_suit = default_k_suit

    def tobler_velocity_kmh(self, slope_deg: float, k_suit: float = None) -> float:
        """
        Modified Tobler hiking formula:
        v = 6 * exp(-3.5 * |tan(theta) + 0.05|) * k_suit
        """
        k = k_suit if k_suit is not None else self.default_k_suit
        rad = np.radians(slope_deg)
        slope_term = np.abs(np.tan(rad) + self.slope_offset)
        velocity = self.base_velocity_kmh * np.exp(-self.exp_factor * slope_term) * k
        # Clamp velocity to physiological minimum and maximum on Mars
        return float(np.clip(velocity, 0.2, 5.0))

    def calculate_eva_budget(
        self,
        segment_lengths_km: List[float],
        segment_slopes_deg: List[float],
        stop_count: int,
        stop_duration_minutes: float,
        suit_budget_hours: float = 4.0,
        reserve_percent: float = 20.0,
        k_suit: float = 0.60
    ) -> Dict[str, Any]:
        """
        Calculates complete EVA duration, walking velocities, and life-support margins.
        """
        lengths = np.array(segment_lengths_km, dtype=np.float64)
        slopes_out = np.array(segment_slopes_deg, dtype=np.float64)
        slopes_ret = -slopes_out  # Return trip reverses ascent/descent profile
        
        # Outbound velocities & times
        v_out = np.array([self.tobler_velocity_kmh(s, k_suit) for s in slopes_out])
        t_out_hours = np.sum(lengths / v_out)
        
        # Return velocities & times
        v_ret = np.array([self.tobler_velocity_kmh(s, k_suit) for s in slopes_ret])
        t_ret_hours = np.sum(lengths / v_ret)
        
        # Science stop time
        t_stops_hours = (stop_count * stop_duration_minutes) / 60.0
        
        # Total mission duration
        t_total_hours = float(t_out_hours + t_ret_hours + t_stops_hours)
        
        # Safety reserves
        required_reserve_hours = suit_budget_hours * (reserve_percent / 100.0)
        effective_budget_hours = suit_budget_hours - required_reserve_hours
        remaining_margin_hours = effective_budget_hours - t_total_hours
        
        budget_exceeded = t_total_hours > effective_budget_hours
        
        if t_total_hours > suit_budget_hours:
            status = "CRITICAL: Total EVA duration exceeds entire life-support capacity!"
            severity = "Critical"
        elif budget_exceeded:
            status = f"WARNING: Mission duration ({round(t_total_hours, 2)}h) encroaches on the {reserve_percent}% safety return reserve."
            severity = "Warning"
        else:
            status = "Nominal: Planned EVA duration operates within safe consumable margins."
            severity = "Nominal"
            
        return {
            "calculations": {
                "outbound_walking_hours": round(float(t_out_hours), 2),
                "return_walking_hours": round(float(t_ret_hours), 2),
                "scientific_activity_hours": round(float(t_stops_hours), 2),
                "total_eva_duration_hours": round(t_total_hours, 2),
                "configured_budget_hours": round(suit_budget_hours, 2),
                "required_reserve_hours": round(required_reserve_hours, 2),
                "effective_budget_hours": round(effective_budget_hours, 2),
                "remaining_margin_hours": round(remaining_margin_hours, 2),
                "average_outbound_velocity_kmh": round(float(np.mean(v_out)), 2),
            },
            "classification": {
                "budget_status": status,
                "budget_exceeded": budget_exceeded,
                "severity": severity,
            }
        }
