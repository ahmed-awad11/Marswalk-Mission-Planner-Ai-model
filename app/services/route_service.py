"""
Deterministic Route Evaluation and Pathfinding Module.
Integrates with scikit-image route_through_array to compute least-cost paths and evaluate route traversability.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
from skimage.graph import route_through_array

class RouteService:
    def __init__(self, max_slope_limit_deg: float = 15.0):
        self.max_slope_limit_deg = max_slope_limit_deg

    def build_cost_surface(self, slope_grid: np.ndarray) -> np.ndarray:
        """
        Calculates cell-by-cell traversability cost:
        cost = 1 + (slope_deg / MAX_SLOPE)**2
        Cells above MAX_SLOPE or containing NaN are blocked (inf or penalty 1e6).
        """
        cost_surface = np.ones_like(slope_grid, dtype=np.float64)
        
        # Valid cells
        valid_mask = ~np.isnan(slope_grid)
        slope_clamped = np.where(valid_mask, slope_grid, 999.0)
        
        # Polynomial cost function
        cost_surface[valid_mask] = 1.0 + (slope_clamped[valid_mask] / self.max_slope_limit_deg) ** 2
        
        # Impassable blocked cells
        blocked_mask = (~valid_mask) | (slope_grid > self.max_slope_limit_deg)
        cost_surface[blocked_mask] = 1e6  # Impassable obstacle cost
        
        return cost_surface

    def find_least_cost_path(self, cost_surface: np.ndarray, start_idx: Tuple[int, int], end_idx: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], float]:
        """
        Uses scikit-image route_through_array to find the minimal cost route.
        """
        indices, weight = route_through_array(
            cost_surface,
            start=start_idx,
            end=end_idx,
            fully_connected=True,
            geometric=True
        )
        return indices, float(weight)

    def evaluate_route_segments(self, segment_slopes: List[float], segment_lengths_m: List[float]) -> Dict[str, Any]:
        """
        Evaluates a specified sequence of route segments.
        """
        slopes = np.array(segment_slopes, dtype=np.float64)
        lengths = np.array(segment_lengths_m, dtype=np.float64)
        total_len = float(np.sum(lengths))
        
        # Calculate segment costs
        costs = np.where(
            slopes <= self.max_slope_limit_deg,
            1.0 + (slopes / self.max_slope_limit_deg) ** 2,
            100.0  # Excessive penalty for slope breach
        )
        weighted_cost = float(np.sum(costs * lengths) / max(total_len, 1e-3))
        
        steep_segments = slopes > self.max_slope_limit_deg
        steep_len = float(np.sum(lengths[steep_segments]))
        
        if np.any(steep_segments):
            traversability = "Hazardous: Slope Limit Exceeded"
            recommendation = "Reroute required: one or more segments exceed the 15° operational slope boundary."
            exceeds_threshold = True
        elif weighted_cost > 1.4:
            traversability = "Challenging Traversal"
            recommendation = "Elevated terrain cost; consider traversing along contour lines."
            exceeds_threshold = False
        else:
            traversability = "Nominal Traversal"
            recommendation = "Low terrain resistance; terrain cost is nominal."
            exceeds_threshold = False
            
        return {
            "total_route_length_m": round(total_len, 2),
            "steep_length_m": round(steep_len, 2),
            "weighted_terrain_cost": round(weighted_cost, 3),
            "exceeds_slope_threshold": exceeds_threshold,
            "traversability_rating": traversability,
            "tactical_recommendation": recommendation,
        }
