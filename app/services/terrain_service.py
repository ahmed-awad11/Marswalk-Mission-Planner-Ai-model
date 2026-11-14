"""
Deterministic Terrain Analysis Module.
Calculates elevation differences, 2D slope gradients, terrain profiles, and difficulty classifications.
"""
from typing import Dict, Any, List, Tuple
import numpy as np

class TerrainService:
    def __init__(self, cell_size_m: float = 20.0):
        self.cell_size_m = cell_size_m  # Default CTX DEM 20m resolution

    def compute_slope_horn(self, elevation_grid: np.ndarray) -> np.ndarray:
        """
        Calculates terrain slope in degrees using Horn's 3x3 finite-difference method.
        """
        rows, cols = elevation_grid.shape
        if rows < 3 or cols < 3:
            raise ValueError("Elevation grid must be at least 3x3 to compute Horn's slope.")
        
        # Calculate gradients using central differences
        dz_dx = np.zeros_like(elevation_grid, dtype=np.float64)
        dz_dy = np.zeros_like(elevation_grid, dtype=np.float64)
        
        # Interior points
        dz_dx[1:-1, 1:-1] = (
            (elevation_grid[0:-2, 2:] + 2 * elevation_grid[1:-1, 2:] + elevation_grid[2:, 2:]) -
            (elevation_grid[0:-2, 0:-2] + 2 * elevation_grid[1:-1, 0:-2] + elevation_grid[2:, 0:-2])
        ) / (8.0 * self.cell_size_m)
        
        dz_dy[1:-1, 1:-1] = (
            (elevation_grid[2:, 0:-2] + 2 * elevation_grid[2:, 1:-1] + elevation_grid[2:, 2:]) -
            (elevation_grid[0:-2, 0:-2] + 2 * elevation_grid[0:-2, 1:-1] + elevation_grid[0:-2, 2:])
        ) / (8.0 * self.cell_size_m)
        
        # Slope angle in degrees
        rise_run = np.sqrt(dz_dx**2 + dz_dy**2)
        slope_deg = np.arctan(rise_run) * (180.0 / np.pi)
        return slope_deg

    def analyze_profile(self, elevations: List[float], distances_m: List[float], max_slope_limit_deg: float = 15.0) -> Dict[str, Any]:
        """
        Analyzes a 1D elevation profile along a transect.
        """
        if not elevations or len(elevations) < 2:
            raise ValueError("At least two elevation points are required to compute profile slope.")
        
        elev_arr = np.array(elevations, dtype=np.float64)
        dist_arr = np.array(distances_m, dtype=np.float64)
        
        elev_min = float(np.min(elev_arr))
        elev_max = float(np.max(elev_arr))
        elev_delta = float(elev_max - elev_min)
        
        # Segment-by-segment slope
        d_elev = np.diff(elev_arr)
        d_dist = np.diff(dist_arr)
        
        # Avoid division by zero
        d_dist = np.where(d_dist <= 0, 1e-3, d_dist)
        
        segment_slopes = np.abs(np.arctan(d_elev / d_dist) * (180.0 / np.pi))
        
        avg_slope = float(np.mean(segment_slopes))
        max_slope = float(np.max(segment_slopes))
        
        gain = float(np.sum(d_elev[d_elev > 0]))
        loss = float(np.abs(np.sum(d_elev[d_elev < 0])))
        
        steep_mask = segment_slopes > max_slope_limit_deg
        steep_count = int(np.sum(steep_mask))
        steep_distance_m = float(np.sum(d_dist[steep_mask]))
        total_dist_m = float(dist_arr[-1] - dist_arr[0])
        
        # Difficulty Classification
        if max_slope > max_slope_limit_deg:
            difficulty = "Severe / Exceeds Slope Limit"
            hazard_level = f"Caution: {steep_count} segment(s) exceed {max_slope_limit_deg}° threshold"
            is_within_limits = False
        elif max_slope > 10.0:
            difficulty = "Challenging"
            hazard_level = "Caution: Significant slopes detected (>10°)"
            is_within_limits = True
        elif avg_slope > 5.0:
            difficulty = "Moderate"
            hazard_level = "Nominal terrain with gentle inclines"
            is_within_limits = True
        else:
            difficulty = "Flat / Easy"
            hazard_level = "Nominal low-angle terrain"
            is_within_limits = True
            
        return {
            "measurements": {
                "elevation_min_m": round(elev_min, 2),
                "elevation_max_m": round(elev_max, 2),
                "elevation_delta_m": round(elev_delta, 2),
                "elevation_gain_m": round(gain, 2),
                "elevation_loss_m": round(loss, 2),
                "slope_avg_deg": round(avg_slope, 2),
                "slope_max_deg": round(max_slope, 2),
            },
            "calculations": {
                "total_distance_m": round(total_dist_m, 2),
                "steep_segments_count": steep_count,
                "steep_distance_m": round(steep_distance_m, 2),
                "steep_percentage": round((steep_distance_m / max(total_dist_m, 1e-3)) * 100.0, 1),
            },
            "classification": {
                "terrain_difficulty": difficulty,
                "hazard_level": hazard_level,
                "is_within_limits": is_within_limits,
            }
        }
