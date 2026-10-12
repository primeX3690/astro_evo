# 14_collision_avoidance/conjunction_screening.py
import numpy as np

class RealTimeSSAScreening:
    def __init__(self, warning_threshold_km: float = 5.0):
        self.warning_threshold_km = warning_threshold_km

    def compute_miss_distance(self, sat_pos: np.ndarray, debris_pos: np.ndarray) -> float:
        """Calculates Euclidean distance between satellite and debris."""
        diff = sat_pos - debris_pos
        return float(np.linalg.norm(diff))

    def evaluate_conjunction(self, sat_state: dict, debris_state: dict) -> dict:
        pos_sat = np.array(sat_state['position'])
        pos_deb = np.array(debris_state['position'])
        miss_dist = self.compute_miss_distance(pos_sat, pos_deb)
        
        risk_level = "GREEN"
        if miss_dist <= self.warning_threshold_km:
            risk_level = "CRITICAL_RED" if miss_dist < 1.0 else "YELLOW"

        return {
            "miss_distance_km": round(miss_dist, 4),
            "risk_level": risk_level,
            "action_required": risk_level != "GREEN"
        }

def screen_conjunction(sat_state, debris_state, warning_threshold_km=5.0):
    """Compatibility wrapper for test_module14.py"""
    screener = RealTimeSSAScreening(warning_threshold_km=warning_threshold_km)
    return screener.evaluate_conjunction(sat_state, debris_state)
