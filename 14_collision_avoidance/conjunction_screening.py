# 14_collision_avoidance/conjunction_screening.py

import numpy as np


class RealTimeSSAScreening:
    def __init__(self, warning_threshold_km: float = 5.0):
        self.warning_threshold_km = warning_threshold_km

    def compute_miss_distance(
        self,
        sat_pos: np.ndarray,
        debris_pos: np.ndarray
    ) -> float:
        """Calculate Euclidean distance between satellite and debris."""
        diff = sat_pos - debris_pos
        return float(np.linalg.norm(diff))

    def evaluate_conjunction(
        self,
        sat_state: dict,
        debris_state: dict
    ) -> dict:
        pos_sat = np.array(sat_state["position"])
        pos_deb = np.array(debris_state["position"])

        miss_dist = self.compute_miss_distance(pos_sat, pos_deb)

        risk_level = "GREEN"

        if miss_dist <= self.warning_threshold_km:
            risk_level = "CRITICAL_RED" if miss_dist < 1.0 else "YELLOW"

        return {
            "miss_distance_km": round(miss_dist, 4),
            "risk_level": risk_level,
            "action_required": risk_level != "GREEN",
        }


def screen_conjunction(
    rel_pos0_km,
    rel_vel0_kms=None,
    n_rad_s=None,
    duration_s=None,
    hard_body_radius_km=0.02,
    safety_buffer_km=1.0,
    warning_threshold_km=None,
):
    """Screen relative motion and estimate the closest approach."""

    # Backward compatibility with the older dictionary-based API.
    if isinstance(rel_pos0_km, dict) and isinstance(rel_vel0_kms, dict):
        threshold = (
            n_rad_s if n_rad_s is not None else 5.0
        )

        screener = RealTimeSSAScreening(
            warning_threshold_km=float(threshold)
        )

        return screener.evaluate_conjunction(
            rel_pos0_km,
            rel_vel0_kms,
        )

    if (
        rel_vel0_kms is None
        or n_rad_s is None
        or duration_s is None
    ):
        raise TypeError(
            "Vector API requires rel_pos0_km, rel_vel0_kms, "
            "n_rad_s, duration_s"
        )

    rel_pos0_km = np.asarray(rel_pos0_km, dtype=float)
    rel_vel0_kms = np.asarray(rel_vel0_kms, dtype=float)

    if rel_pos0_km.shape != (3,) or rel_vel0_kms.shape != (3,):
        raise ValueError(
            "Relative position and velocity must each be "
            "3-element vectors"
        )

    if duration_s <= 0 or n_rad_s <= 0:
        raise ValueError(
            "duration_s and n_rad_s must be positive"
        )

    # Reuse the project's Clohessy-Wiltshire analytical propagator.
    import os
    import sys

    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "4_neural_controller",
        ),
    )

    from docking_env import cw_analytic_propagate

    state0 = np.concatenate(
        [rel_pos0_km, rel_vel0_kms]
    )

    # Coarse search at 1-second intervals.
    coarse_times = np.arange(
        0.0,
        duration_s + 1.0,
        1.0,
    )

    coarse_times = np.minimum(
        coarse_times,
        duration_s,
    )

    coarse_distances = np.array([
        np.linalg.norm(
            cw_analytic_propagate(
                state0,
                float(t),
                n_rad_s,
            )[:3]
        )
        for t in coarse_times
    ])

    idx = int(np.argmin(coarse_distances))

    lo = max(
        0.0,
        coarse_times[idx] - 1.0,
    )

    hi = min(
        float(duration_s),
        coarse_times[idx] + 1.0,
    )

    # Refine the closest approach at 0.1-second intervals.
    fine_times = np.arange(
        lo,
        hi + 0.05,
        0.1,
    )

    fine_distances = np.array([
        np.linalg.norm(
            cw_analytic_propagate(
                state0,
                float(t),
                n_rad_s,
            )[:3]
        )
        for t in fine_times
    ])

    best_idx = int(np.argmin(fine_distances))

    tca_s = float(fine_times[best_idx])
    miss_distance_km = float(
        fine_distances[best_idx]
    )

    threshold_km = float(
        warning_threshold_km
        if warning_threshold_km is not None
        else hard_body_radius_km + safety_buffer_km
    )

    return {
        "tca_s": tca_s,
        "miss_distance_km": miss_distance_km,
        "threshold_km": threshold_km,
        "is_conjunction": (
            miss_distance_km <= threshold_km
        ),
        "hard_body_radius_km": float(
            hard_body_radius_km
        ),
        "safety_buffer_km": float(
            safety_buffer_km
        ),
    }
