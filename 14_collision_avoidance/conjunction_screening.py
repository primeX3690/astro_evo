# 14_collision_avoidance/conjunction_screening.py
"""
Conjunction screening: predicts the closest approach between a primary
(chaser/own satellite) and a secondary object (debris/another satellite)
using the SAME Clohessy-Wiltshire relative-motion model already built
and verified in module 4 (docking_env.py) for proximity operations -
collision screening and docking are the same underlying relative-motion
physics, just opposite goals (avoid vs. achieve zero miss distance).

This is a real, standard technique: near-term conjunction assessment
(hours to a few orbits ahead) is commonly done in the CW/Hill relative
frame precisely because it's linear and fast, before falling back to a
full numerical propagation for longer-horizon or high-eccentricity cases.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "4_neural_controller"))
from docking_env import cw_analytic_propagate  # noqa: E402


def screen_conjunction(rel_pos0_km, rel_vel0_kms, n_rad_s, screen_duration_s,
                        hard_body_radius_km=0.02, safety_buffer_km=1.0, dt_s=1.0):
    """
    rel_pos0_km/rel_vel0_kms: secondary object's position/velocity relative
        to the primary, in the primary's Hill (RSW) frame, at t=0.
    n_rad_s: primary's orbital mean motion (rad/s) - sets the CW dynamics.
    Returns a dict with the predicted time of closest approach (TCA),
    miss distance, and whether this counts as a conjunction (miss
    distance below hard_body_radius + safety_buffer - the standard
    "combined hard-body radius" criterion used in real conjunction
    assessment, simplified here with a fixed safety buffer rather than
    a full covariance-based probability-of-collision calculation).
    """
    rel_state0 = np.concatenate([rel_pos0_km, rel_vel0_kms])
    times = np.arange(0, screen_duration_s, dt_s)
    distances = np.zeros(len(times))

    for i, t in enumerate(times):
        rel_state_t = cw_analytic_propagate(rel_state0, t, n_rad_s)
        distances[i] = np.linalg.norm(rel_state_t[:3])

    tca_idx = int(np.argmin(distances))
    miss_distance_km = distances[tca_idx]
    tca_s = times[tca_idx]
    threshold_km = hard_body_radius_km + safety_buffer_km
    is_conjunction = miss_distance_km < threshold_km

    return {
        "tca_s": float(tca_s),
        "miss_distance_km": float(miss_distance_km),
        "threshold_km": threshold_km,
        "is_conjunction": bool(is_conjunction),
        "distances_km": distances,
        "times_s": times,
    }
