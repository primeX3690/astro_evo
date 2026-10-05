# 14_collision_avoidance/avoidance_maneuver.py
"""
Minimum-delta-V collision avoidance maneuver: reuses module 4's
`solve_cw_rendezvous` CW-targeting solver (originally built for docking
rendezvous) in reverse - instead of solving for the velocity that
DRIVES miss distance to zero (docking), we solve for the velocity that
PUSHES the relative position at TCA to a safe offset (avoidance).

This is the real technique operational collision-avoidance maneuvers
use for near-circular LEO conjunctions: a single along-track or
cross-track impulse, sized via the same linear relative-motion state
transition matrix used for rendezvous targeting.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "4_neural_controller"))
from docking_env import solve_cw_rendezvous, cw_analytic_propagate  # noqa: E402


def compute_avoidance_maneuver(rel_pos0_km, rel_vel0_kms, n_rad_s, tca_s,
                                safe_miss_distance_km=5.0, burn_lead_time_s=0.0,
                                direction="cross_track"):
    """
    Solves for the single impulsive delta-V (applied at t=burn_lead_time_s,
    i.e. "now" by default) that retargets the relative position AT the
    predicted TCA to a point `safe_miss_distance_km` away along the
    requested axis, instead of the uncorrected close pass.

    direction: "cross_track" (z-axis, out of orbital plane - typically
        the cheapest/most common real avoidance-maneuver direction since
        it doesn't perturb the orbit's period/phasing) or "radial" (x-axis).
    Returns delta_v_kms (3-vector) and the resulting (verified) new miss
    distance after applying it.
    """
    axis_map = {"cross_track": 2, "radial": 0, "along_track": 1}
    if direction not in axis_map:
        raise ValueError(f"direction must be one of {list(axis_map.keys())}")
    axis = axis_map[direction]

    # Propagate to the burn time first (usually t=0, "burn now")
    rel_state0 = np.concatenate([rel_pos0_km, rel_vel0_kms])
    rel_state_burn = cw_analytic_propagate(rel_state0, burn_lead_time_s, n_rad_s)
    pos_at_burn = rel_state_burn[:3]
    vel_at_burn_before = rel_state_burn[3:]

    tof_remaining_s = tca_s - burn_lead_time_s
    if tof_remaining_s <= 0:
        raise ValueError("burn_lead_time_s must be before tca_s - cannot maneuver after closest approach")

    # Where would we end up at TCA with NO maneuver (for comparison)?
    rel_state_tca_unmaneuvered = cw_analytic_propagate(rel_state_burn, tof_remaining_s, n_rad_s)
    target_pos_at_tca = rel_state_tca_unmaneuvered[:3].copy()
    target_pos_at_tca[axis] += safe_miss_distance_km  # push off-axis by the safety margin

    v0_required, v_arrival = solve_cw_rendezvous(pos_at_burn, target_pos_at_tca, tof_remaining_s, n_rad_s)
    delta_v_kms = v0_required - vel_at_burn_before

    # Verify: propagate WITH the maneuver applied and confirm the new miss distance
    rel_state_after_burn = np.concatenate([pos_at_burn, v0_required])
    rel_state_tca_maneuvered = cw_analytic_propagate(rel_state_after_burn, tof_remaining_s, n_rad_s)
    achieved_miss_distance_km = np.linalg.norm(rel_state_tca_maneuvered[:3])

    unmaneuvered_miss_distance_km = float(np.linalg.norm(rel_state_tca_unmaneuvered[:3]))

    return {
        "delta_v_kms": delta_v_kms,
        "delta_v_magnitude_kms": float(np.linalg.norm(delta_v_kms)),
        "direction": direction,
        "achieved_miss_distance_km": float(achieved_miss_distance_km),
        "unmaneuvered_miss_distance_km": unmaneuvered_miss_distance_km,
    }
