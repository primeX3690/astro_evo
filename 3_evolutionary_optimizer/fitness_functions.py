# 3_evolutionary_optimizer/fitness_functions.py
"""Computes total delta-V for a candidate trajectory gene via lambert_solver, penalizing budget violations."""
import numpy as np
from lambert_solver import solve_lambert, LambertError, MU_EARTH
from trajectory_encoding import decode_gene


def circular_velocity(r_vec, mu=MU_EARTH):
    r = np.linalg.norm(r_vec)
    v_mag = np.sqrt(mu / r)
    direction = np.array([-r_vec[1], r_vec[0], 0.0]) / r
    return v_mag * direction


def evaluate_trajectory(gene, mission_spec, penalty_weight=50.0):
    r1_vec, r2_vec, tof_s = decode_gene(gene, mission_spec.r1, mission_spec.r2)
    try:
        v1_transfer, v2_transfer = solve_lambert(r1_vec, r2_vec, tof_s)
    except LambertError:
        return -1e6, {"feasible": False, "reason": "lambert_failed"}

    v1_circ = circular_velocity(r1_vec)
    v2_circ = circular_velocity(r2_vec)
    dv1 = np.linalg.norm(v1_transfer - v1_circ)
    dv2 = np.linalg.norm(v2_circ - v2_transfer)
    dv_total = dv1 + dv2

    cost = dv_total
    violations = []
    if mission_spec.max_delta_v is not None and dv_total > mission_spec.max_delta_v:
        cost += penalty_weight * (dv_total - mission_spec.max_delta_v)
        violations.append("delta_v_budget")
    if mission_spec.max_time is not None and tof_s > mission_spec.max_time:
        cost += penalty_weight * (tof_s - mission_spec.max_time) / 3600
        violations.append("time_budget")

    return -cost, {
        "feasible": len(violations) == 0, "violations": violations,
        "dv1_kms": dv1, "dv2_kms": dv2, "dv_total_kms": dv_total,
        "tof_hr": tof_s / 3600, "transfer_angle_deg": gene[0],
    }