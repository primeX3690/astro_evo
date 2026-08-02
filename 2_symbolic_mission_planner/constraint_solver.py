# 2_symbolic_mission_planner/constraint_solver.py
"""
constraint_solver.py
Computes Hohmann transfer delta-V and checks it against mission constraints
(fuel budget via Tsiolkovsky equation, time budget via module 1 orbital periods).
"""
import numpy as np
from mission_spec_parser import MU_EARTH


def hohmann_delta_v(r1, r2, mu=MU_EARTH):
    a_transfer = (r1 + r2) / 2
    v1_circ = np.sqrt(mu / r1)
    v1_transfer = np.sqrt(mu * (2 / r1 - 1 / a_transfer))
    dv1 = abs(v1_transfer - v1_circ)
    v2_circ = np.sqrt(mu / r2)
    v2_transfer = np.sqrt(mu * (2 / r2 - 1 / a_transfer))
    dv2 = abs(v2_circ - v2_transfer)
    return dv1, dv2, dv1 + dv2


def hohmann_transfer_time(r1, r2, mu=MU_EARTH):
    a_transfer = (r1 + r2) / 2
    T_transfer = 2 * np.pi * np.sqrt(a_transfer ** 3 / mu)
    return T_transfer / 2


def propellant_fraction(delta_v_kms, isp_s, g0=9.80665e-3):
    v_e = isp_s * g0
    mass_ratio = np.exp(delta_v_kms / v_e)
    return 1 - (1 / mass_ratio)


def solve_constraints(mission_spec, isp_s=300):
    dv1, dv2, dv_total = hohmann_delta_v(mission_spec.r1, mission_spec.r2)
    transfer_time_s = hohmann_transfer_time(mission_spec.r1, mission_spec.r2)
    prop_frac = propellant_fraction(dv_total, isp_s)

    report = {
        "mission": mission_spec.name,
        "dv1_kms": dv1, "dv2_kms": dv2, "dv_total_kms": dv_total,
        "transfer_time_hr": transfer_time_s / 3600,
        "propellant_mass_fraction": prop_frac,
        "feasible": True,
        "violations": [],
    }

    if mission_spec.max_delta_v is not None and dv_total > mission_spec.max_delta_v:
        report["feasible"] = False
        report["violations"].append(
            f"delta-V budget exceeded: needs {dv_total:.4f} km/s, budget {mission_spec.max_delta_v} km/s"
        )

    if mission_spec.max_time is not None and transfer_time_s > mission_spec.max_time:
        report["feasible"] = False
        report["violations"].append(
            f"time budget exceeded: needs {transfer_time_s/3600:.2f} hr, "
            f"budget {mission_spec.max_time/3600:.2f} hr"
        )

    return report