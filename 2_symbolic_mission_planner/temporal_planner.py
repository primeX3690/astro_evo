# 2_symbolic_mission_planner/temporal_planner.py
"""
temporal_planner.py
Builds a burn sequence for a Hohmann transfer and actually propagates it
through module 1's RK4 integrator.
"""
import sys
import os
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE1_DIR = os.path.join(_THIS_DIR, "..", "1_orbital_mechanics")
sys.path.insert(0, _MODULE1_DIR)

from two_body_problem import propagate_orbit  # noqa: E402
from constraint_solver import hohmann_delta_v, hohmann_transfer_time

MU_EARTH = 398600.4418


def build_burn_sequence(mission_spec):
    dv1, dv2, _ = hohmann_delta_v(mission_spec.r1, mission_spec.r2)
    t_transfer = hohmann_transfer_time(mission_spec.r1, mission_spec.r2)
    return [
        {"t": 0.0, "dv": dv1, "description": f"Departure burn at r1={mission_spec.r1:.0f}km"},
        {"t": t_transfer, "dv": -dv2 if mission_spec.r2 < mission_spec.r1 else dv2,
         "description": f"Arrival burn at r2={mission_spec.r2:.0f}km"},
    ]


def simulate_transfer(mission_spec, dt=20.0):
    burns = build_burn_sequence(mission_spec)
    t_transfer = burns[1]["t"]

    r0 = np.array([mission_spec.r1, 0.0, 0.0])
    v_circ = np.sqrt(MU_EARTH / mission_spec.r1)
    v0 = np.array([0.0, v_circ, 0.0])
    v0_boosted = v0 * (1 + burns[0]["dv"] / v_circ)

    leg1 = propagate_orbit(r0, v0_boosted, duration_s=t_transfer, dt=dt)

    r_arrival = leg1[-1][:3]
    v_arrival = leg1[-1][3:]
    v_arrival_mag = np.linalg.norm(v_arrival)
    v_target = np.sqrt(MU_EARTH / mission_spec.r2)
    v_arrival_circularized = v_arrival * (v_target / v_arrival_mag)

    leg2 = propagate_orbit(r_arrival, v_arrival_circularized, duration_s=3600, dt=dt)
    full_traj = np.vstack([leg1, leg2])

    final_r = np.linalg.norm(leg2[-1][:3])
    r_error_pct = abs(final_r - mission_spec.r2) / mission_spec.r2 * 100

    verification = {
        "planned_r2_km": mission_spec.r2, "achieved_r_km": final_r,
        "radius_error_pct": r_error_pct, "burns": burns,
        "verified": r_error_pct < 1.0,
    }
    return full_traj, verification