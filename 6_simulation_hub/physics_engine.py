# 6_simulation_hub/physics_engine.py
"""Continuous (finite-burn) powered-flight propagation - reuses module 1's rk4_step()."""
import sys
import os
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "1_orbital_mechanics"))
from two_body_problem import rk4_step, MU_EARTH  # noqa: E402


def powered_flight_eom(t, state, rocket, mu=MU_EARTH):
    """state = [x, y, z, vx, vy, vz, mass]. Thrust applied prograde."""
    r = state[:3]
    v = state[3:6]
    mass = state[6]

    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    gravity_accel = -mu * r / r_norm ** 3

    if mass > rocket.dry_mass and v_norm > 1e-9:
        thrust_dir = v / v_norm
        thrust_accel = rocket.thrust_accel_kms2(mass) * thrust_dir
        mdot = -rocket.mass_flow_rate_kg_s
    else:
        thrust_accel = np.zeros(3)
        mdot = 0.0

    accel = gravity_accel + thrust_accel
    return np.concatenate([v, accel, [mdot]])


def propagate_powered_burn(r0, v0, m0, rocket, burn_duration_s, dt=1.0, mu=MU_EARTH):
    state = np.concatenate([r0, v0, [m0]])
    t, traj = 0.0, [state.copy()]

    def eom(t, y):
        return powered_flight_eom(t, y, rocket, mu)

    while t < burn_duration_s:
        step_dt = min(dt, burn_duration_s - t)
        state = rk4_step(eom, t, state, step_dt)
        t += step_dt
        traj.append(state.copy())
    return np.array(traj)


def propagate_coast(r0, v0, duration_s, dt=10.0, mu=MU_EARTH):
    sys.path.insert(0, os.path.join(_THIS_DIR, "..", "1_orbital_mechanics"))
    from two_body_problem import propagate_orbit
    return propagate_orbit(r0, v0, duration_s, dt, mu)