# 4_neural_controller/docking_env.py
"""
Clohessy-Wiltshire (CW/Hill's) equations: linearized relative motion of a
chaser near a target in circular orbit - real 3D translational
relative-motion docking model (not full 6-DOF attitude+translation).

Frame (Hill/RSW): x=radial, y=along-track, z=cross-track.
"""
import sys
import os
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "1_orbital_mechanics"))
from two_body_problem import rk4_step  # noqa: E402


def cw_dynamics(t, state, n, accel=np.zeros(3)):
    x, y, z, vx, vy, vz = state
    ax = 3 * n ** 2 * x + 2 * n * vy + accel[0]
    ay = -2 * n * vx + accel[1]
    az = -n ** 2 * z + accel[2]
    return np.array([vx, vy, vz, ax, ay, az])


def propagate_cw(rel_state0, duration_s, n, dt=1.0, accel=np.zeros(3)):
    state = np.array(rel_state0, dtype=float)
    t, traj = 0.0, [state.copy()]

    def eom(t, y):
        return cw_dynamics(t, y, n, accel)

    while t < duration_s:
        step_dt = min(dt, duration_s - t)
        state = rk4_step(eom, t, state, step_dt)
        t += step_dt
        traj.append(state.copy())
    return np.array(traj)


def cw_state_transition_matrix(n, t):
    """Closed-form 6x6 CW STM (Clohessy-Wiltshire). Used to verify RK4 and to solve rendezvous."""
    nt = n * t
    s, c = np.sin(nt), np.cos(nt)

    Phi_rr = np.array([[4 - 3 * c, 0, 0], [6 * (s - nt), 1, 0], [0, 0, c]])
    Phi_rv = np.array([[s / n, 2 * (1 - c) / n, 0],
                        [2 * (c - 1) / n, (4 * s - 3 * nt) / n, 0], [0, 0, s / n]])
    Phi_vr = np.array([[3 * n * s, 0, 0], [6 * n * (c - 1), 0, 0], [0, 0, -n * s]])
    Phi_vv = np.array([[c, 2 * s, 0], [-2 * s, 4 * c - 3, 0], [0, 0, c]])

    Phi = np.zeros((6, 6))
    Phi[:3, :3] = Phi_rr
    Phi[:3, 3:] = Phi_rv
    Phi[3:, :3] = Phi_vr
    Phi[3:, 3:] = Phi_vv
    return Phi


def cw_analytic_propagate(rel_state0, t, n):
    Phi = cw_state_transition_matrix(n, t)
    return Phi @ np.array(rel_state0, dtype=float)


def solve_cw_rendezvous(rel_pos0, rel_pos_target, tof_s, n):
    """Two-impulse rendezvous targeting (CW equivalent of Lambert's problem)."""
    Phi = cw_state_transition_matrix(n, tof_s)
    Phi_rr, Phi_rv = Phi[:3, :3], Phi[:3, 3:]
    Phi_vr, Phi_vv = Phi[3:, :3], Phi[3:, 3:]

    rel_pos0 = np.array(rel_pos0, dtype=float)
    rel_pos_target = np.array(rel_pos_target, dtype=float)

    v0 = np.linalg.solve(Phi_rv, rel_pos_target - Phi_rr @ rel_pos0)
    v_arrival = Phi_vr @ rel_pos0 + Phi_vv @ v0
    return v0, v_arrival