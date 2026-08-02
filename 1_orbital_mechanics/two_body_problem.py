# 1_orbital_mechanics/two_body_problem.py
"""
two_body_problem.py
Numerical integrator (RK4) for the two-body equations of motion.
"""
import numpy as np

MU_EARTH = 398600.4418  # km^3/s^2


def two_body_eom(t, state, mu=MU_EARTH):
    """State = [x, y, z, vx, vy, vz]. Returns derivative [vx, vy, vz, ax, ay, az]."""
    r, v = state[:3], state[3:]
    a = -mu * r / np.linalg.norm(r) ** 3
    return np.concatenate([v, a])


def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt / 2 * k1)
    k3 = f(t + dt / 2, y + dt / 2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def propagate_orbit(r0, v0, duration_s, dt=10.0, mu=MU_EARTH):
    """Propagate from initial state (r0, v0) for duration_s seconds. Returns Nx6 trajectory array."""
    state = np.concatenate([r0, v0])
    t, traj = 0.0, [state.copy()]
    while t < duration_s:
        state = rk4_step(two_body_eom, t, state, dt)
        t += dt
        traj.append(state.copy())
    return np.array(traj)