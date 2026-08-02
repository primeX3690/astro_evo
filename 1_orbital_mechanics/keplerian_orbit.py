# 1_orbital_mechanics/keplerian_orbit.py
import numpy as np

MU_EARTH = 398600.4418  # km^3/s^2


def keplerian_to_state_vector(a, e, i, raan, argp, nu, mu=MU_EARTH):
    """Classical orbital elements -> position/velocity (km, km/s) in ECI frame."""
    i, raan, argp, nu = map(np.radians, (i, raan, argp, nu))
    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(nu))

    r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0])

    R3_W = np.array([[np.cos(raan), -np.sin(raan), 0],
                      [np.sin(raan),  np.cos(raan), 0],
                      [0, 0, 1]])
    R1_i = np.array([[1, 0, 0],
                      [0, np.cos(i), -np.sin(i)],
                      [0, np.sin(i),  np.cos(i)]])
    R3_w = np.array([[np.cos(argp), -np.sin(argp), 0],
                      [np.sin(argp),  np.cos(argp), 0],
                      [0, 0, 1]])
    Q = R3_W @ R1_i @ R3_w

    return Q @ r_pf, Q @ v_pf