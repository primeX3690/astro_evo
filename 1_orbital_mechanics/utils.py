# 1_orbital_mechanics/utils.py
import numpy as np


def kepler_solve(M, e, tol=1e-10, max_iter=100):
    """Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E (radians)."""
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return E


def true_from_eccentric(E, e):
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))


def period(a, mu=398600.4418):
    return 2 * np.pi * np.sqrt(a ** 3 / mu)