# 3_evolutionary_optimizer/lambert_solver.py
"""
lambert_solver.py
Solves Lambert's problem via universal variables (Stumpff C(z), S(z)).
Includes both single-arc (M=0) and multi-revolution (M>=1) solvers.
"""
import sys
import os
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "1_orbital_mechanics"))
from two_body_problem import MU_EARTH  # noqa: E402


def _stumpff_C(z):
    z = np.asarray(z, dtype=float)
    C = np.zeros_like(z)
    pos = z > 1e-6
    neg = z < -1e-6
    zero = ~pos & ~neg
    C[pos] = (1 - np.cos(np.sqrt(z[pos]))) / z[pos]
    C[neg] = (np.cosh(np.sqrt(-z[neg])) - 1) / (-z[neg])
    C[zero] = 0.5 - z[zero] / 24 + z[zero] ** 2 / 720
    return C


def _stumpff_S(z):
    z = np.asarray(z, dtype=float)
    S = np.zeros_like(z)
    pos = z > 1e-6
    neg = z < -1e-6
    zero = ~pos & ~neg
    sz = np.sqrt(z[pos])
    S[pos] = (sz - np.sin(sz)) / sz ** 3
    sz2 = np.sqrt(-z[neg])
    S[neg] = (np.sinh(sz2) - sz2) / sz2 ** 3
    S[zero] = 1 / 6 - z[zero] / 120 + z[zero] ** 2 / 5040
    return S


class LambertError(Exception):
    pass


def solve_lambert(r1_vec, r2_vec, tof_s, mu=MU_EARTH, prograde=True, grid_size=3000, bisect_iter=60):
    r1_vec = np.asarray(r1_vec, dtype=float)
    r2_vec = np.asarray(r2_vec, dtype=float)
    r1, r2 = np.linalg.norm(r1_vec), np.linalg.norm(r2_vec)

    cross_z = r1_vec[0] * r2_vec[1] - r1_vec[1] * r2_vec[0]
    cos_dnu = np.dot(r1_vec, r2_vec) / (r1 * r2)
    dnu = np.arccos(np.clip(cos_dnu, -1.0, 1.0))
    if prograde:
        if cross_z < 0:
            dnu = 2 * np.pi - dnu
    else:
        if cross_z >= 0:
            dnu = 2 * np.pi - dnu

    A = np.sin(dnu) * np.sqrt(r1 * r2 / (1 - np.cos(dnu)))
    if abs(A) < 1e-9:
        raise LambertError(f"Degenerate geometry (transfer angle={np.degrees(dnu):.2f} deg, A~=0)")

    z_grid = np.linspace(-4 * np.pi ** 2 + 0.05, 4 * np.pi ** 2 - 0.05, grid_size)
    C = _stumpff_C(z_grid)
    S = _stumpff_S(z_grid)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = r1 + r2 + A * (z_grid * S - 1) / np.sqrt(C)
    valid = (C > 1e-9) & (y >= 0)
    if valid.sum() < 2:
        raise LambertError("No valid universal-variable domain found for this geometry")

    z_v, C_v, S_v, y_v = z_grid[valid], C[valid], S[valid], y[valid]
    chi = np.sqrt(y_v / C_v)
    t = (chi ** 3 * S_v + A * np.sqrt(y_v)) / np.sqrt(mu)

    if tof_s < t.min() or tof_s > t.max():
        raise LambertError(
            f"tof_s={tof_s:.1f}s outside solvable range [{t.min():.1f}, {t.max():.1f}]s for this geometry"
        )

    idx = np.searchsorted(t, tof_s)
    z_lo, z_hi = z_v[idx - 1], z_v[idx]

    def eval_t(z):
        Cz = _stumpff_C(np.array([z]))[0]
        Sz = _stumpff_S(np.array([z]))[0]
        yz = r1 + r2 + A * (z * Sz - 1) / np.sqrt(Cz)
        if yz < 0:
            return None
        chiz = np.sqrt(yz / Cz)
        tz = (chiz ** 3 * Sz + A * np.sqrt(yz)) / np.sqrt(mu)
        return tz, yz

    y_mid = None
    for _ in range(bisect_iter):
        z_mid = (z_lo + z_hi) / 2
        result = eval_t(z_mid)
        if result is None:
            z_lo = z_mid
            continue
        t_mid, y_mid = result
        if abs(t_mid - tof_s) < 1e-6 * max(1.0, tof_s):
            break
        if t_mid < tof_s:
            z_lo = z_mid
        else:
            z_hi = z_mid

    f = 1 - y_mid / r1
    g = A * np.sqrt(y_mid / mu)
    gdot = 1 - y_mid / r2
    v1 = (r2_vec - f * r1_vec) / g
    v2 = (gdot * r2_vec - r1_vec) / g
    return v1, v2


def solve_lambert_multirev(r1_vec, r2_vec, tof_s, mu=MU_EARTH, prograde=True, M=0, grid_size=4000):
    """M=0 -> same as solve_lambert(). M>=1 -> finds both branches (low-path/high-path)
    within the z-band (2*pi*M)^2 < z < (2*pi*(M+1))^2."""
    if M == 0:
        v1, v2 = solve_lambert(r1_vec, r2_vec, tof_s, mu, prograde)
        return [(v1, v2, "direct")]

    r1_vec = np.asarray(r1_vec, dtype=float)
    r2_vec = np.asarray(r2_vec, dtype=float)
    r1, r2 = np.linalg.norm(r1_vec), np.linalg.norm(r2_vec)

    cross_z = r1_vec[0] * r2_vec[1] - r1_vec[1] * r2_vec[0]
    cos_dnu = np.dot(r1_vec, r2_vec) / (r1 * r2)
    dnu = np.arccos(np.clip(cos_dnu, -1.0, 1.0))
    if prograde:
        if cross_z < 0:
            dnu = 2 * np.pi - dnu
    else:
        if cross_z >= 0:
            dnu = 2 * np.pi - dnu

    A = np.sin(dnu) * np.sqrt(r1 * r2 / (1 - np.cos(dnu)))
    if abs(A) < 1e-9:
        raise LambertError(f"Degenerate geometry (transfer angle={np.degrees(dnu):.2f} deg, A~=0)")

    z_lo_band = (2 * np.pi * M) ** 2 + 0.05
    z_hi_band = (2 * np.pi * (M + 1)) ** 2 - 0.05
    z_grid = np.linspace(z_lo_band, z_hi_band, grid_size)
    C = _stumpff_C(z_grid)
    S = _stumpff_S(z_grid)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = r1 + r2 + A * (z_grid * S - 1) / np.sqrt(C)
    valid = (C > 1e-9) & (y >= 0)
    if valid.sum() < 2:
        return []

    z_v, y_v = z_grid[valid], y[valid]
    chi = np.sqrt(y_v / np.maximum(_stumpff_C(z_v), 1e-12))
    t_v = (chi ** 3 * _stumpff_S(z_v) + A * np.sqrt(y_v)) / np.sqrt(mu)

    diff = t_v - tof_s
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]

    def eval_t(z):
        Cz = _stumpff_C(np.array([z]))[0]
        Sz = _stumpff_S(np.array([z]))[0]
        if Cz <= 1e-12:
            return None
        yz = r1 + r2 + A * (z * Sz - 1) / np.sqrt(Cz)
        if yz < 0:
            return None
        chiz = np.sqrt(yz / Cz)
        tz = (chiz ** 3 * Sz + A * np.sqrt(yz)) / np.sqrt(mu)
        return tz, yz

    solutions = []
    for idx, sc in enumerate(sign_changes):
        z_lo, z_hi = z_v[sc], z_v[sc + 1]
        y_mid = None
        for _ in range(60):
            z_mid = (z_lo + z_hi) / 2
            result = eval_t(z_mid)
            if result is None:
                z_lo = z_mid
                continue
            t_mid, y_mid = result
            if abs(t_mid - tof_s) < 1e-6 * max(1.0, tof_s):
                break
            if t_mid < tof_s:
                z_lo = z_mid
            else:
                z_hi = z_mid

        f = 1 - y_mid / r1
        g = A * np.sqrt(y_mid / mu)
        gdot = 1 - y_mid / r2
        v1 = (r2_vec - f * r1_vec) / g
        v2 = (gdot * r2_vec - r1_vec) / g
        branch = "low-path" if idx == 0 else "high-path"
        solutions.append((v1, v2, branch))

    return solutions