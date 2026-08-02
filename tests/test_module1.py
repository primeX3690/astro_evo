import sys
import os
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "1_orbital_mechanics"))

from keplerian_orbit import keplerian_to_state_vector  # noqa: E402
from two_body_problem import propagate_orbit  # noqa: E402
from utils import period  # noqa: E402

MU = 398600.4418

# ISS-jaisi LEO orbit
a, e, i, raan, argp, nu = 6798, 0.0007, 51.6, 0, 0, 0
r0, v0 = keplerian_to_state_vector(a, e, i, raan, argp, nu)
T = period(a)

traj = propagate_orbit(r0, v0, duration_s=T, dt=5.0)


def specific_energy(state):
    r = np.linalg.norm(state[:3])
    v = np.linalg.norm(state[3:])
    return v ** 2 / 2 - MU / r


energies = [specific_energy(s) for s in traj]
drift_pct = (max(energies) - min(energies)) / abs(energies[0]) * 100
pos_error_km = np.linalg.norm(traj[-1][:3] - traj[0][:3])

print(f"Orbital period: {T/60:.2f} min")
print(f"Energy drift over 1 orbit: {drift_pct:.8f}%")
print(f"Start pos (km): {traj[0][:3]}")
print(f"End pos after 1 period (km): {traj[-1][:3]}")
print(f"Position error after 1 full orbit: {pos_error_km:.4f} km")
print(f"PASS" if drift_pct < 0.01 and pos_error_km < 50 else "FAIL")

# ---- J2 perturbation test: numeric RAAN drift vs. known analytic formula ----
from two_body_problem import rk4_step  # noqa: E402
from perturbation_models import two_body_j2_eom, analytic_nodal_regression_rate  # noqa: E402

state_j2 = np.concatenate([r0, v0])
dt_j2 = 20.0
n_orbits = 15
duration_j2 = n_orbits * T
t_j2 = 0.0
raans, times_j2 = [], []


def j2_eom(t, y):
    return two_body_j2_eom(t, y, MU)


while t_j2 < duration_j2:
    rr, vv = state_j2[:3], state_j2[3:]
    h = np.cross(rr, vv)
    node = np.cross([0, 0, 1], h)
    raans.append(np.arctan2(node[1], node[0]))
    times_j2.append(t_j2)
    state_j2 = rk4_step(j2_eom, t_j2, state_j2, dt_j2)
    t_j2 += dt_j2

raans_unwrapped = np.unwrap(raans)
slope, _ = np.polyfit(np.array(times_j2), raans_unwrapped, 1)
numeric_rate_deg_day = np.degrees(slope) * 86400
analytic_rate_deg_day = np.degrees(analytic_nodal_regression_rate(a, e, i, MU)) * 86400
j2_error_pct = abs(numeric_rate_deg_day / analytic_rate_deg_day - 1) * 100

print(f"\nJ2 numeric RAAN drift: {numeric_rate_deg_day:.4f} deg/day")
print(f"J2 analytic RAAN drift: {analytic_rate_deg_day:.4f} deg/day")
print(f"J2 match: {j2_error_pct:.3f}% difference")
print("PASS" if j2_error_pct < 2.0 else "FAIL")

# ---- SRP + drag test: order-of-magnitude sanity + qualitative decay behavior ----
from perturbation_models import srp_acceleration, drag_acceleration, full_perturbed_eom  # noqa: E402

a_srp = srp_acceleration(np.array([7000.0, 0, 0]), np.array([1.0, 0, 0]), cr=1.3, area_to_mass_m2_kg=0.02)
srp_ms2 = np.linalg.norm(a_srp) * 1000
print(f"\nSRP magnitude: {srp_ms2:.3e} m/s^2 (expect ~1e-7)")
print("PASS" if 1e-8 < srp_ms2 < 1e-6 else "FAIL")

a_srp_eclipse = srp_acceleration(np.array([-7000.0, 0, 0]), np.array([1.0, 0, 0]), cr=1.3, area_to_mass_m2_kg=0.02)
print(f"SRP in eclipse: {np.linalg.norm(a_srp_eclipse)} (expect 0.0)")
print("PASS" if np.allclose(a_srp_eclipse, 0) else "FAIL")

a_drag = drag_acceleration(np.array([6771.0, 0, 0]), np.array([0, 7.66, 0]), cd=2.2, area_to_mass_m2_kg=0.01)
drag_ms2 = np.linalg.norm(a_drag) * 1000
print(f"Drag magnitude at 400km: {drag_ms2:.3e} m/s^2 (expect 1e-6 to 1e-5)")
print("PASS" if 1e-7 < drag_ms2 < 1e-4 else "FAIL")

# qualitative decay: low orbit + drag should lose altitude monotonically over days
r0_d, v0_d = keplerian_to_state_vector(6678.0, 0.001, 0.0, 0, 0, 0)
state_d = np.concatenate([r0_d, v0_d])


def decay_eom(t, y):
    return full_perturbed_eom(t, y, MU, cd=2.2, area_to_mass_drag=0.05,
                               cr=1.3, area_to_mass_srp=0.02, include_srp=False)


t_d, alt_start = 0.0, np.linalg.norm(state_d[:3]) - 6378.137
for _ in range(int(2 * 86400 / 30.0)):
    state_d = rk4_step(decay_eom, t_d, state_d, 30.0)
    t_d += 30.0
alt_end = np.linalg.norm(state_d[:3]) - 6378.137
print(f"Altitude after 2 days of drag: {alt_start:.1f} -> {alt_end:.1f} km")
print("PASS" if alt_end < alt_start else "FAIL")

# ---- Numba-accelerated propagator: correctness + speed check ----
import time  # noqa: E402
from fast_propagator import propagate_orbit_fast  # noqa: E402

traj_py = propagate_orbit(r0, v0, duration_s=3000, dt=10.0)
traj_nb = propagate_orbit_fast(r0, v0, duration_s=3000, dt=10.0)
max_diff = np.max(np.abs(traj_py - traj_nb[:len(traj_py)]))
print(f"\nNumba vs pure-Python propagator max diff: {max_diff:.2e} (expect ~1e-10 or better)")
print("PASS" if max_diff < 1e-6 else "FAIL")

_ = propagate_orbit_fast(r0, v0, duration_s=100, dt=10.0)  # JIT warmup, excluded from timing
N = 200
t0 = time.time()
for _ in range(N):
    propagate_orbit(r0, v0, duration_s=3000, dt=10.0)
py_time = time.time() - t0
t0 = time.time()
for _ in range(N):
    propagate_orbit_fast(r0, v0, duration_s=3000, dt=10.0)
nb_time = time.time() - t0
speedup = py_time / nb_time
print(f"Speedup: {speedup:.1f}x ({py_time/N*1000:.3f} ms/call -> {nb_time/N*1000:.3f} ms/call)")
print("PASS" if speedup > 5 else "FAIL")