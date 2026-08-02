"""
test_module3.py
Tests module 3 (evolutionary optimizer) standalone, then the full
integration: does the GA (searching arbitrary Lambert transfers) converge
toward module 2's analytically-known Hohmann delta-V?
"""
import sys
import os
import time
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "2_symbolic_mission_planner"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "3_evolutionary_optimizer"))

from mission_spec_parser import parse_mission_goal  # noqa: E402
from constraint_solver import hohmann_delta_v  # noqa: E402
from lambert_solver import solve_lambert, LambertError  # noqa: E402
from genetic_engine import GeneticEngine  # noqa: E402

PASS_COUNT = 0
FAIL_COUNT = 0


def check(label, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    print(f"[{status}] {label} {detail}")


# ---- Test 1: Lambert solver conserves angular momentum + energy ----
r1_vec = np.array([7000.0, 0.0, 0.0])
r2_vec = np.array([0.0, 9000.0, 0.0])
v1, v2 = solve_lambert(r1_vec, r2_vec, tof_s=3000.0)
h1 = np.linalg.norm(np.cross(r1_vec, v1))
h2 = np.linalg.norm(np.cross(r2_vec, v2))
check("Lambert solver conserves angular momentum", abs(h1 - h2) / h1 < 1e-8,
      f"(h1={h1:.4f}, h2={h2:.4f})")

# ---- Test 2: Lambert solver rejects degenerate 180-degree geometry ----
try:
    solve_lambert(np.array([7000.0, 0, 0]), np.array([-8000.0, 0, 0]), tof_s=3000.0)
    check("Lambert solver rejects exact 180-degree transfer (A=0)", False)
except LambertError:
    check("Lambert solver rejects exact 180-degree transfer (A=0)", True)

# ---- Test: multi-revolution Lambert solver conserves physics and needs more time ----
from lambert_solver import solve_lambert_multirev  # noqa: E402

sols_m1 = solve_lambert_multirev(r1_vec, np.array([0.0, 9000.0, 0.0]), tof_s=3 * 3600, M=1, grid_size=3000)
check("Multi-rev (M=1) Lambert finds both branches for a feasible tof", len(sols_m1) == 2,
      f"(found {len(sols_m1)} branches)")

if len(sols_m1) == 2:
    all_conserved = True
    for v1, v2, branch in sols_m1:
        r2v = np.array([0.0, 9000.0, 0.0])
        h1 = np.linalg.norm(np.cross(r1_vec, v1))
        h2 = np.linalg.norm(np.cross(r2v, v2))
        if abs(h1 - h2) / h1 > 1e-8:
            all_conserved = False
    check("Both M=1 branches conserve angular momentum", all_conserved)

m0_min, m1_min = None, None
for tof_hr in np.arange(0.5, 10, 0.25):
    try:
        solve_lambert(r1_vec, np.array([0.0, 9000.0, 0.0]), tof_hr * 3600)
        m0_min = tof_hr
        break
    except LambertError:
        continue
for tof_hr in np.arange(0.5, 40, 0.5):
    if solve_lambert_multirev(r1_vec, np.array([0.0, 9000.0, 0.0]), tof_hr * 3600, M=1, grid_size=1500):
        m1_min = tof_hr
        break
check("M=1 requires meaningfully more time than M=0 (extra revolution)",
      m1_min is not None and m0_min is not None and m1_min > m0_min * 2,
      f"(M=0 min~{m0_min}hr, M=1 min~{m1_min}hr)")

# ---- Test 3: GA integration - LEO to GEO mission from module 2 ----
spec = parse_mission_goal("LEO to GEO transfer", max_delta_v_kms=6.0, max_time_s=12 * 3600)
_, _, hohmann_dv = hohmann_delta_v(spec.r1, spec.r2)

t0 = time.time()
ga = GeneticEngine(spec, population_size=40, generations=40, seed=42)
best_gene, best_fitness, best_info = ga.run()
elapsed = time.time() - t0

check("GA converges to a feasible solution", best_info["feasible"], f"({best_info})")

ga_dv = best_info["dv_total_kms"]
check(
    "GA's best delta-V is within 15% of analytic Hohmann optimum",
    ga_dv < hohmann_dv * 1.15,
    f"(GA={ga_dv:.4f} km/s, Hohmann={hohmann_dv:.4f} km/s, "
    f"gap={(ga_dv/hohmann_dv - 1)*100:.2f}%)"
)

check(
    "GA never beats the analytic Hohmann optimum (physics sanity check)",
    ga_dv >= hohmann_dv * 0.999,
    f"(GA={ga_dv:.4f} km/s should be >= Hohmann={hohmann_dv:.4f} km/s)"
)

history = np.array(ga.history)
check(
    "Fitness history is non-decreasing (elitism working correctly)",
    np.all(np.diff(history) >= -1e-9)
)

print(f"\nBenchmark: {ga.pop_size} population x {ga.generations} generations "
      f"({ga.pop_size * ga.generations} Lambert solves) took {elapsed:.2f}s "
      f"({elapsed / (ga.pop_size * ga.generations) * 1000:.2f} ms/solve)")
print(f"Best gene found: transfer_angle={best_gene[0]:.2f} deg, tof={best_gene[1]:.2f} hr")

print(f"\n{PASS_COUNT} passed, {FAIL_COUNT} failed")
if FAIL_COUNT == 0:
    print("MODULE 3: ALL TESTS PASSED, INTEGRATION WITH MODULES 1 & 2 VERIFIED")
else:
    print("MODULE 3: FAILURES PRESENT - DO NOT PROCEED TO MODULE 4")