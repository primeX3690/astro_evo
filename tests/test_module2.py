"""
test_module2.py
Correctness test + integration test (calls into module 1's RK4 propagator)
+ benchmark for module 2.
"""
import sys
import os
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "2_symbolic_mission_planner"))

from mission_spec_parser import parse_mission_goal  # noqa: E402
from constraint_solver import hohmann_delta_v, hohmann_transfer_time, solve_constraints  # noqa: E402
from temporal_planner import simulate_transfer  # noqa: E402

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


# ---- Test 1: mission spec parsing ----
spec = parse_mission_goal("LEO to GEO transfer", max_delta_v_kms=5.0, max_time_s=6 * 3600)
check("Parse 'LEO to GEO transfer'", spec.r1 < spec.r2, f"(r1={spec.r1}, r2={spec.r2})")

# ---- Test 2: delta-V against known real-world value ----
dv1, dv2, dv_total = hohmann_delta_v(spec.r1, spec.r2)
check(
    "Hohmann delta-V in expected real-world range (3.5-4.3 km/s)",
    3.5 <= dv_total <= 4.3,
    f"(dv_total={dv_total:.4f} km/s)"
)

# ---- Test 3: transfer time against known value (~5.25 hr for LEO-GEO) ----
t_hr = hohmann_transfer_time(spec.r1, spec.r2) / 3600
check("Transfer time near known ~5.25 hr", 5.0 <= t_hr <= 5.5, f"(t={t_hr:.2f} hr)")

# ---- Test 4: constraint solver flags infeasible mission ----
tight_spec = parse_mission_goal("LEO to GEO transfer", max_delta_v_kms=2.0)  # too tight
report = solve_constraints(tight_spec)
check("Constraint solver flags infeasible (dv budget too tight)", not report["feasible"])

# ---- Test 5: constraint solver passes feasible mission ----
loose_spec = parse_mission_goal("LEO to GEO transfer", max_delta_v_kms=5.0, max_time_s=8 * 3600)
report2 = solve_constraints(loose_spec)
check("Constraint solver passes feasible mission", report2["feasible"])

# ---- Test 6: INTEGRATION - actually propagate transfer through module 1's RK4 ----
t0 = time.time()
traj, verification = simulate_transfer(spec, dt=20.0)
elapsed = time.time() - t0
check(
    "Integration: simulated transfer reaches target radius (<1% error)",
    verification["verified"],
    f"(target={verification['planned_r2_km']:.1f}km, "
    f"achieved={verification['achieved_r_km']:.1f}km, "
    f"error={verification['radius_error_pct']:.4f}%)"
)

# ---- Benchmark ----
print(f"\nBenchmark: full transfer simulation ({len(traj)} propagation steps) "
      f"took {elapsed*1000:.1f} ms on this machine")

print(f"\n{PASS_COUNT} passed, {FAIL_COUNT} failed")
if FAIL_COUNT == 0:
    print("MODULE 2: ALL TESTS PASSED, INTEGRATION WITH MODULE 1 VERIFIED")
else:
    print("MODULE 2: FAILURES PRESENT - DO NOT PROCEED TO MODULE 3")