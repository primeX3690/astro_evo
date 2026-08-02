"""
test_module5.py
Tests module 5's four pieces, integrating with earlier modules.
"""
import sys
import os
import time
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "5_self_improve_loop"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "2_symbolic_mission_planner"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "3_evolutionary_optimizer"))

from experiment_logger import ExperimentLogger  # noqa: E402
from mission_generator import generate_mission_batch, generate_lander_scenario  # noqa: E402
from meta_optimizer import optimize_ga_transfer, optimize_ppo_lander  # noqa: E402
from curriculum_trainer import train_lander_curriculum  # noqa: E402
from constraint_solver import solve_constraints  # noqa: E402

PASS_COUNT = 0
FAIL_COUNT = 0

# clean up log files from any previous run - ExperimentLogger APPENDS to
# existing files, so repeated test runs accumulate stale trials otherwise
_MODULE5_DIR = os.path.join(_THIS_DIR, "..", "5_self_improve_loop")
for _stale_log in ["logs_ga_search.json", "logs_ppo_search.json", "logs_curriculum.json"]:
    _stale_path = os.path.join(_MODULE5_DIR, _stale_log)
    if os.path.exists(_stale_path):
        os.remove(_stale_path)


def check(label, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    print(f"[{status}] {label} {detail}")


t_start = time.time()

# ---- Test 1: experiment_logger read/write correctness ----
test_log_path = os.path.join(_THIS_DIR, "_test_logger.json")
if os.path.exists(test_log_path):
    os.remove(test_log_path)
logger = ExperimentLogger(test_log_path)
logger.log_trial({"x": 1}, {"objective": 5.0})
logger.log_trial({"x": 2}, {"objective": 9.0})
logger.log_trial({"x": 3}, {"objective": 2.0})
best = logger.get_best("objective", maximize=True)
check("Logger retrieves the correct best trial", best["config"]["x"] == 2, f"(best={best['config']})")
check("Logger persists trials to disk", len(logger.all_trials()) == 3)
os.remove(test_log_path)

# ---- Test 2: mission_generator produces missions module 2/3 can solve ----
missions = generate_mission_batch(n=5, seed=1)
solved_ok = 0
for m in missions:
    report = solve_constraints(m)
    if "dv_total_kms" in report and report["dv_total_kms"] > 0:
        solved_ok += 1
check(
    "mission_generator produces missions module 2's constraint_solver can process",
    solved_ok == 5,
    f"({solved_ok}/5 solved without error)"
)

# ---- Test 3: mission_generator's lander difficulty ramp is monotonic ----
scenarios = [generate_lander_scenario(np.random.default_rng(1), d) for d in [0.0, 0.5, 1.0]]
ys = [s["start_y"] for s in scenarios]
check("Lander difficulty ramp increases start_y monotonically", ys[0] < ys[1] < ys[2], f"({ys})")

# ---- Test 4: meta_optimizer GA search (fast) ----
print("\nRunning GA hyperparameter search (module 3 integration)...")
ga_config, ga_trial, ga_logger = optimize_ga_transfer(n_trials=5, seed=1)
check(
    "GA meta-optimizer finds a config with small gap to Hohmann optimum",
    ga_trial["metrics"]["gap_to_hohmann_pct"] < 5.0,
    f"(best config={ga_config}, gap={ga_trial['metrics']['gap_to_hohmann_pct']:.2f}%)"
)

# ---- Test 5: meta_optimizer PPO search (module 4 integration, kept SHORT for test speed) ----
print("\nRunning PPO hyperparameter search (module 4 integration, short trials for test speed)...")
ppo_config, ppo_trial, ppo_logger = optimize_ppo_lander(n_trials=3, episodes_per_trial=600, seed=1)
check(
    "PPO meta-optimizer completes trials and logs a valid landing rate",
    0 <= ppo_trial["metrics"]["landing_rate_pct"] <= 100,
    f"(best config={ppo_config}, landing_rate={ppo_trial['metrics']['landing_rate_pct']:.1f}%)"
)

# ---- Test 6: curriculum_trainer runs across stages without crashing (short run) ----
print("\nRunning short curriculum training (module 4 integration)...")
agent, stage_results, curr_logger = train_lander_curriculum(n_stages=3, episodes_per_stage=300, seed=1)
check(
    "Curriculum trainer completes all stages and logs results",
    len(stage_results) == 3 and len(curr_logger.all_trials()) == 3,
    f"(stage results={[(round(d,2), round(r,1)) for d,r in stage_results]})"
)

elapsed = time.time() - t_start
print(f"\nBenchmark: full module 5 test suite took {elapsed:.1f}s")

print(f"\n{PASS_COUNT} passed, {FAIL_COUNT} failed")
if FAIL_COUNT == 0:
    print("MODULE 5: ALL TESTS PASSED, INTEGRATION WITH MODULES 2, 3 & 4 VERIFIED")
else:
    print("MODULE 5: FAILURES PRESENT")