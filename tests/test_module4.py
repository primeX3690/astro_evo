"""
test_module4.py
Trains the real clipped-PPO agent directly on the FULL task (start
altitude 50m, +-8m horizontal spawn) - no curriculum needed.
"""
import sys
import os
import time
import numpy as np
from collections import Counter

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "4_neural_controller"))

from lunar_lander_env import LunarLanderEnv, STATE_DIM, N_ACTIONS  # noqa: E402
from ppo_agent import PPOAgent  # noqa: E402
from docking_env import propagate_cw, cw_analytic_propagate, solve_cw_rendezvous  # noqa: E402

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


# ---- Test 1: network weights update during training ----
env = LunarLanderEnv(seed=1, start_y=50.0, start_x_range=8.0)
agent = PPOAgent(state_dim=STATE_DIM, action_dim=N_ACTIONS, hidden=32, lr=0.005, seed=2)
w2_before = agent.net.W2.copy()
agent.train_iteration(env, episodes_per_rollout=10)
check("Network weights update during training", not np.allclose(w2_before, agent.net.W2))

# ---- Test 2: full training run on the FULL (non-curriculum) task ----
print("\nTraining PPO on full task (10,000 episodes, ~3 min on CPU)...")
env = LunarLanderEnv(seed=1, start_y=50.0, start_x_range=8.0)
agent = PPOAgent(state_dim=STATE_DIM, action_dim=N_ACTIONS, hidden=32, lr=0.005, seed=2)
t0 = time.time()
outcomes = agent.train(env, iterations=1000, episodes_per_rollout=10, verbose_every=200)
elapsed = time.time() - t0

final_1000 = outcomes[-1000:]
land_rate = Counter(final_1000).get("landed", 0) / len(final_1000) * 100

check(
    "PPO achieves >=60% landing success on the FULL task (not curriculum)",
    land_rate >= 60.0,
    f"(land_rate={land_rate:.1f}% over last 1000 episodes, seed=2)"
)

check("Training completes without divergence/NaN weights", not np.any(np.isnan(agent.net.W1)))

print(f"\nBenchmark: {len(outcomes)} training episodes took {elapsed:.1f}s "
      f"({elapsed/len(outcomes)*1000:.2f} ms/episode)")
print(f"Final outcome breakdown (last 1000 ep): {Counter(final_1000)}")

# ---- Docking (Clohessy-Wiltshire) tests ----
n_cw = 0.0011  # rad/s, typical LEO mean motion
rel_state0 = np.array([0.5, -2.0, 0.1, 0.001, -0.0005, 0.0002])
traj_cw = propagate_cw(rel_state0, duration_s=1800, n=n_cw, dt=1.0)
analytic_final = cw_analytic_propagate(rel_state0, 1800, n_cw)
check("CW RK4 propagation matches closed-form analytic STM",
      np.linalg.norm(traj_cw[-1] - analytic_final) < 1e-9,
      f"(diff={np.linalg.norm(traj_cw[-1] - analytic_final):.2e})")

rel_pos0 = np.array([2.0, -5.0, 0.0])
rel_pos_target = np.array([0.0, 0.0, 0.0])
v0, v_arrival = solve_cw_rendezvous(rel_pos0, rel_pos_target, tof_s=1800.0, n=n_cw)
traj_rdv = propagate_cw(np.concatenate([rel_pos0, v0]), duration_s=1800.0, n=n_cw, dt=1.0)
pos_error = np.linalg.norm(traj_rdv[-1][:3] - rel_pos_target)
check("CW rendezvous solver reaches the docking target position",
      pos_error < 1e-6, f"(position error={pos_error:.2e} km)")

print(f"\n{PASS_COUNT} passed, {FAIL_COUNT} failed")
if FAIL_COUNT == 0:
    print("MODULE 4: PPO VERIFIED WORKING ON THE FULL TASK.")
else:
    print("MODULE 4: FAILURES PRESENT - see detail above")