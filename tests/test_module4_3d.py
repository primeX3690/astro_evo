"""
test_module4_3d.py
Trains real PPO on the 3D lunar lander (x, y, z motion - not just x, y
like the original 2D env) directly on the full task.
"""
import sys
import os
import time
import numpy as np
from collections import Counter

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "4_neural_controller"))

from lunar_lander_env_3d import LunarLanderEnv3D, STATE_DIM, N_ACTIONS, normalize_state  # noqa: E402
from ppo_agent import PPOAgent  # noqa: E402

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


# ---- Test 1: weights update ----
env = LunarLanderEnv3D(seed=1, start_z=50.0, start_xy_range=8.0)
agent = PPOAgent(state_dim=STATE_DIM, action_dim=N_ACTIONS, hidden=32, lr=0.005, seed=2,
                  normalize_fn=normalize_state)
w2_before = agent.net.W2.copy()
agent.train_iteration(env, episodes_per_rollout=10)
check("Network weights update during training", not np.allclose(w2_before, agent.net.W2))

# ---- Test 2: full 3D training run ----
print("\nTraining PPO on full 3D lander task (10,000 episodes, ~1.5 min on CPU)...")
env = LunarLanderEnv3D(seed=1, start_z=50.0, start_xy_range=8.0)
agent = PPOAgent(state_dim=STATE_DIM, action_dim=N_ACTIONS, hidden=32, lr=0.005, seed=2,
                  normalize_fn=normalize_state)
t0 = time.time()
outcomes = agent.train(env, iterations=1000, episodes_per_rollout=10, verbose_every=200)
elapsed = time.time() - t0

final_1000 = outcomes[-1000:]
land_rate = Counter(final_1000).get("landed", 0) / len(final_1000) * 100

check(
    "PPO achieves >=70% landing success on the full 3D task",
    land_rate >= 70.0,
    f"(land_rate={land_rate:.1f}% over last 1000 episodes, seed=2)"
)
check("Training completes without divergence/NaN weights", not np.any(np.isnan(agent.net.W1)))

print(f"\nBenchmark: {len(outcomes)} training episodes took {elapsed:.1f}s "
      f"({elapsed/len(outcomes)*1000:.2f} ms/episode)")
print(f"Final outcome breakdown (last 1000 ep): {Counter(final_1000)}")

print(f"\n{PASS_COUNT} passed, {FAIL_COUNT} failed")
if FAIL_COUNT == 0:
    print("MODULE 4 (3D): PPO VERIFIED WORKING ON THE FULL 3D TASK.")
else:
    print("MODULE 4 (3D): FAILURES PRESENT")