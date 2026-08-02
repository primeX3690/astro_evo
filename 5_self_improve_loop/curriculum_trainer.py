# 5_self_improve_loop/curriculum_trainer.py
"""Generic difficulty-ramp trainer for module 4's lander, using mission_generator's difficulty ramp."""
import sys
import os
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "4_neural_controller"))

from lunar_lander_env import LunarLanderEnv, STATE_DIM, N_ACTIONS  # noqa: E402
from ppo_agent import PPOAgent  # noqa: E402
from mission_generator import generate_lander_scenario  # noqa: E402
from experiment_logger import ExperimentLogger  # noqa: E402


def train_lander_curriculum(n_stages=5, episodes_per_stage=1000, log_path=None, seed=0):
    from collections import Counter

    log_path = log_path or os.path.join(_THIS_DIR, "logs_curriculum.json")
    logger = ExperimentLogger(log_path)

    agent = PPOAgent(state_dim=STATE_DIM, action_dim=N_ACTIONS, hidden=32, lr=0.005, seed=seed)
    difficulties = np.linspace(0.0, 1.0, n_stages)

    stage_results = []
    for stage_idx, difficulty in enumerate(difficulties):
        scenario = generate_lander_scenario(np.random.default_rng(seed), difficulty=difficulty)
        env = LunarLanderEnv(seed=seed, **scenario)
        outcomes = agent.train(env, iterations=episodes_per_stage // 10, episodes_per_rollout=10)
        land_rate = Counter(outcomes[-200:]).get("landed", 0) / min(200, len(outcomes)) * 100

        metrics = {"objective": land_rate, "landing_rate_pct": land_rate, "difficulty": difficulty}
        logger.log_trial({**scenario, "difficulty": difficulty, "stage": stage_idx}, metrics, tag="curriculum_stage")
        stage_results.append((difficulty, land_rate))
        print(f"stage {stage_idx+1}/{n_stages} (difficulty={difficulty:.2f}, "
              f"start_y={scenario['start_y']:.1f}): landing_rate={land_rate:.1f}%")

    return agent, stage_results, logger