# 5_self_improve_loop/meta_optimizer.py
"""Random hyperparameter search - tunes module 3's GA and module 4's PPO, logging every trial."""
import sys
import os
import time
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "3_evolutionary_optimizer"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "4_neural_controller"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "2_symbolic_mission_planner"))

from experiment_logger import ExperimentLogger


def random_search(param_space, objective_fn, n_trials, logger, maximize=True, seed=0, tag=""):
    rng = np.random.default_rng(seed)
    for i in range(n_trials):
        config = {}
        for name, spec in param_space.items():
            if isinstance(spec, tuple):
                low, high = spec
                config[name] = float(rng.uniform(low, high))
            else:
                config[name] = spec[rng.integers(0, len(spec))]
        metrics = objective_fn(config)
        logger.log_trial(config, metrics, tag=tag)
        print(f"  trial {i+1}/{n_trials}: config={config} -> {metrics}")

    best = logger.get_best("objective", maximize=maximize)
    return best["config"], best


def optimize_ppo_lander(n_trials=6, episodes_per_trial=1500, log_path=None, seed=0):
    from lunar_lander_env import LunarLanderEnv, STATE_DIM, N_ACTIONS
    from ppo_agent import PPOAgent
    from collections import Counter

    log_path = log_path or os.path.join(_THIS_DIR, "logs_ppo_search.json")
    logger = ExperimentLogger(log_path)

    def objective(config):
        env = LunarLanderEnv(seed=1, start_y=50.0, start_x_range=8.0)
        agent = PPOAgent(state_dim=STATE_DIM, action_dim=N_ACTIONS,
                          hidden=int(config["hidden"]), lr=config["lr"], seed=seed)
        t0 = time.time()
        outcomes = agent.train(env, iterations=episodes_per_trial // 10, episodes_per_rollout=10)
        elapsed = time.time() - t0
        final = outcomes[-min(300, len(outcomes)):]
        land_rate = Counter(final).get("landed", 0) / len(final) * 100
        return {"objective": land_rate, "landing_rate_pct": land_rate, "train_seconds": elapsed}

    param_space = {"lr": (0.001, 0.01), "hidden": [16, 32, 48]}
    print(f"Searching PPO hyperparameters ({n_trials} trials x {episodes_per_trial} episodes each)...")
    best_config, best_trial = random_search(param_space, objective, n_trials, logger, seed=seed, tag="ppo_lander")
    return best_config, best_trial, logger


def optimize_ga_transfer(n_trials=6, log_path=None, seed=0):
    from mission_spec_parser import parse_mission_goal
    from constraint_solver import hohmann_delta_v
    from genetic_engine import GeneticEngine

    log_path = log_path or os.path.join(_THIS_DIR, "logs_ga_search.json")
    logger = ExperimentLogger(log_path)

    spec = parse_mission_goal("LEO to GEO transfer", max_delta_v_kms=6.0, max_time_s=12 * 3600)
    _, _, hohmann_dv = hohmann_delta_v(spec.r1, spec.r2)

    def objective(config):
        ga = GeneticEngine(spec, population_size=int(config["population_size"]),
                            generations=30, mutation_rate=config["mutation_rate"], seed=seed)
        t0 = time.time()
        _, _, best_info = ga.run()
        elapsed = time.time() - t0
        gap_pct = (best_info["dv_total_kms"] / hohmann_dv - 1) * 100
        return {"objective": -gap_pct, "gap_to_hohmann_pct": gap_pct, "search_seconds": elapsed}

    param_space = {"population_size": [20, 40, 60], "mutation_rate": (0.1, 0.5)}
    print(f"Searching GA hyperparameters ({n_trials} trials)...")
    best_config, best_trial = random_search(param_space, objective, n_trials, logger, seed=seed, tag="ga_transfer")
    return best_config, best_trial, logger