# run_mission_ai.py
"""
Master CLI switch. Examples:
  python run_mission_ai.py --mission geo_transfer --mode evolve
  python run_mission_ai.py --mission geo_transfer --mode evolve --origin LEO --destination MEO
  python run_mission_ai.py --mode lander --episodes 2000
  python run_mission_ai.py --mode dashboard

Reads defaults from config/mission_config.yaml; CLI flags override them.
"""
import argparse
import os
import sys
import subprocess
import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config():
    config_path = os.path.join(_THIS_DIR, "config", "mission_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_evolve(config, origin, destination, max_dv, max_time_hr):
    sys.path.insert(0, os.path.join(_THIS_DIR, "2_symbolic_mission_planner"))
    sys.path.insert(0, os.path.join(_THIS_DIR, "3_evolutionary_optimizer"))
    from mission_spec_parser import ORBIT_LIBRARY, MissionSpec
    from genetic_engine import GeneticEngine
    from constraint_solver import hohmann_delta_v

    spec = MissionSpec(
        name=f"{origin}_to_{destination}", r1_km=ORBIT_LIBRARY[origin], r2_km=ORBIT_LIBRARY[destination],
        max_delta_v_kms=max_dv, max_time_s=max_time_hr * 3600,
    )
    ga_cfg = config["genetic_algorithm"]
    ga = GeneticEngine(spec, population_size=ga_cfg["population_size"], generations=ga_cfg["generations"],
                        mutation_rate=ga_cfg["mutation_rate"], elite_count=ga_cfg["elite_count"], seed=ga_cfg["seed"])
    best_gene, best_fitness, best_info = ga.run()
    _, _, hohmann_dv = hohmann_delta_v(spec.r1, spec.r2)

    print(f"\nMission: {spec.name}")
    print(f"GA best transfer: angle={best_gene[0]:.2f} deg, tof={best_gene[1]:.2f} hr")
    print(f"GA delta-V: {best_info['dv_total_kms']:.4f} km/s  (Hohmann optimum: {hohmann_dv:.4f} km/s)")
    print(f"Feasible: {best_info['feasible']}")


def run_lander(config, episodes):
    sys.path.insert(0, os.path.join(_THIS_DIR, "4_neural_controller"))
    from lunar_lander_env import LunarLanderEnv, STATE_DIM, N_ACTIONS
    from ppo_agent import PPOAgent
    from collections import Counter

    ppo_cfg = config["ppo"]
    lander_cfg = config["lander"]
    env = LunarLanderEnv(seed=1, start_y=lander_cfg["start_altitude_m"], start_x_range=lander_cfg["start_x_range_m"])
    agent = PPOAgent(state_dim=STATE_DIM, action_dim=N_ACTIONS, hidden=ppo_cfg["hidden_size"],
                      lr=ppo_cfg["learning_rate"], gamma=ppo_cfg["gamma"], lam=ppo_cfg["gae_lambda"],
                      clip_eps=ppo_cfg["clip_epsilon"], epochs=ppo_cfg["epochs_per_update"],
                      minibatch_size=ppo_cfg["minibatch_size"], seed=2)

    iterations = max(1, episodes // ppo_cfg["episodes_per_rollout"])
    outcomes = agent.train(env, iterations=iterations, episodes_per_rollout=ppo_cfg["episodes_per_rollout"],
                            verbose_every=max(1, iterations // 10))
    final = outcomes[-min(200, len(outcomes)):]
    land_rate = Counter(final).get("landed", 0) / len(final) * 100
    print(f"\nTrained {len(outcomes)} episodes. Landing rate (last {len(final)} ep): {land_rate:.1f}%")


def run_dashboard():
    app_path = os.path.join(_THIS_DIR, "7_mission_dashboard", "app.py")
    subprocess.run(["streamlit", "run", app_path])


def main():
    parser = argparse.ArgumentParser(description="AstroEvo master switch")
    parser.add_argument("--mission", default="geo_transfer")
    parser.add_argument("--mode", required=True, choices=["evolve", "lander", "dashboard"])
    parser.add_argument("--origin", default=None)
    parser.add_argument("--destination", default=None)
    parser.add_argument("--max-dv", type=float, default=None)
    parser.add_argument("--max-time-hr", type=float, default=None)
    parser.add_argument("--episodes", type=int, default=10000)
    args = parser.parse_args()

    config = load_config()

    if args.mode == "evolve":
        origin = args.origin or config["mission"]["origin"]
        destination = args.destination or config["mission"]["destination"]
        max_dv = args.max_dv if args.max_dv is not None else config["mission"]["max_delta_v_kms"]
        max_time_hr = args.max_time_hr if args.max_time_hr is not None else config["mission"]["max_time_hours"]
        run_evolve(config, origin, destination, max_dv, max_time_hr)
    elif args.mode == "lander":
        run_lander(config, args.episodes)
    elif args.mode == "dashboard":
        run_dashboard()


if __name__ == "__main__":
    main()