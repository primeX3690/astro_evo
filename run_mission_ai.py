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


def run_validate_tle(config, norad_id, orbits):
    sys.path.insert(0, os.path.join(_THIS_DIR, "8_realworld_validation"))
    from validate_against_tle import validate
    cfg = config.get("tle_validation", {})
    norad_id = norad_id if norad_id is not None else cfg.get("norad_id", 25544)
    orbits = orbits if orbits is not None else cfg.get("orbits", 3)
    result = validate(norad_id=norad_id, duration_orbits=orbits)
    print(f"\nObject: {result['object_name']} (NORAD {result['norad_id']})")
    print(f"TLE epoch: {result['tle_epoch']}")
    print(f"a = {result['semi_major_axis_km']:.1f} km, period = {result['period_min']:.2f} min")
    print(f"Position error vs SGP4: max={result['max_error_km']:.3f} km, mean={result['mean_error_km']:.3f} km")


def _propagate_orbit_j2(orbit_name, inclination_deg, duration_hours, dt_seconds=60.0):
    sys.path.insert(0, os.path.join(_THIS_DIR, "1_orbital_mechanics"))
    sys.path.insert(0, os.path.join(_THIS_DIR, "2_symbolic_mission_planner"))
    import numpy as np
    from keplerian_orbit import keplerian_to_state_vector, MU_EARTH
    from perturbation_models import two_body_j2_eom
    from two_body_problem import rk4_step
    from mission_spec_parser import ORBIT_LIBRARY

    a = ORBIT_LIBRARY[orbit_name]
    r0, v0 = keplerian_to_state_vector(a=a, e=0.001, i=inclination_deg, raan=0, argp=0, nu=0)
    state = np.concatenate([r0, v0])
    duration_s = duration_hours * 3600.0
    n_steps = int(duration_s / dt_seconds)
    times_s = np.arange(n_steps) * dt_seconds
    states = np.zeros((n_steps, 6))
    states[0] = state
    for k in range(1, n_steps):
        state = rk4_step(lambda t, y: two_body_j2_eom(t, y, MU_EARTH), times_s[k - 1], state, dt_seconds)
        states[k] = state
    return times_s, states


def run_export_oem(config, orbit, inclination, duration_hours, dt_seconds, output, object_name, object_id):
    sys.path.insert(0, os.path.join(_THIS_DIR, "9_ccsds_interop"))
    from ccsds_oem import write_oem
    import datetime as dt
    cfg = config.get("oem_export", {})
    orbit = orbit or cfg.get("orbit", "LEO")
    inclination = inclination if inclination is not None else cfg.get("inclination_deg", 51.6)
    duration_hours = duration_hours if duration_hours is not None else cfg.get("duration_hours", 2.0)
    dt_seconds = dt_seconds if dt_seconds is not None else cfg.get("dt_seconds", 60.0)
    output = output or cfg.get("output_path", "mission_output.oem")
    object_name = object_name or cfg.get("object_name", "ASTROEVO_SAT")
    object_id = object_id or cfg.get("object_id", "2026-001A")
    times_s, states = _propagate_orbit_j2(orbit, inclination, duration_hours, dt_seconds)
    epoch = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    write_oem(times_s, states, epoch, object_name, object_id, output_path=output)
    print(f"\nPropagated {orbit} orbit ({inclination} deg incl.) for {duration_hours} hr, {len(times_s)} points")
    print(f"CCSDS OEM ephemeris written to: {output}")


def run_ground_pass(config, orbit, inclination, duration_hours, station_name,
                     station_lat, station_lon, station_alt, min_elevation):
    sys.path.insert(0, os.path.join(_THIS_DIR, "10_ground_ops"))
    from groundstation import GroundStation
    from pass_predictor import predict_passes
    from link_budget import evaluate_link
    import datetime as dt
    cfg = config.get("ground_ops", {})
    orbit = orbit or cfg.get("orbit", "LEO")
    inclination = inclination if inclination is not None else cfg.get("inclination_deg", 51.6)
    duration_hours = duration_hours if duration_hours is not None else cfg.get("duration_hours", 24.0)
    station_name = station_name or cfg.get("station_name", "Gorakhpur")
    station_lat = station_lat if station_lat is not None else cfg.get("station_lat_deg", 26.7606)
    station_lon = station_lon if station_lon is not None else cfg.get("station_lon_deg", 83.3732)
    station_alt = station_alt if station_alt is not None else cfg.get("station_alt_km", 0.084)
    min_elevation = min_elevation if min_elevation is not None else cfg.get("min_elevation_deg", 10.0)

    times_s, states = _propagate_orbit_j2(orbit, inclination, duration_hours)
    epoch = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    gs = GroundStation(station_name, station_lat, station_lon, station_alt)
    passes, geom = predict_passes(times_s, states, epoch, gs, min_elevation_deg=min_elevation)
    print(f"\n{orbit} orbit ({inclination} deg incl.), station: {station_name}, "
          f"{duration_hours} hr window, {min_elevation} deg mask")
    print(f"Passes found: {len(passes)}")
    for idx, p in enumerate(passes, start=1):
        print(f"\nPass {idx}: AOS={p['aos_time']}, LOS={p['los_time']}, "
              f"duration={p['duration_s']:.0f}s, max_elevation={p['max_elevation_deg']:.1f} deg")
        min_range_km = min(geom["range_km"][i] for i in range(len(times_s))
                            if p["aos_s"] <= times_s[i] <= p["los_s"])
        link = evaluate_link(
            tx_power_w=cfg.get("tx_power_w", 1.0), tx_antenna_gain_dbi=cfg.get("tx_antenna_gain_dbi", 0.0),
            tx_line_loss_db=cfg.get("tx_line_loss_db", 0.5), range_km=min_range_km,
            freq_hz=cfg.get("freq_hz", 437e6), rx_antenna_gain_dbi=cfg.get("rx_antenna_gain_dbi", 15.0),
            rx_system_noise_temp_k=cfg.get("rx_system_noise_temp_k", 600.0),
            rx_line_loss_db=cfg.get("rx_line_loss_db", 0.5), data_rate_bps=cfg.get("data_rate_bps", 9600),
            required_eb_n0_db=cfg.get("required_eb_n0_db", 10.0),
        )
        status = "LINK CLOSES" if link["link_closes"] else "LINK DOES NOT CLOSE"
        print(f"  At closest range ({min_range_km:.0f} km): margin={link['margin_db']:.2f} dB -> {status}")


def run_lander_safe(config, episodes):
    """Runs the lander with the module 11 deterministic safety guardrail
    wrapping the PPO agent's action selection - demonstrates the hybrid
    RL + physics-guardrail pattern end to end."""
    sys.path.insert(0, os.path.join(_THIS_DIR, "4_neural_controller"))
    sys.path.insert(0, os.path.join(_THIS_DIR, "11_safety_guardrails"))
    import numpy as np
    from lunar_lander_env import LunarLanderEnv, STATE_DIM, N_ACTIONS, SAFE_LANDING_SPEED
    from ppo_agent import PPOAgent, normalize_state
    from guarded_policy import SafetyGuardedPolicy

    ppo_cfg = config["ppo"]
    lander_cfg = config["lander"]
    env = LunarLanderEnv(seed=1, start_y=lander_cfg["start_altitude_m"], start_x_range=lander_cfg["start_x_range_m"])
    agent = PPOAgent(state_dim=STATE_DIM, action_dim=N_ACTIONS, hidden=ppo_cfg["hidden_size"],
                      lr=ppo_cfg["learning_rate"], gamma=ppo_cfg["gamma"], lam=ppo_cfg["gae_lambda"],
                      clip_eps=ppo_cfg["clip_epsilon"], epochs=ppo_cfg["epochs_per_update"],
                      minibatch_size=ppo_cfg["minibatch_size"], seed=2)
    iterations = max(1, episodes // ppo_cfg["episodes_per_rollout"])
    agent.train(env, iterations=iterations, episodes_per_rollout=ppo_cfg["episodes_per_rollout"],
                verbose_every=max(1, iterations // 10))

    def rl_propose(state):
        probs, _, _, _ = agent.net.forward(normalize_state(state)[None, :])
        return int(np.argmax(probs[0]))

    guard = SafetyGuardedPolicy(rl_propose)
    n_eval = 100
    hard_crashes, overrides_used = 0, 0
    for ep in range(n_eval):
        env.rng = np.random.default_rng(5000 + ep)
        state = env.reset()
        for _ in range(300):
            action, overridden = guard.act(state)
            if overridden:
                overrides_used += 1
            state, _, done, _ = env.step(action)
            if done:
                break
        if abs(state[1]) <= 0.5 and abs(state[3]) > SAFE_LANDING_SPEED:
            hard_crashes += 1
    print(f"\nGuarded PPO lander over {n_eval} eval episodes: "
          f"{hard_crashes} hard crashes, guardrail overrode {overrides_used} steps "
          f"({guard.override_rate*100:.1f}% of all steps taken)")


def run_serve_api(host, port):
    api_dir = os.path.join(_THIS_DIR, "12_api")
    subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--host", host, "--port", str(port)],
                   cwd=api_dir)


def run_stream_telemetry(host, port, rate_hz):
    sys.path.insert(0, os.path.join(_THIS_DIR, "13_realtime_telemetry"))
    from telemetry_server import TelemetryUDPServer
    times_s, states = _propagate_orbit_j2("LEO", 51.6, duration_hours=0.2, dt_seconds=1.0)
    server = TelemetryUDPServer(host=host, port=port)
    print(f"Streaming {len(times_s)} real-time CCSDS telemetry packets to {host}:{port} at {rate_hz} Hz ...")
    server.stream_trajectory(times_s, states, rate_hz=rate_hz)
    server.close()


def main():
    parser = argparse.ArgumentParser(description="AstroEvo master switch")
    parser.add_argument("--mission", default="geo_transfer")
    parser.add_argument("--mode", required=True,
                         choices=["evolve", "lander", "dashboard", "validate-tle", "export-oem",
                                  "ground-pass", "lander-safe", "serve-api", "stream-telemetry"])
    parser.add_argument("--origin", default=None)
    parser.add_argument("--destination", default=None)
    parser.add_argument("--max-dv", type=float, default=None)
    parser.add_argument("--max-time-hr", type=float, default=None)
    parser.add_argument("--episodes", type=int, default=10000)

    parser.add_argument("--norad-id", type=int, default=None)
    parser.add_argument("--orbits", type=int, default=None)

    parser.add_argument("--orbit", default=None, choices=[None, "LEO", "MEO", "GEO"])
    parser.add_argument("--inclination", type=float, default=None)
    parser.add_argument("--duration-hr", type=float, default=None)
    parser.add_argument("--dt-seconds", type=float, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--object-name", default=None)
    parser.add_argument("--object-id", default=None)

    parser.add_argument("--station-name", default=None)
    parser.add_argument("--station-lat", type=float, default=None)
    parser.add_argument("--station-lon", type=float, default=None)
    parser.add_argument("--station-alt", type=float, default=None)
    parser.add_argument("--min-elevation", type=float, default=None)

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--telemetry-port", type=int, default=52001)
    parser.add_argument("--rate-hz", type=float, default=10.0)

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
    elif args.mode == "validate-tle":
        run_validate_tle(config, args.norad_id, args.orbits)
    elif args.mode == "export-oem":
        run_export_oem(config, args.orbit, args.inclination, args.duration_hr,
                        args.dt_seconds, args.output, args.object_name, args.object_id)
    elif args.mode == "ground-pass":
        run_ground_pass(config, args.orbit, args.inclination, args.duration_hr,
                         args.station_name, args.station_lat, args.station_lon,
                         args.station_alt, args.min_elevation)
    elif args.mode == "lander-safe":
        run_lander_safe(config, args.episodes)
    elif args.mode == "serve-api":
        run_serve_api(args.host, args.port)
    elif args.mode == "stream-telemetry":
        run_stream_telemetry(args.host.replace("0.0.0.0", "127.0.0.1"), args.telemetry_port, args.rate_hz)


if __name__ == "__main__":
    main()