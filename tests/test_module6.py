"""
test_module6.py
Tests module 6: rocket_model's Tsiolkovsky math, physics_engine's RK4
powered-flight integration, satellite_model's power budget, EKF, and
orbit_visualizer generating real plots.
"""
import sys
import os
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "6_simulation_hub"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "1_orbital_mechanics"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "2_symbolic_mission_planner"))

from rocket_model import RocketModel  # noqa: E402
from physics_engine import propagate_powered_burn, propagate_coast  # noqa: E402
from satellite_model import SatelliteModel, SpinStabilizedAttitude  # noqa: E402
from orbit_visualizer import plot_orbit_trajectory, plot_ground_track, plot_delta_v_budget  # noqa: E402
from keplerian_orbit import keplerian_to_state_vector  # noqa: E402
from mission_spec_parser import parse_mission_goal  # noqa: E402
from constraint_solver import hohmann_delta_v  # noqa: E402

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


# ---- Test 1: rocket_model Tsiolkovsky math ----
rocket = RocketModel(dry_mass_kg=500, propellant_mass_kg=800, isp_s=300, thrust_n=2000)
expected_dv = rocket.exhaust_velocity_kms * np.log(rocket.wet_mass / rocket.dry_mass)
check("RocketModel delta_v_available matches Tsiolkovsky by hand",
      abs(rocket.delta_v_available() - expected_dv) < 1e-9)

# ---- Test 2: physics_engine's RK4 powered burn matches Tsiolkovsky (gravity-free isolation) ----
burn_time = rocket.burn_time_for_delta_v(2.0)
r0 = np.array([7000.0, 0.0, 0.0])
v0 = np.array([0.0, 0.001, 0.0])
traj = propagate_powered_burn(r0, v0, rocket.wet_mass, rocket, burn_time, dt=0.5, mu=0.0)
achieved_dv = np.linalg.norm(traj[-1][3:6]) - np.linalg.norm(v0)
check("RK4 powered-flight integration matches requested delta-V",
      abs(achieved_dv - 2.0) < 0.001, f"(requested=2.0, achieved={achieved_dv:.6f} km/s)")

# ---- Test 3: powered burn WITH gravity shows expected gravity losses ----
from keplerian_orbit import MU_EARTH
r0_leo = np.array([6798.0, 0.0, 0.0])
v0_leo = np.array([0.0, np.sqrt(MU_EARTH / 6798.0), 0.0])
traj_grav = propagate_powered_burn(r0_leo, v0_leo, rocket.wet_mass, rocket, burn_time, dt=0.5)
speed_gain_with_gravity = np.linalg.norm(traj_grav[-1][3:6]) - np.linalg.norm(v0_leo)
check(
    "Powered burn with real gravity shows gravity losses vs. impulsive Tsiolkovsky value",
    speed_gain_with_gravity < 2.0,
    f"(pure Tsiolkovsky=2.0 km/s, with gravity losses={speed_gain_with_gravity:.4f} km/s)"
)

# ---- Test 4: satellite_model power balance ----
sat = SatelliteModel(solar_panel_area_m2=2.0, panel_efficiency=0.28,
                      subsystem_power_draws_w={"comms": 15, "obc": 5, "payload": 30},
                      battery_capacity_wh=100)
expected_gen = 1361.0 * 2.0 * 0.28
check("Solar power generation matches hand calculation",
      abs(sat.solar_power_generated_w(0) - expected_gen) < 1e-9)
check("Power balance in eclipse equals negative of total draw",
      sat.power_balance_w(90) == -sat.total_power_draw_w())

times, batt, min_batt = sat.simulate_orbit_power(orbit_period_s=5400, eclipse_fraction=0.35)
check("Battery never fully depletes with this power margin", min_batt > 0, f"(min_batt={min_batt:.1f} Wh)")

# ---- Test 5: attitude kinematics ----
att = SpinStabilizedAttitude(spin_rate_deg_s=2.0, moment_of_inertia_kg_m2=15.0)
check("Spin-stabilized orientation matches spin_rate * time",
      abs(np.degrees(att.orientation_at(45)) - 90.0) < 1e-9)

# ---- Test 6: generate real plots from module 1 + module 2 data ----
os.makedirs(os.path.join(_THIS_DIR, "_plots"), exist_ok=True)
spec = parse_mission_goal("LEO to GEO transfer")
dv1, dv2, _ = hohmann_delta_v(spec.r1, spec.r2)

r0, v0 = keplerian_to_state_vector(a=spec.r1, e=0.001, i=51.6, raan=0, argp=0, nu=0)
coast_traj = propagate_coast(r0, v0, duration_s=6000, dt=20.0)

path1 = plot_orbit_trajectory(coast_traj, title="LEO Coast Orbit",
                               filepath=os.path.join(_THIS_DIR, "_plots", "orbit.png"))
path2 = plot_ground_track(coast_traj, dt_s=20.0,
                           filepath=os.path.join(_THIS_DIR, "_plots", "ground_track.png"))
path3 = plot_delta_v_budget(dv1, dv2, filepath=os.path.join(_THIS_DIR, "_plots", "delta_v.png"))

check("Orbit trajectory plot file created", os.path.exists(path1) and os.path.getsize(path1) > 5000)
check("Ground track plot file created", os.path.exists(path2) and os.path.getsize(path2) > 5000)
check("Delta-V budget plot file created", os.path.exists(path3) and os.path.getsize(path3) > 5000)

# ---- Extended Kalman Filter: verifies estimate error is meaningfully lower than raw noise ----
from kalman_filter import ExtendedKalmanFilter  # noqa: E402
from two_body_problem import rk4_step, two_body_eom  # noqa: E402

rng = np.random.default_rng(42)
true_state = np.concatenate([r0, v0])
dt_ekf, n_steps, meas_noise_std = 10.0, 200, 2.0

init_guess = true_state.copy()
init_guess[:3] += rng.normal(0, meas_noise_std, 3)
ekf = ExtendedKalmanFilter(
    init_guess, np.eye(6) * 10.0, np.eye(6) * 1e-6, np.eye(3) * meas_noise_std ** 2, mu=MU_EARTH
)

raw_errors, ekf_errors = [], []
state = true_state.copy()


def eom_ekf(t, y):
    return two_body_eom(t, y, MU_EARTH)


for _ in range(n_steps):
    state = rk4_step(eom_ekf, 0, state, dt_ekf)
    noisy_measurement = state[:3] + rng.normal(0, meas_noise_std, 3)
    est_state = ekf.step(dt_ekf, noisy_measurement)
    raw_errors.append(np.linalg.norm(noisy_measurement - state[:3]))
    ekf_errors.append(np.linalg.norm(est_state[:3] - state[:3]))

raw_rms = np.sqrt(np.mean(np.array(raw_errors[10:]) ** 2))
ekf_rms = np.sqrt(np.mean(np.array(ekf_errors[10:]) ** 2))
check(
    "EKF estimate error is meaningfully lower than raw measurement noise",
    ekf_rms < raw_rms * 0.7,
    f"(raw={raw_rms:.3f}km, ekf={ekf_rms:.3f}km, improvement={(1-ekf_rms/raw_rms)*100:.1f}%)"
)

print(f"\n{PASS_COUNT} passed, {FAIL_COUNT} failed")
if FAIL_COUNT == 0:
    print("MODULE 6: ALL TESTS PASSED, INTEGRATION WITH MODULES 1 & 2 VERIFIED")
else:
    print("MODULE 6: FAILURES PRESENT")