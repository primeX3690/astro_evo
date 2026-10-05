import sys, os
import numpy as np
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "1_orbital_mechanics"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "10_ground_ops"))
from keplerian_orbit import keplerian_to_state_vector, MU_EARTH
from perturbation_models import two_body_j2_eom
from two_body_problem import rk4_step
from groundstation import GroundStation, geodetic_to_ecef, gmst_rad
from pass_predictor import predict_passes
from link_budget import evaluate_link, free_space_path_loss_db
passed, failed = 0, 0
def check(name, cond, detail=""):
    global passed, failed
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if cond: passed += 1
    else: failed += 1
if __name__ == "__main__":
    gs = GroundStation("Gorakhpur", lat_deg=26.7606, lon_deg=83.3732, alt_km=0.084)
    r_ecef = geodetic_to_ecef(0.0, 0.0, 0.0)
    check("Equator point lands on WGS-84 equatorial radius", abs(np.linalg.norm(r_ecef) - 6378.137) < 0.01)
    epoch = datetime(2000, 1, 1, 12, 0, 0)
    r_overhead_ecef = geodetic_to_ecef(26.7606, 83.3732, 500.0)
    theta = gmst_rad(epoch)
    c, s = np.cos(theta), np.sin(theta)
    R_eci_from_ecef = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    r_overhead_eci = R_eci_from_ecef @ r_overhead_ecef
    az, el, rng = gs.look_angles(r_overhead_eci, epoch)
    check("Satellite directly overhead reads ~90 deg elevation", el > 89.9, f"(el={el:.4f})")
    expected_range = 500.0 - gs.alt_km
    check("Overhead range matches altitude difference", abs(rng - expected_range) < 0.01, f"(range={rng:.4f})")
    r0, v0 = keplerian_to_state_vector(a=6793.0, e=0.001, i=51.6, raan=80, argp=0, nu=0)
    state = np.concatenate([r0, v0])
    dt = 20.0
    duration_s = 6 * 3600
    n_steps = int(duration_s / dt)
    times_s = np.arange(n_steps) * dt
    states = np.zeros((n_steps, 6))
    states[0] = state
    for k in range(1, n_steps):
        state = rk4_step(lambda t, y: two_body_j2_eom(t, y, MU_EARTH), times_s[k-1], state, dt)
        states[k] = state
    epoch2 = datetime(2026, 9, 16, 0, 0, 0)
    passes, geom = predict_passes(times_s, states, epoch2, gs, min_elevation_deg=10.0)
    check("At least one pass found over 6 hours", len(passes) >= 1, f"(found {len(passes)})")
    check("Elevation trace stays within [-90, 90] deg", np.all(geom["elevation_deg"] >= -90) and np.all(geom["elevation_deg"] <= 90))
    if passes:
        p = passes[0]
        check("Pass duration realistic (30s-20min)", 30 <= p["duration_s"] <= 1200, f"(duration={p['duration_s']:.0f}s)")
        check("AOS before LOS", p["aos_s"] < p["los_s"])
        check("Max elevation >= mask", p["max_elevation_deg"] >= 10.0, f"(max_el={p['max_elevation_deg']:.1f})")
    fspl = free_space_path_loss_db(range_km=1000, freq_hz=437e6)
    check("FSPL at 437MHz/1000km matches textbook (~145 dB)", abs(fspl - 145.25) < 0.5, f"(fspl={fspl:.2f})")
    result = evaluate_link(tx_power_w=1.0, tx_antenna_gain_dbi=0.0, tx_line_loss_db=0.5, range_km=1000, freq_hz=437e6,
                            rx_antenna_gain_dbi=15.0, rx_system_noise_temp_k=600.0, rx_line_loss_db=0.5,
                            data_rate_bps=9600, required_eb_n0_db=10.0)
    check("Link budget produces sane C/N0 (0-100 dB-Hz)", 0 < result["c_n0_db_hz"] < 100, f"(C/N0={result['c_n0_db_hz']:.2f})")
    result_2x = evaluate_link(tx_power_w=2.0, tx_antenna_gain_dbi=0.0, tx_line_loss_db=0.5, range_km=1000, freq_hz=437e6,
                               rx_antenna_gain_dbi=15.0, rx_system_noise_temp_k=600.0, rx_line_loss_db=0.5,
                               data_rate_bps=9600, required_eb_n0_db=10.0)
    delta = result_2x["margin_db"] - result["margin_db"]
    check("Doubling TX power raises margin by 3.01 dB", abs(delta - 10*np.log10(2)) < 1e-9, f"(delta={delta:.4f})")
    print(f"\n{passed} passed, {failed} failed")
    if failed == 0: print("MODULE 10: GROUND OPS VERIFIED")
    sys.exit(1 if failed else 0)
