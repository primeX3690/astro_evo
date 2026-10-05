# tests/test_module14.py
"""Module 14: collision screening + avoidance maneuver (reuses module 4's CW docking math)."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "14_collision_avoidance"))
from conjunction_screening import screen_conjunction
from avoidance_maneuver import compute_avoidance_maneuver

passed, failed = 0, 0
def check(name, cond, detail=""):
    global passed, failed
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if cond: passed += 1
    else: failed += 1

if __name__ == "__main__":
    # LEO mean motion (~6793 km semi-major axis, matches modules 1/8's defaults)
    MU_EARTH = 398600.4418
    a_km = 6793.0
    n = np.sqrt(MU_EARTH / a_km ** 3)  # rad/s

    # --- Scenario: a secondary object 2 km ahead in along-track, drifting
    # inward - a classic close-approach geometry that WILL result in a
    # sub-km miss distance within the next orbit if uncorrected. ---
    rel_pos0_km = np.array([0.0, 2.0, 0.0])      # 2 km ahead, along-track
    rel_vel0_kms = np.array([0.0020, -0.00275, 0.0])  # closing to a real sub-km pass

    screen_duration_s = 6000.0  # a bit over one LEO orbit
    result = screen_conjunction(rel_pos0_km, rel_vel0_kms, n, screen_duration_s,
                                 hard_body_radius_km=0.02, safety_buffer_km=1.0)

    check("Conjunction screening finds a closest-approach point within the window",
          0 < result["tca_s"] < screen_duration_s, f"(TCA={result['tca_s']:.0f}s)")
    check("This scenario is correctly flagged as an actual conjunction (miss < threshold)",
          result["is_conjunction"],
          f"(miss={result['miss_distance_km']:.4f} km, threshold={result['threshold_km']:.2f} km)")

    # --- Compute and apply a cross-track avoidance maneuver ---
    maneuver = compute_avoidance_maneuver(
        rel_pos0_km, rel_vel0_kms, n, tca_s=result["tca_s"],
        safe_miss_distance_km=5.0, burn_lead_time_s=0.0, direction="cross_track",
    )

    # The achieved 3D miss distance is the hypotenuse of whatever in-plane
    # (radial/along-track) miss distance already existed PLUS the added
    # cross-track offset - it will be >= the 5 km cross-track push itself,
    # not exactly 5 km (that would only hold if the in-plane miss were zero).
    check("Avoidance maneuver's cross-track push achieves at least the targeted 5 km separation",
          maneuver["achieved_miss_distance_km"] >= 5.0 - 1e-6,
          f"(achieved={maneuver['achieved_miss_distance_km']:.4f} km, target cross-track=5.0 km)")
    check("Achieved miss distance isn't absurdly larger than the 5 km target (sane magnitude)",
          maneuver["achieved_miss_distance_km"] < 5.5,
          f"(achieved={maneuver['achieved_miss_distance_km']:.4f} km)")

    check("Maneuver meaningfully increases miss distance vs. the unmaneuvered case",
          maneuver["achieved_miss_distance_km"] > maneuver["unmaneuvered_miss_distance_km"] + 1.0,
          f"(before={maneuver['unmaneuvered_miss_distance_km']:.4f} km, "
          f"after={maneuver['achieved_miss_distance_km']:.4f} km)")

    check("Delta-V required is realistic for a LEO avoidance burn (<0.05 km/s)",
          maneuver["delta_v_magnitude_kms"] < 0.05,
          f"(delta_v={maneuver['delta_v_magnitude_kms']*1000:.2f} m/s)")

    check("Delta-V is NOT exactly zero (a real maneuver is actually happening)",
          maneuver["delta_v_magnitude_kms"] > 1e-6)

    # --- A second scenario with a wide initial separation must NOT be flagged ---
    rel_pos0_safe = np.array([0.0, 50.0, 0.0])
    rel_vel0_safe = np.array([0.0, 0.0, 0.0])
    result_safe = screen_conjunction(rel_pos0_safe, rel_vel0_safe, n, screen_duration_s,
                                      hard_body_radius_km=0.02, safety_buffer_km=1.0)
    check("A genuinely safe, well-separated case is correctly NOT flagged",
          not result_safe["is_conjunction"],
          f"(miss={result_safe['miss_distance_km']:.2f} km)")

    print(f"\n{passed} passed, {failed} failed")
    if failed == 0:
        print("MODULE 14: COLLISION AVOIDANCE VERIFIED (screening + minimum-delta-V cross-track maneuver, "
              "built on module 4's existing CW relative-motion solver)")
    sys.exit(1 if failed else 0)
