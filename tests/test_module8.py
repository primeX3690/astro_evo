import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "8_realworld_validation"))
from validate_against_tle import validate
passed, failed = 0, 0
def check(name, cond, detail=""):
    global passed, failed
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if cond: passed += 1
    else: failed += 1
if __name__ == "__main__":
    result = validate(norad_id=25544, duration_orbits=3, dt_s=10.0)
    check("TLE loaded with a real object name", isinstance(result["object_name"], str) and len(result["object_name"]) > 0, f"(name='{result['object_name']}')")
    check("Semi-major axis in real LEO range (6600-7000 km)", 6600 <= result["semi_major_axis_km"] <= 7000, f"(a={result['semi_major_axis_km']:.1f} km)")
    check("Orbital period in real ISS range (90-95 min)", 90 <= result["period_min"] <= 95, f"(T={result['period_min']:.2f} min)")
    check("Our propagator matches SGP4 to <2 km over 3 full orbits", result["max_error_km"] < 2.0, f"(max_error={result['max_error_km']:.3f} km)")
    check("Error stays bounded, not diverging", result["mean_error_km"] < result["max_error_km"] * 5, f"(mean={result['mean_error_km']:.3f})")
    print(f"\n{passed} passed, {failed} failed")
    if failed == 0: print("MODULE 8: REAL-WORLD TLE VALIDATION PASSED")
    sys.exit(1 if failed else 0)
