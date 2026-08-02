"""
test_module7.py
Tests orbital_view.py's data functions, then uses Streamlit's AppTest
framework to actually execute app.py - including clicking the GA
optimizer button (this is what runs the SAME code path your browser's
"Run GA optimizer" button runs, minus the browser).
"""
import sys
import os
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "7_mission_dashboard"))

from orbital_view import get_orbit_points_3d, get_earth_sphere, get_rotating_earth_sphere, build_rotating_earth_figure  # noqa: E402

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


# ---- Test 1: orbit point generator produces a near-circular orbit at the right radius ----
pts = get_orbit_points_3d(a=6798, e=0.001, i=51.6, raan=0, argp=0, n_points=100)
radii = np.linalg.norm(pts, axis=1)
check("Orbit points stay within expected radius range for near-circular orbit",
      6790 <= radii.min() and radii.max() <= 6810, f"(range={radii.min():.1f}-{radii.max():.1f} km)")

# ---- Test 2: Earth sphere mesh has the correct radius everywhere ----
x, y, z = get_earth_sphere(resolution=15)
r = np.sqrt(x**2 + y**2 + z**2)
check("Earth sphere mesh radius is exactly 6371 km everywhere",
      np.allclose(r, 6371.0, atol=1e-6))

# ---- Test: rotating Earth actually rotates + track builds up ----
x0, y0, z0, _ = get_rotating_earth_sphere(0)
x90, y90, z90, _ = get_rotating_earth_sphere(90)
check("Rotating Earth sphere genuinely changes coordinates between angles",
      not np.allclose(x0, x90))
check("Rotated sphere radius still matches Earth radius (rotation preserves shape)",
      np.allclose(np.sqrt(x90**2 + y90**2 + z90**2), 6371.0, atol=1e-6))

pts_anim = get_orbit_points_3d(a=6798, e=0.001, i=51.6, raan=0, argp=0)
fig_anim = build_rotating_earth_figure(pts_anim, n_frames=36)
check("Animation has the requested number of frames", len(fig_anim.frames) == 36)
track_growth = len(fig_anim.frames[35].data[1].x) > len(fig_anim.frames[0].data[1].x)
check("Ground track builds up progressively across frames", track_growth)

# ---- Test 3: dashboard runs with zero exceptions ----
try:
    from streamlit.testing.v1 import AppTest
    app_path = os.path.join(_THIS_DIR, "..", "7_mission_dashboard", "app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)
    check("Dashboard runs top-to-bottom with zero exceptions", len(list(at.exception)) == 0,
          f"({list(at.exception)})")

    # ---- Test 4: clicking "Run GA optimizer" runs the REAL module 3 GA ----
    at.button[0].click().run(timeout=60)
    check("GA button click runs with zero exceptions", len(list(at.exception)) == 0)
    success_msgs = [s.value for s in at.success]
    ga_msg = next((m for m in success_msgs if "GA finished" in m), None)
    check("GA optimizer completes and reports a best fitness", ga_msg is not None, f"({ga_msg})")
    if ga_msg:
        best_fitness = float(ga_msg.split(":")[-1].strip())
        check("Dashboard's live GA result matches module 3's known standalone result",
              abs(best_fitness - (-3.8465)) < 0.01, f"(dashboard={best_fitness}, expected~=-3.8465)")

    # ---- Test 5: PDF report button generates a real file ----
    pdf_buttons = [b for b in at.button if b.label == "Generate PDF report"]
    if pdf_buttons:
        pdf_buttons[0].click().run(timeout=30)
        check("PDF report button runs with zero exceptions", len(list(at.exception)) == 0)
        report_path = os.path.join(_THIS_DIR, "..", "7_mission_dashboard", "_mission_report.pdf")
        check("PDF report file was actually created", os.path.exists(report_path) and os.path.getsize(report_path) > 1000)
        if os.path.exists(report_path):
            os.remove(report_path)
    else:
        check("PDF report button is present after GA run", False)
except ImportError:
    check("Streamlit AppTest available", False, "(streamlit not installed)")

print(f"\n{PASS_COUNT} passed, {FAIL_COUNT} failed")
if FAIL_COUNT == 0:
    print("MODULE 7: ALL TESTS PASSED, DASHBOARD INTEGRATION WITH MODULES 1, 2 & 3 VERIFIED")
else:
    print("MODULE 7: FAILURES PRESENT")