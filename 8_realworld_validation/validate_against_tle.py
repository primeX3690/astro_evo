# 8_realworld_validation/validate_against_tle.py
"""
Real-world validation: how close does AstroEvo's own J2-perturbed RK4
propagator (module 1) track a *real* satellite, using SGP4 + a real
NORAD TLE as the independent ground truth?

HONEST NOTE: SGP4 propagates in the TEME frame using its own analytic
mean-element theory (drag term, deep-space terms, etc.), while our
propagator is a numerical two-body+J2 integrator seeded from the TLE's
osculating-ish mean elements in an ECI-like frame. These are NOT the
same frame or the same force model, so we do not expect machine-precision
agreement. What this test demonstrates is that our from-scratch
propagator, given real orbital elements from an operational satellite,
produces the same orbit shape/period/altitude regime as the real object.

Requires: pip install sgp4
"""
import sys
import os
import numpy as np
from sgp4.api import Satrec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "1_orbital_mechanics"))
from keplerian_orbit import keplerian_to_state_vector, MU_EARTH  # noqa: E402
from perturbation_models import two_body_j2_eom  # noqa: E402
from two_body_problem import rk4_step  # noqa: E402

from tle_loader import fetch_tle_celestrak, tle_to_keplerian  # noqa: E402


def validate(norad_id=25544, duration_orbits=3, dt_s=10.0):
    tle = fetch_tle_celestrak(norad_id)
    sat = Satrec.twoline2rv(tle["line1"], tle["line2"])
    elems = tle_to_keplerian(tle["line1"], tle["line2"])

    a = elems["a_km"]
    period_s = 2 * np.pi * np.sqrt(a ** 3 / MU_EARTH)
    duration_s = duration_orbits * period_s

    e0, r0, v0 = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF)
    if e0 != 0:
        raise RuntimeError(f"SGP4 error code {e0} at epoch")
    r0 = np.array(r0)
    v0 = np.array(v0)

    state = np.concatenate([r0, v0])
    t = 0.0
    errors_km = []
    sample_times_min = []

    while t <= duration_s:
        e, r_sgp4, _ = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF + t / 86400.0)
        if e != 0:
            t += dt_s
            continue
        r_sgp4 = np.array(r_sgp4)
        pos_err = np.linalg.norm(state[:3] - r_sgp4)
        errors_km.append(pos_err)
        sample_times_min.append(t / 60.0)

        state = rk4_step(lambda tt, yy: two_body_j2_eom(tt, yy, MU_EARTH), t, state, dt_s)
        t += dt_s

    errors_km = np.array(errors_km)
    return {
        "object_name": tle["name"],
        "norad_id": norad_id,
        "tle_epoch": elems["epoch"],
        "semi_major_axis_km": a,
        "period_min": period_s / 60.0,
        "duration_orbits": duration_orbits,
        "n_samples": len(errors_km),
        "error_at_start_km": errors_km[0] if len(errors_km) else float("nan"),
        "error_at_end_km": errors_km[-1] if len(errors_km) else float("nan"),
        "max_error_km": errors_km.max() if len(errors_km) else float("nan"),
        "mean_error_km": errors_km.mean() if len(errors_km) else float("nan"),
        "times_min": sample_times_min,
        "errors_km": errors_km.tolist(),
    }


if __name__ == "__main__":
    result = validate()
    print(f"Object: {result['object_name']} (NORAD {result['norad_id']})")
    print(f"TLE epoch: {result['tle_epoch']}")
    print(f"a = {result['semi_major_axis_km']:.1f} km, period = {result['period_min']:.2f} min")
    print(f"Propagated {result['duration_orbits']} orbits, {result['n_samples']} samples")
    print(f"Position error vs SGP4: start={result['error_at_start_km']:.4f} km, "
          f"end={result['error_at_end_km']:.3f} km, max={result['max_error_km']:.3f} km, "
          f"mean={result['mean_error_km']:.3f} km")
