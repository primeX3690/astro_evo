# 10_ground_ops/pass_predictor.py
"""Ground-station contact-window (pass) prediction: AOS/LOS/peak elevation."""
import numpy as np
from datetime import timedelta
from groundstation import GroundStation  # noqa: F401


def predict_passes(times_s, states_eci_km, epoch_utc, ground_station, min_elevation_deg=10.0):
    elevations = np.zeros(len(times_s))
    azimuths = np.zeros(len(times_s))
    ranges_km = np.zeros(len(times_s))

    for idx, (t, state) in enumerate(zip(times_s, states_eci_km)):
        r_eci = state[:3]
        dt_utc = epoch_utc + timedelta(seconds=float(t))
        az, el, rng = ground_station.look_angles(r_eci, dt_utc)
        elevations[idx] = el
        azimuths[idx] = az
        ranges_km[idx] = rng

    visible = elevations >= min_elevation_deg
    passes = []
    in_pass = False
    start_idx = None

    for i in range(len(times_s)):
        if visible[i] and not in_pass:
            in_pass = True
            start_idx = i
        ending_now = in_pass and (not visible[i] or i == len(times_s) - 1)
        if ending_now:
            window = slice(start_idx, i + 1) if visible[i] else slice(start_idx, i)
            seg_el = elevations[window]
            seg_t = times_s[window]
            if len(seg_el) == 0:
                in_pass = False
                continue
            peak_local = int(np.argmax(seg_el))
            passes.append({
                "aos_s": float(seg_t[0]),
                "los_s": float(seg_t[-1]),
                "aos_time": epoch_utc + timedelta(seconds=float(seg_t[0])),
                "los_time": epoch_utc + timedelta(seconds=float(seg_t[-1])),
                "duration_s": float(seg_t[-1] - seg_t[0]),
                "max_elevation_deg": float(seg_el[peak_local]),
                "max_elevation_time": epoch_utc + timedelta(seconds=float(seg_t[peak_local])),
            })
            in_pass = False

    return passes, {"elevation_deg": elevations, "azimuth_deg": azimuths, "range_km": ranges_km}
