# 8_realworld_validation/tle_loader.py
"""
Load real NORAD Two-Line Element (TLE) sets, either live from Celestrak
or from a bundled offline snapshot (so tests still run with no internet).

TLE data is free and public (Celestrak, celestrak.org) - no auth needed.
"""
import urllib.request
from datetime import datetime, timedelta, timezone

# Offline fallback snapshot - real ISS (ZARYA) TLE, NORAD ID 25544.
# Bundled so validation still runs without network access; refresh
# periodically by calling fetch_tle_celestrak() when online, since TLEs
# decay in accuracy after ~1-2 weeks from their epoch.
FALLBACK_TLE = {
    "name": "ISS (ZARYA)",
    "line1": "1 25544U 98067A   24045.51782528  .00016717  00000-0  30412-3 0  9994",
    "line2": "2 25544  51.6416 122.3927 0004056  38.4136  62.6832 15.50351663438520",
}


def fetch_tle_celestrak(norad_id=25544, timeout_s=6):
    """
    Fetch a live TLE for a given NORAD catalog ID from Celestrak's free
    API. Returns dict(name, line1, line2). Falls back to the bundled
    offline snapshot on any network failure (no internet, blocked
    egress, Celestrak down, etc.) so this never hard-crashes a pipeline.
    """
    url = (
        "https://celestrak.org/NORAD/elements/gp.php"
        f"?CATNR={norad_id}&FORMAT=TLE"
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8").strip().splitlines()
        if len(text) >= 3:
            return {"name": text[0].strip(), "line1": text[1], "line2": text[2]}
    except Exception as e:
        print(f"[tle_loader] Celestrak fetch failed ({e}); using bundled fallback TLE.")
    return dict(FALLBACK_TLE)


def tle_epoch_to_datetime(line1):
    """Decode the TLE epoch field (cols 19-32 of line 1) to a UTC datetime."""
    epoch_str = line1[18:32]
    year2 = int(epoch_str[0:2])
    year = 2000 + year2 if year2 < 57 else 1900 + year2
    day_of_year = float(epoch_str[2:])
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1)


def tle_to_keplerian(line1, line2):
    """
    Parse the classical elements directly out of the TLE lines.
    Returned angles in degrees, semi-major axis in km (mean-motion based,
    Kozai convention - close enough for a cross-check, not meant to
    replace SGP4's own internal element handling).
    """
    MU_EARTH = 398600.4418  # km^3/s^2
    inc = float(line2[8:16])
    raan = float(line2[17:25])
    ecc = float("0." + line2[26:33].strip())
    argp = float(line2[34:42])
    mean_anomaly = float(line2[43:51])
    mean_motion_rev_day = float(line2[52:63])

    n_rad_s = mean_motion_rev_day * 2 * 3.141592653589793 / 86400.0
    a = (MU_EARTH / n_rad_s ** 2) ** (1.0 / 3.0)

    return {
        "a_km": a,
        "e": ecc,
        "i_deg": inc,
        "raan_deg": raan,
        "argp_deg": argp,
        "mean_anomaly_deg": mean_anomaly,
        "epoch": tle_epoch_to_datetime(line1),
    }
