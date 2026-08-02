# 2_symbolic_mission_planner/mission_spec_parser.py
"""
mission_spec_parser.py
Converts a human mission goal (e.g. "LEO to GEO transfer") into a formal
MissionSpec object with real orbital parameters, using module 1's constants.
"""
import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE1_DIR = os.path.join(_THIS_DIR, "..", "1_orbital_mechanics")
sys.path.insert(0, _MODULE1_DIR)

from utils import period  # noqa: E402  (module 1 import)

MU_EARTH = 398600.4418  # km^3/s^2, kept consistent with module 1

ORBIT_LIBRARY = {
    "LEO": 6798.0,    # ~420 km altitude, ISS-class
    "MEO": 20200.0,   # GPS-class
    "GEO": 42164.0,   # geostationary
}


class MissionSpec:
    def __init__(self, name, r1_km, r2_km, max_delta_v_kms=None, max_time_s=None):
        self.name = name
        self.r1 = r1_km
        self.r2 = r2_km
        self.max_delta_v = max_delta_v_kms
        self.max_time = max_time_s
        self.period_r1 = period(r1_km)
        self.period_r2 = period(r2_km)

    def __repr__(self):
        return (f"MissionSpec({self.name}: r1={self.r1:.1f}km -> r2={self.r2:.1f}km, "
                f"T1={self.period_r1/60:.1f}min, T2={self.period_r2/60:.1f}min)")


def parse_mission_goal(goal_str, max_delta_v_kms=None, max_time_s=None):
    """Parse 'LEO to GEO transfer' style strings into a MissionSpec."""
    goal_str = goal_str.strip().upper()
    parts = goal_str.replace("TRANSFER", "").split(" TO ")
    if len(parts) != 2:
        raise ValueError(
            f"Could not parse goal '{goal_str}'. Expected format: '<ORIGIN> to <DESTINATION> transfer'"
        )
    origin, dest = parts[0].strip(), parts[1].strip()

    if origin not in ORBIT_LIBRARY or dest not in ORBIT_LIBRARY:
        raise ValueError(
            f"Unknown orbit name(s): {origin}, {dest}. Known orbits: {list(ORBIT_LIBRARY.keys())}"
        )

    return MissionSpec(
        name=f"{origin}_to_{dest}",
        r1_km=ORBIT_LIBRARY[origin],
        r2_km=ORBIT_LIBRARY[dest],
        max_delta_v_kms=max_delta_v_kms,
        max_time_s=max_time_s,
    )