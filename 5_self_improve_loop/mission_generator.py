# 5_self_improve_loop/mission_generator.py
"""Generates randomized orbit-transfer missions (modules 2/3) and lander difficulty settings (module 4)."""
import sys
import os
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "2_symbolic_mission_planner"))
from mission_spec_parser import ORBIT_LIBRARY, MissionSpec  # noqa: E402


def generate_transfer_mission(rng, allow_same=False):
    names = list(ORBIT_LIBRARY.keys())
    origin = names[rng.integers(0, len(names))]
    dest = names[rng.integers(0, len(names))]
    while not allow_same and dest == origin:
        dest = names[rng.integers(0, len(names))]

    max_dv = float(rng.uniform(3.0, 8.0))
    max_time_hr = float(rng.uniform(4.0, 20.0))

    return MissionSpec(
        name=f"{origin}_to_{dest}", r1_km=ORBIT_LIBRARY[origin], r2_km=ORBIT_LIBRARY[dest],
        max_delta_v_kms=max_dv, max_time_s=max_time_hr * 3600,
    )


def generate_lander_scenario(rng, difficulty=1.0):
    """difficulty in [0,1]: 0=trivial (near ground, centered), 1=full task."""
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    start_y = 8.0 + difficulty * (50.0 - 8.0)
    start_x_range = 1.5 + difficulty * (8.0 - 1.5)
    return {"start_y": start_y, "start_x_range": start_x_range}


def generate_mission_batch(n, seed=0):
    rng = np.random.default_rng(seed)
    return [generate_transfer_mission(rng) for _ in range(n)]