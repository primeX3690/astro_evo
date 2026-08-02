# 4_neural_controller/reward_shaping.py
"""Reward + terminal-condition logic for lunar_lander_env.py."""
import numpy as np

PAD_X_TOLERANCE = 6.0
SAFE_LANDING_SPEED = 2.0
MAX_STEPS = 300


def compute_reward_and_done(x, y, vx, vy, fuel, steps, y_prev=None):
    speed = np.sqrt(vx ** 2 + vy ** 2)
    dist_to_pad = abs(x)

    reward = -0.01 * dist_to_pad - 0.02 * speed * (1.0 + 5.0 / (y + 1.0)) - 0.005

    # DIRECT (non-telescoping) altitude cost - accumulates every step spent
    # high up, punishing the hover-and-burn-fuel exploit found via
    # trajectory tracing (agent sat at y~50 for 190 steps burning all fuel,
    # then free-fell uncontrolled). A potential-based version doesn't
    # work here since it telescopes to a path-independent total.
    reward -= 0.004 * y

    if y <= 0.0:
        landed_on_pad = dist_to_pad <= PAD_X_TOLERANCE
        soft_landing = speed <= SAFE_LANDING_SPEED
        if landed_on_pad and soft_landing:
            reward += 100.0 - 5.0 * speed
            return reward, True, {"outcome": "landed"}
        else:
            reward -= 20.0 + 3.0 * speed + 0.5 * dist_to_pad
            return reward, True, {"outcome": "crashed"}

    if steps >= MAX_STEPS:
        reward -= 20.0
        return reward, True, {"outcome": "timeout"}

    if abs(x) > 100.0:
        reward -= 50.0
        return reward, True, {"outcome": "out_of_bounds"}

    return reward, False, {"outcome": "flying"}
