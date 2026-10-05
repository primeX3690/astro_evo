# 11_safety_guardrails/safety_envelope.py
"""
Deterministic, physics-derived safety envelope for the lunar lander
(module 4). This is NOT learned - it's the same "suicide burn" stopping-
distance physics every real landing GNC system uses: given the vehicle's
maximum available deceleration, how much altitude do you need left to
stop before you hit the ground at your current descent speed?

    stopping_distance = v_descent^2 / (2 * max_decel)

If stopping_distance >= altitude (with a safety margin), the vehicle
is in a state where NO non-thrust action can avoid a hard impact next
tick, independent of what any learned policy "thinks" is best - this
is a deterministic, verifiable fact about the physics, not a judgment
call the RL agent can be wrong about.
"""
import numpy as np

MOON_GRAVITY = 1.62          # m/s^2, must match lunar_lander_env.py
THRUST_ACCEL = 3.2           # m/s^2, must match lunar_lander_env.py
MAX_NET_DECEL = THRUST_ACCEL - MOON_GRAVITY  # m/s^2 of achievable upward deceleration


def stopping_distance_m(descent_speed_mps, max_decel=MAX_NET_DECEL):
    """How much altitude is needed to brake from this descent speed to 0."""
    descent_speed_mps = max(descent_speed_mps, 0.0)
    return descent_speed_mps ** 2 / (2 * max_decel)


def is_safety_critical(state, margin=0.85, max_decel=MAX_NET_DECEL):
    """
    Returns (is_critical: bool, stopping_distance_m: float, altitude_m: float).

    `margin` < 1 fires the guardrail BEFORE the exact physical limit
    (d_stop == altitude) is reached: critical when d_stop >= margin*altitude.
    This buffer is required because we only act once per DT=0.1s control
    tick - gravity keeps accelerating the vehicle during that tick, so
    waiting until the exact d_stop==altitude boundary leaves zero slack
    and can still result in impact. margin=0.85 means the guardrail
    engages once the required stopping distance reaches 85% of the
    remaining altitude.
    """
    x, y, vx, vy, fuel = state
    descent_speed = -vy if vy < 0 else 0.0
    d_stop = stopping_distance_m(descent_speed, max_decel)
    critical = d_stop >= margin * max(y, 1e-6)
    return critical, d_stop, y
