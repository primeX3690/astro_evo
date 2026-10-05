# 11_safety_guardrails/deterministic_fallback.py
"""
Deterministic fallback controller: a plain, hand-derivable "suicide
burn" + bang-bang horizontal corrector. No learning, no randomness -
this is the thing that takes over WHEN (not if) the learned policy's
decision would violate the safety envelope in safety_envelope.py.

This mirrors how real flight software is structured: a simple,
provably-correct controller as the safety-of-flight baseline, with the
more capable (but less provable) autonomy layer only allowed to act
when it's not going to kill the vehicle.
"""
import numpy as np
from safety_envelope import MAX_NET_DECEL, is_safety_critical

ACTION_NOOP, ACTION_MAIN, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2, 3
PAD_X_TOLERANCE = 3.0


def deterministic_fallback_action(state, max_decel=MAX_NET_DECEL):
    """
    Priority order (matches standard GNC cascaded-loop design):
      1. If out of fuel or already on the ground, do nothing.
      2. Vertical safety ALWAYS takes priority: if descent speed exceeds
         what remaining altitude allows us to brake from, fire the main
         thruster - full stop, no exceptions.
      3. Otherwise, correct horizontal drift back toward the pad center.
      4. Otherwise, coast (save fuel).
    """
    x, y, vx, vy, fuel = state
    if fuel <= 0 or y <= 0:
        return ACTION_NOOP

    critical, d_stop, altitude = is_safety_critical(state, margin=0.85, max_decel=max_decel)
    if critical:
        return ACTION_MAIN

    if abs(x) > PAD_X_TOLERANCE:
        return ACTION_LEFT if x > 0 else ACTION_RIGHT

    return ACTION_NOOP
