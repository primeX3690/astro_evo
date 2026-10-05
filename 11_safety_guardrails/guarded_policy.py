# 11_safety_guardrails/guarded_policy.py
"""
SafetyGuardedPolicy: wraps ANY action-selection function (a trained PPO
policy, a random policy, a buggy policy - doesn't matter) with the
deterministic safety envelope. This is the "Hybrid Deterministic
RL-Guardrail Controller" pattern: the learned policy proposes, the
physics-based guardrail disposes.

The point is explicit and demonstrable: the guardrail's guarantee does
NOT depend on the learned policy being good. We prove this below by
wrapping a deliberately bad policy (always NOOP) and showing the
guardrail alone prevents hard-impact crashes.
"""
from safety_envelope import is_safety_critical, MAX_NET_DECEL
from deterministic_fallback import deterministic_fallback_action


class SafetyGuardedPolicy:
    def __init__(self, propose_action_fn, max_decel=MAX_NET_DECEL, margin=0.85):
        """propose_action_fn(state) -> int action, from any source (RL, rule-based, random)."""
        self.propose_action_fn = propose_action_fn
        self.max_decel = max_decel
        self.margin = margin
        self.total_steps = 0
        self.override_steps = 0

    def act(self, state):
        self.total_steps += 1
        critical, d_stop, altitude = is_safety_critical(state, margin=self.margin, max_decel=self.max_decel)
        if critical:
            self.override_steps += 1
            return deterministic_fallback_action(state, self.max_decel), True
        return self.propose_action_fn(state), False

    @property
    def override_rate(self):
        return self.override_steps / self.total_steps if self.total_steps else 0.0
