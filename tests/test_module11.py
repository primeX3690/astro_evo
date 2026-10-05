# tests/test_module11.py
"""Module 11: deterministic safety guardrail wrapping the lander policy."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "4_neural_controller"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "11_safety_guardrails"))
from lunar_lander_env import LunarLanderEnv, ACTION_NOOP, N_ACTIONS, SAFE_LANDING_SPEED
from guarded_policy import SafetyGuardedPolicy

passed, failed = 0, 0
def check(name, cond, detail=""):
    global passed, failed
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if cond: passed += 1
    else: failed += 1


def run_episode(env, action_fn, max_steps=300):
    """action_fn(state) -> action. Returns final (vy, y, crashed_hard, steps)."""
    state = env.reset()
    for _ in range(max_steps):
        action = action_fn(state)
        state, reward, done, info = env.step(action)
        if done:
            break
    vy_final = state[3]
    y_final = state[1]
    hard_crash = y_final <= 0.5 and abs(vy_final) > SAFE_LANDING_SPEED
    return vy_final, y_final, hard_crash


if __name__ == "__main__":
    N_EPISODES = 30

    # --- Deliberately bad policy: never fires any thruster ---
    def do_nothing_policy(state):
        return ACTION_NOOP

    # 1) WITHOUT the guardrail: do-nothing policy must crash hard every time
    #    (nothing is counteracting gravity - this is the expected failure mode).
    env = LunarLanderEnv(seed=1)
    crashes_unguarded = 0
    for ep in range(N_EPISODES):
        env.rng = np.random.default_rng(100 + ep)
        vy, y, crashed = run_episode(env, do_nothing_policy)
        if crashed:
            crashes_unguarded += 1
    check("Unguarded do-nothing policy crashes hard on (nearly) every episode",
          crashes_unguarded >= N_EPISODES - 1,
          f"({crashes_unguarded}/{N_EPISODES} hard crashes - expected, proves the test is meaningful)")

    # 2) WITH the guardrail wrapping the SAME bad policy: the deterministic
    #    suicide-burn fallback must take over and prevent hard crashes,
    #    purely from physics, with zero learned intelligence involved.
    env2 = LunarLanderEnv(seed=1)
    crashes_guarded = 0
    override_rates = []
    for ep in range(N_EPISODES):
        env2.rng = np.random.default_rng(100 + ep)
        guard = SafetyGuardedPolicy(do_nothing_policy, margin=0.85)
        vy, y, crashed = run_episode(env2, lambda s: guard.act(s)[0])
        override_rates.append(guard.override_rate)
        if crashed:
            crashes_guarded += 1
    check("Guardrail prevents hard crashes even with a policy that never acts",
          crashes_guarded == 0,
          f"({crashes_guarded}/{N_EPISODES} hard crashes, vs {crashes_unguarded}/{N_EPISODES} unguarded)")

    check("Guardrail only intervenes near the ground (override rate < 50% of steps)",
          np.mean(override_rates) < 0.5,
          f"(mean override rate={np.mean(override_rates)*100:.1f}% of steps)")

    # 3) The guardrail must NOT interfere with a policy that is ALREADY
    #    conservative on its own (independent logic from the guardrail's own
    #    fallback - a simple "always brake gently once descending" rule) -
    #    override rate should be low since this policy never lets descent
    #    speed build up anywhere near the critical stopping-distance boundary.
    def gentle_brake_policy(state):
        x, y, vx, vy, fuel = state
        if vy < -0.5 and fuel > 0:
            return 1  # ACTION_MAIN
        if abs(x) > 3.0 and fuel > 0:
            return 2 if x > 0 else 3  # ACTION_LEFT / ACTION_RIGHT
        return 0  # ACTION_NOOP

    env3 = LunarLanderEnv(seed=2)
    guard2 = SafetyGuardedPolicy(gentle_brake_policy, margin=0.85)
    vy3, y3, crashed3 = run_episode(env3, lambda s: guard2.act(s)[0])
    check("Guardrail stays mostly out of the way of an independently-conservative policy",
          guard2.override_rate < 0.3,
          f"(override_rate={guard2.override_rate*100:.1f}%, final vy={vy3:.2f} m/s)")
    check("Conservative policy + guardrail lands within safe speed, no hard crash",
          not crashed3 and abs(vy3) <= SAFE_LANDING_SPEED + 0.5,
          f"(final descent speed={abs(vy3):.2f} m/s, safe limit={SAFE_LANDING_SPEED}, crashed={crashed3})")

    print(f"\n{passed} passed, {failed} failed")
    if failed == 0:
        print("MODULE 11: SAFETY GUARDRAIL VERIFIED (deterministic fallback prevents hard crashes "
              "independent of learned-policy quality)")
    sys.exit(1 if failed else 0)
