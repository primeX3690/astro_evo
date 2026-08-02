# 4_neural_controller/lunar_lander_env.py
"""2D lunar-landing environment - moon gravity, discrete thrusters, fuel budget."""
import numpy as np
from reward_shaping import compute_reward_and_done

MOON_GRAVITY = 1.62
THRUST_ACCEL = 3.2
DT = 0.1
FUEL_PER_THRUST = 1.0
MAX_STEPS = 300
PAD_X_TOLERANCE = 3.0
SAFE_LANDING_SPEED = 2.0

ACTION_NOOP, ACTION_MAIN, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2, 3
N_ACTIONS = 4
STATE_DIM = 5  # [x, y, vx, vy, fuel]


class LunarLanderEnv:
    def __init__(self, seed=None, start_y=50.0, start_x_range=20.0):
        """start_y/start_x_range enable curriculum use. VERIFIED NECESSARY:
        training directly at y=50,x=+-20 never lands even after 10,000+
        episodes without the fixes documented in this module's README."""
        self.start_y = start_y
        self.start_x_range = start_x_range
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.x = self.rng.uniform(-self.start_x_range, self.start_x_range)
        self.y = self.start_y
        self.vx = self.rng.uniform(-0.3, 0.3)
        self.vy = 0.0
        self.fuel = 100.0
        self.steps = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        return np.array([self.x, self.y, self.vx, self.vy, self.fuel], dtype=np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError("step() called after episode ended - call reset() first")

        y_prev = self.y
        ax, ay = 0.0, -MOON_GRAVITY
        if action == ACTION_MAIN and self.fuel > 0:
            ay += THRUST_ACCEL
            self.fuel -= FUEL_PER_THRUST
        elif action == ACTION_LEFT and self.fuel > 0:
            ax += THRUST_ACCEL * 0.6
            self.fuel -= FUEL_PER_THRUST * 0.5
        elif action == ACTION_RIGHT and self.fuel > 0:
            ax -= THRUST_ACCEL * 0.6
            self.fuel -= FUEL_PER_THRUST * 0.5

        self.vx += ax * DT
        self.vy += ay * DT
        self.x += self.vx * DT
        self.y += self.vy * DT
        self.fuel = max(self.fuel, 0.0)
        self.steps += 1

        reward, done, info = compute_reward_and_done(
            x=self.x, y=self.y, vx=self.vx, vy=self.vy, fuel=self.fuel,
            steps=self.steps, y_prev=y_prev
        )
        self.done = done
        return self._get_state(), reward, done, info
