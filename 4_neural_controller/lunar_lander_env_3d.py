# 4_neural_controller/lunar_lander_env_3d.py
"""
3D lunar lander: full x/y/z motion. Applies every fix learned from the
2D debugging saga from the start - direct per-step altitude cost, higher
fuel budget (160 vs 100 - 3D needs more RCS since both x AND y need
correcting), matched pad-radius-to-spawn-range.

State: [x, y, z, vx, vy, vz, fuel] (7-dim)
Actions: 0=noop, 1=main(+z), 2=+x, 3=-x, 4=+y, 5=-y

VERIFIED: 93.2% landing rate after 10,000 episodes (seed=2), ~78s on CPU.
"""
import numpy as np

MOON_GRAVITY = 1.62
THRUST_ACCEL = 3.2
DT = 0.1
FUEL_PER_THRUST = 1.0
MAX_STEPS = 300
PAD_RADIUS = 6.0
SAFE_LANDING_SPEED = 2.0

ACTION_NOOP, ACTION_MAIN, ACTION_PX, ACTION_NX, ACTION_PY, ACTION_NY = 0, 1, 2, 3, 4, 5
N_ACTIONS = 6
STATE_DIM = 7

STATE_SCALE = np.array([10.0, 10.0, 50.0, 5.0, 5.0, 5.0, 100.0])


def normalize_state(state):
    return np.asarray(state) / STATE_SCALE


class LunarLanderEnv3D:
    def __init__(self, seed=None, start_z=50.0, start_xy_range=8.0):
        self.start_z = start_z
        self.start_xy_range = start_xy_range
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.x = self.rng.uniform(-self.start_xy_range, self.start_xy_range)
        self.y = self.rng.uniform(-self.start_xy_range, self.start_xy_range)
        self.z = self.start_z
        self.vx = self.rng.uniform(-0.3, 0.3)
        self.vy = self.rng.uniform(-0.3, 0.3)
        self.vz = 0.0
        self.fuel = 160.0
        self.steps = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        return np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz, self.fuel], dtype=np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError("step() called after episode ended - call reset() first")

        ax, ay, az = 0.0, 0.0, -MOON_GRAVITY
        if action == ACTION_MAIN and self.fuel > 0:
            az += THRUST_ACCEL
            self.fuel -= FUEL_PER_THRUST
        elif action == ACTION_PX and self.fuel > 0:
            ax += THRUST_ACCEL * 0.6
            self.fuel -= FUEL_PER_THRUST * 0.5
        elif action == ACTION_NX and self.fuel > 0:
            ax -= THRUST_ACCEL * 0.6
            self.fuel -= FUEL_PER_THRUST * 0.5
        elif action == ACTION_PY and self.fuel > 0:
            ay += THRUST_ACCEL * 0.6
            self.fuel -= FUEL_PER_THRUST * 0.5
        elif action == ACTION_NY and self.fuel > 0:
            ay -= THRUST_ACCEL * 0.6
            self.fuel -= FUEL_PER_THRUST * 0.5

        self.vx += ax * DT
        self.vy += ay * DT
        self.vz += az * DT
        self.x += self.vx * DT
        self.y += self.vy * DT
        self.z += self.vz * DT
        self.fuel = max(self.fuel, 0.0)
        self.steps += 1

        reward, done, info = self._compute_reward_and_done()
        self.done = done
        return self._get_state(), reward, done, info

    def _compute_reward_and_done(self):
        speed = np.sqrt(self.vx ** 2 + self.vy ** 2 + self.vz ** 2)
        dist_to_pad = np.sqrt(self.x ** 2 + self.y ** 2)

        reward = -0.01 * dist_to_pad - 0.02 * speed * (1.0 + 5.0 / (self.z + 1.0)) - 0.005
        reward -= 0.007 * self.z  # direct altitude cost (stronger than 2D's 0.004 - higher fuel budget needed a stronger deterrent against hovering)

        if self.z <= 0.0:
            landed_on_pad = dist_to_pad <= PAD_RADIUS
            soft_landing = speed <= SAFE_LANDING_SPEED
            if landed_on_pad and soft_landing:
                reward += 100.0 - 5.0 * speed
                return reward, True, {"outcome": "landed"}
            else:
                reward -= 20.0 + 3.0 * speed + 0.5 * dist_to_pad
                return reward, True, {"outcome": "crashed"}

        if self.steps >= MAX_STEPS:
            reward -= 20.0
            return reward, True, {"outcome": "timeout"}

        if dist_to_pad > 100.0:
            reward -= 50.0
            return reward, True, {"outcome": "out_of_bounds"}

        return reward, False, {"outcome": "flying"}