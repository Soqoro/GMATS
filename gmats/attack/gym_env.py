# gmats/attack/gym_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class GymAttackEnv(gym.Env):
    """
    Wraps RLAttackEnv to expose continuous action space:
      action = [a_trade, a_sent] in [-1, 1]^2
    Observation = flat float32 vector from RLAttackEnv.obs_vec
    """
    metadata = {"render.modes": []}

    def __init__(self, core_env):
        super().__init__()
        self.core = core_env
        obs = self.core.reset()
        vec = obs["obs_vec"]
        self.obs_dim = int(vec.shape[0])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._last_info = {}

    def reset(self, *, seed=None, options=None):
        obs = self.core.reset()
        return obs["obs_vec"], {}

    def step(self, action):
        a_trade = float(np.clip(action[0], -1.0, 1.0))
        a_sent  = float(np.clip(action[1], -1.0, 1.0))
        obs_next, r, done, info = self.core.step(a_trade, a_sent, seed_key="td3")
        self._last_info = info
        if done:
            return np.zeros((self.obs_dim,), dtype=np.float32), float(r), True, False, info
        return obs_next["obs_vec"], float(r), False, False, info

    def render(self):
        pass

    def close(self):
        pass
