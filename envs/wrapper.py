import gym
from typing import Tuple
import numpy as np
from PIL import Image


class ResizeObsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, size: Tuple[int, int]) -> None:
        gym.ObservationWrapper.__init__(self, env)
        self.size = tuple(size)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(size[0], size[1], 3), dtype=np.uint8)

    def resize(self, obs: np.ndarray):
        img = Image.fromarray(obs)
        img = img.resize(self.size, Image.BILINEAR)
        return np.array(img)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return self.resize(observation)
    
    def step(self, action):
        observation, reward, terminated, truncated = self.env.step(action)
        return self.observation(observation), reward, terminated, truncated

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env : gym.Env, noop_max : int):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(0, self.noop_max + 1)
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)





