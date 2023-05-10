import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import InterpolationMode, resize

from envs import SingleProcessEnv, WorldModelEnv
from game.keymap import get_keymap_and_action_names


class PlayerEnv:
    def __init__(self, env: SingleProcessEnv) -> None:
        assert isinstance(env, SingleProcessEnv) or isinstance(env, WorldModelEnv)
        self.env = env
        _, self.action_names = get_keymap_and_action_names()
        self.obs = None
        self._t = None
        self._return = None

    def reset(self):
        obs = self.env.reset()
        self.obs = obs
        self._t = 0
        self._return = 0
        return obs[0]
    
    def collect(self, obs, reward, done):
        pass

    def step(self, act) -> torch.FloatTensor:
        self.obs, reward, done, _ = self.env.step([act])
        self._t += 1
        self._return += reward[0]
        info = {
            'timestep': self._t,
            'action': self.action_names[act],
            'return': self._return,
        }
        return self.obs[0], np.array(reward), np.array(done), info

    def render(self) -> Image.Image:
        return Image.fromarray(self.obs[0])

