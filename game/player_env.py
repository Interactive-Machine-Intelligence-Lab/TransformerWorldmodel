import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import InterpolationMode, resize

from envs import SingleProcessEnv, WorldModelEnv
from game.keymap import get_keymap_and_action_names


class PlayerEnv:
    def __init__(self, env: SingleProcessEnv, agent_num, agent_id) -> None:
        assert isinstance(env, SingleProcessEnv) or isinstance(env, WorldModelEnv)
        self.env = env
        _, self.action_names = get_keymap_and_action_names()
        self.obs = None
        self._t = None
        self._return = None
        self.agent_id = agent_id
        self.agent_num = agent_num

    def reset(self):
        obs = self.env.reset()
        self.obs = obs[self.agent_id]
        self._t = 0
        self._return = 0
        return obs[0]
    
    def collect(self, obs, reward, done):
        pass

    def step(self, act) -> torch.FloatTensor:
        acts = {}
        for i in range(self.agent_num):
            acts[i] = 0
        acts[self.agent_id] = act
        self.obs, reward, done, _ = self.env.step(acts)

        self.obs, reward, done = self.obs[self.agent_id], reward[self.agent_id], done[self.agent_id]
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

