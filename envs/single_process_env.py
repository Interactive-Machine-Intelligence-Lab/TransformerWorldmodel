from typing import Any, Tuple, Dict

import numpy as np

from .done_tracker import DoneTrackerEnv
from config import env_cfg


class SingleProcessEnv(DoneTrackerEnv):
    def __init__(self, env_fn, pid):
        super().__init__(num_envs=1)
        self.env = env_fn(pid[0])
        self.num_actions = self.env.action_space[0].n

    def should_reset(self) -> bool:
        return self.num_envs_done == 1

    def reset(self) -> np.ndarray:
        self.reset_done_tracker()
        obs = self.env.reset()
        for agent_id in obs:
            obs[agent_id] = obs[agent_id][None, ...]
        return obs

    def step(self, action : Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        obs, reward, done, _ = self.env.step(action)  # action is supposed to be dict {0 : ?, 1 : ?}
        self.update_done_tracker(done['__all__'])
        for agent_id in range(env_cfg.agent_num):
            obs[agent_id] = obs[agent_id][None, ...]
            reward[agent_id] = np.array([reward[agent_id]])
            done[agent_id] = np.array([done[agent_id]])
        return obs, reward, done, None

    # def render(self) -> None:
    #     self.env.render()

    def close(self) -> None:
        self.env.close()
