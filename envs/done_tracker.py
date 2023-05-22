import numpy as np


class DoneTrackerEnv:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        # 아직 안끝났으면 0, 방금 끝났으면 1, 1step이전에 끝났다면 2
        self.done_tracker = None
        self.reset_done_tracker()

    def reset_done_tracker(self) -> None:
        self.done_tracker = np.zeros(self.num_envs, dtype=np.uint8)

    def update_done_tracker(self, done: np.ndarray) -> None:
        self.done_tracker = np.clip(2 * self.done_tracker + done, 0, 2)

    @property
    def num_envs_done(self) -> int:
        return (self.done_tracker > 0).sum()

    @property
    def mask_dones(self) -> np.ndarray:
        return np.logical_not(self.done_tracker)

    @property
    def mask_new_dones(self) -> np.ndarray:
        return np.logical_not(self.done_tracker[self.done_tracker <= 1])
