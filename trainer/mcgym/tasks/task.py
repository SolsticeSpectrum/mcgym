"""task contract, a task defines everything reward related, the env stays generic"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Task(ABC):
    """per gym instance, trackers are (N,) arrays over that gyms agents,
    a task owns everything that is not game physics and mechanics, reward,
    shaping, episode length, terminal rules, the progress metric"""

    name:  str
    eplen: int

    def done(self, died: np.ndarray) -> np.ndarray:
        """terminal mask for this step, default is terminal on death"""
        return died

    @abstractmethod
    def reset(self, obs: np.ndarray) -> None:
        """rebase trackers on a fresh episode"""

    @abstractmethod
    def reward(self, obs: np.ndarray, act: np.ndarray,
               died: np.ndarray, respawned: np.ndarray) -> np.ndarray:
        """full per step reward, including shaping and death"""

    @abstractmethod
    def metric(self, obs: np.ndarray) -> np.ndarray:
        """progress number per agent for logs and the monitor"""
