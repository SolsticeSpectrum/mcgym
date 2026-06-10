"""Task contract, a task defines everything reward related, the env stays generic."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Task(ABC):
    """Per gym instance, trackers are (N,) arrays over that gym's agents.

    A task owns everything that is not game physics and mechanics, reward,
    shaping, episode length, terminal rules, the progress metric.
    """

    name:  str
    eplen: int

    def done(self, died: np.ndarray) -> np.ndarray:
        """Terminal mask for this step, default is terminal on death."""
        return died

    @abstractmethod
    def reset(self, obs: np.ndarray) -> None:
        """Rebase trackers on a fresh episode."""

    @abstractmethod
    def reward(self, obs: np.ndarray, act: np.ndarray,
               died: np.ndarray, respawned: np.ndarray) -> np.ndarray:
        """Full per step reward, including shaping and death."""

    @abstractmethod
    def metric(self, obs: np.ndarray) -> np.ndarray:
        """Progress number per agent for logs and the monitor."""
