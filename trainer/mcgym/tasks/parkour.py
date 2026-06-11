"""spiral parkour task, climb to the goal, falling is the enemy"""
from __future__ import annotations

import numpy as np

from mcgym.schema.registry import Registry

from .task import Task

GOAL   = np.array([38.0, 104.0, -64.0], dtype=np.float32)
GOAL_Y = 104.0
DONE_R = 1.5

# reward weights
W_GOAL   = 1.0     # per block of gap closed to the goal, the real objective
W_FALL   = 2.0     # per block of fresh drop below the best height, falls must not pay
W_LOW    = 0.02    # per tick below the goal height, constant pressure upward
W_DONE   = 50.0    # standing at the goal
W_DEATH  = 10.0    # fall damage kill, respawns at the pool
# camera shaping as in wood, no jump term, parkour lives on the jump key
W_CAMERA = 0.0006
W_JERK   = 0.002
W_SWING  = 0.001   # attack does nothing here, stop the arm flailing

# signed degrees per camera bin so jerk catches direction reversals
CAM = np.array([-10.0,  -3.0,   0.0,   3.0,  10.0], dtype=np.float32)


class Parkour(Task):
    """vectorized over all N agents, prev trackers are (N,) arrays"""

    name  = "parkour"
    eplen = 2048

    world  = "worlds/spiral"
    spawn  = (100.5, -60.0, -59.5, 90.0)
    mining = False

    def __init__(self, agents: int, registry: Registry) -> None:
        self._dist    = None
        self._best    = None
        self._drop    = None
        self._ground  = None
        self._success = np.zeros(agents, dtype=bool)
        self._yaw     = np.zeros(agents, dtype=np.float32)
        self._pitch   = np.zeros(agents, dtype=np.float32)

    def _gap(self, obs):
        return np.linalg.norm(obs["pos"] - GOAL[None, :], axis=1).astype(np.float32)

    def _rebase(self, mask, y, dist):
        self._best[mask]   = y[mask]
        self._drop[mask]   = 0.0
        self._ground[mask] = y[mask]
        self._dist[mask]   = dist[mask]

    def reset(self, obs: np.ndarray) -> None:
        y = obs["pos"][:, 1].astype(np.float32)

        self._dist       = self._gap(obs)
        self._best       = y.copy()
        self._drop       = np.zeros_like(y)
        self._ground     = y.copy()
        self._success[:] = False
        self._yaw[:]     = 0.0
        self._pitch[:]   = 0.0

    def done(self, died: np.ndarray) -> np.ndarray:
        return died | self._success

    def reward(self, obs, act, died, respawned) -> np.ndarray:
        dist = self._gap(obs)
        y    = obs["pos"][:, 1].astype(np.float32)

        # drop is measured on grounded height only so jump arcs cost nothing,
        # only fresh drop is punished, the climb back is not re paid because
        # goal shaping is a potential
        ground = np.where(obs["on_ground"] == 1, y, self._ground).astype(np.float32)
        best   = np.maximum(self._best, ground)
        drop   = best - ground
        fall   = np.maximum(drop - self._drop, 0.0)

        self._success = dist <= DONE_R
        r = (
            W_GOAL * (self._dist - dist)
            - W_FALL * fall
            - W_LOW * (y < GOAL_Y)
            + W_DONE * self._success
        )

        # camera shaping
        yaw   = CAM[act[:, 4]]
        pitch = CAM[act[:, 5]]
        jerk  = np.abs(yaw - self._yaw) + np.abs(pitch - self._pitch)
        vel   = np.abs(yaw) + np.abs(pitch)
        r = r - W_CAMERA * vel - W_JERK * jerk - W_SWING * act[:, 6]

        # zero the respawn frame, its cross episode delta is spurious, then
        # override the death frame with the penalty
        r = r.astype(np.float32)
        r[respawned] = 0.0
        r[died] = -W_DEATH

        self._dist   = dist
        self._best   = best
        self._drop   = drop
        self._ground = ground
        self._yaw   = yaw.astype(np.float32)
        self._pitch = pitch.astype(np.float32)
        self._rebase(died | respawned, y, dist)
        self._yaw[died]   = 0.0
        self._pitch[died] = 0.0

        return r

    def metric(self, obs: np.ndarray) -> np.ndarray:
        if self._best is None:
            return obs["pos"][:, 1].astype(np.float32)
        return self._best
