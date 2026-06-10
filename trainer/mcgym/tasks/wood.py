"""Gather wood task, dense shaping toward finding and mining logs"""
from __future__ import annotations

import numpy as np

from mcgym.schema import spec
from mcgym.schema.registry import Registry

from .task import Task

# vanilla wood families, mushroom_stem ends in _stem but is fungus not wood
SUFFIXES = ("_log", "_wood", "_stem", "_hyphae")
EXCLUDED = frozenset({"minecraft:mushroom_stem"})

EDGE   = spec.VOXEL_EDGE
RADIUS = spec.PARAMS["voxel_radius"]

# sentinel distances when no log is in the grid, above max possible so shaping stays bounded
NO_LOG     = 16.0
NO_LOG_FAR = 80.0

# reward weights
W_WOOD     = 10.0    # delta wood count, the real objective
W_APPROACH = 0.3     # potential shaping toward nearest log in the near grid
W_FAR      = 0.1     # same on the far shell, pulls toward distant trees
W_FACE     = 0.05    # looking at an in range log
W_ATTACK   = 0.15    # attacking an in range log, sustained mining
W_ANYITEM  = 0.2     # delta total inventory, bootstraps pick something up
W_DEATH    = 10.0    # death penalty, real signal not engineered away
# CAPS style action rate shaping (arXiv 2012.06644), jerk punishes camera reversals,
# vel curbs endless spinning, jump curbs random hopping
W_CAMERA   = 0.0006
W_JERK     = 0.002
W_JUMP     = 0.001

# signed degrees per camera bin so jerk catches direction reversals
CAM = np.array([-10.0,  -3.0,   0.0,   3.0,  10.0], dtype=np.float32)


def _wood_name(name: str) -> bool:
    if name in EXCLUDED:
        return False
    return name.startswith("minecraft:") and name.endswith(SUFFIXES)


def log_item_ids(registry: Registry) -> set[int]:
    # item ids for the inventory, separate id space from blocks
    items = registry.item_ids()
    return {items[n] for n in items if _wood_name(n)}


def log_block_ids(registry: Registry) -> set[int]:
    # block ids for the voxel grid and target_block
    blocks = registry.block_ids()
    return {blocks[n] for n in blocks if _wood_name(n)}


class Wood(Task):
    """Vectorized over all N agents, prev trackers are (N,) arrays"""

    name  = "wood"
    eplen = 256

    def __init__(self, agents: int, registry: Registry) -> None:
        items  = np.array(sorted(log_item_ids(registry)), dtype=np.int64)
        blocks = np.array(sorted(log_block_ids(registry)), dtype=np.int64)
        self._blocks = blocks

        # bool lookup tables, a gather is far cheaper than np.isin over two
        # 4913 cell grids every step, clip handles out of range ids as air
        self._lut = 4096
        self._log_block = np.zeros(self._lut, dtype=bool)
        self._log_block[blocks[blocks < self._lut]] = True
        self._log_item = np.zeros(self._lut, dtype=bool)
        self._log_item[items[items < self._lut]] = True

        # distance from grid center per cell, far shell is stride blocks apart
        r    = (EDGE - 1) // 2
        cell = np.empty(EDGE * EDGE * EDGE, dtype=np.float32)
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    cell[((dy + r) * EDGE + (dz + r)) * EDGE + (dx + r)] = (dx * dx + dy * dy + dz * dz) ** 0.5

        self._cell     = cell
        self._cell_far = cell * float(spec.VOXEL_FAR_STRIDE)

        self._wood  = None
        self._dist  = None
        self._far   = None
        self._total = None
        self._yaw   = np.zeros(agents, dtype=np.float32)
        self._pitch = np.zeros(agents, dtype=np.float32)

    def _count(self, obs):
        ids = np.clip(obs["inv_item_id"], 0, self._lut - 1)
        return (self._log_item[ids] * obs["inv_count"]).sum(axis=1).astype(np.float32)

    def _items(self, obs):
        ct = obs["inv_count"].astype(np.int64)
        return ((obs["inv_item_id"] != 0) * ct).sum(axis=1).astype(np.float32)

    def _near(self, obs):
        mask = self._log_block[np.clip(obs["voxel_blocks"], 0, self._lut - 1)]
        return np.where(mask, self._cell[None, :], NO_LOG).min(axis=1).astype(np.float32)

    def _shell(self, obs):
        mask = self._log_block[np.clip(obs["voxel_far"], 0, self._lut - 1)]
        return np.where(mask, self._cell_far[None, :], NO_LOG_FAR).min(axis=1).astype(np.float32)

    def reset(self, obs: np.ndarray) -> None:
        self._wood     = self._count(obs)
        self._dist     = self._near(obs)
        self._far      = self._shell(obs)
        self._total    = self._items(obs)
        self._yaw[:]   = 0.0
        self._pitch[:] = 0.0

    def reward(self, obs, act, died, respawned) -> np.ndarray:
        wood  = self._count(obs)
        dist  = self._near(obs)
        far   = self._shell(obs)
        total = self._items(obs)

        r = (
            W_WOOD * (wood - self._wood)
            + W_APPROACH * (self._dist - dist)
            + W_FAR * (self._far - far)
            + W_ANYITEM * np.maximum(total - self._total, 0.0)
        )

        looking  = (obs["target_in_range"] == 1) & np.isin(obs["target_block"], self._blocks)
        attacked = act[:, 6] == 1
        r = r + W_FACE * looking + W_ATTACK * (looking & attacked)

        # camera shaping
        yaw   = CAM[act[:, 4]]
        pitch = CAM[act[:, 5]]
        jerk  = np.abs(yaw - self._yaw) + np.abs(pitch - self._pitch)
        vel   = np.abs(yaw) + np.abs(pitch)
        r = r - W_CAMERA * vel - W_JERK * jerk - W_JUMP * act[:, 2]

        # zero the respawn frame, its cross episode delta is spurious, then
        # override the death frame with the penalty
        r = r.astype(np.float32)
        r[respawned] = 0.0
        r[died] = -W_DEATH

        self._wood        = wood
        self._dist        = dist
        self._far         = far
        self._total       = total
        self._yaw         = yaw.astype(np.float32)
        self._pitch       = pitch.astype(np.float32)
        self._yaw[died]   = 0.0
        self._pitch[died] = 0.0

        return r

    def metric(self, obs: np.ndarray) -> np.ndarray:
        return self._count(obs)
