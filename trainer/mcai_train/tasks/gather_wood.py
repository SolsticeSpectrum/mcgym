"""Gather-wood task: dense shaping reward over decoded observation records.

A single agent row of ``OBS_DTYPE`` carries:
  * ``inv_item_id`` / ``inv_count`` — ITEM ids (BuiltInRegistries.ITEM) and stack
    sizes. Use ITEM ids to count wood actually collected.
  * ``voxel_blocks`` — a flat (VOXEL_EDGE**3,) array of BLOCK ids
    (BuiltInRegistries.BLOCK). Use BLOCK ids to locate logs in the world.
  * ``target_block`` — the BLOCK id the agent is looking at; ``target_in_range``
    is 1 when it is within reach.

Block ids and item ids are SEPARATE id spaces, so the same name resolves to a
different integer in each (oak_log: block 49 vs item 134). We resolve log names
against the correct table for each use (block grid vs inventory).
"""
from __future__ import annotations

import numpy as np

from mcai_train.schema import spec
from mcai_train.schema.registry import Registry

# Vanilla "wood" families the agent gathers by mining: natural logs, bark (_wood)
# variants, nether stems/hyphae, and stripped forms. mushroom_stem also ends in
# "_stem" but is a fungus, not wood, so it is excluded by an explicit guard.
_WOOD_SUFFIXES = ("_log", "_wood", "_stem", "_hyphae")
_EXCLUDED = frozenset({"minecraft:mushroom_stem"})

# Voxel grid geometry (mirrors spec): center index of the flat voxel array and
# the per-axis edge length. index = ((dy+R)*EDGE + (dz+R))*EDGE + (dx+R).
_EDGE = spec.VOXEL_EDGE
_RADIUS = spec.PARAMS["voxel_radius"]
_CENTER = (_EDGE**3) // 2
# Distance returned when no log is present anywhere in the grid (> max possible
# Euclidean offset within the radius), so potential shaping stays finite/bounded.
NO_LOG_DIST = 16.0

# --- reward component weights (tune here) ----------------------------------
W_WOOD = 10.0      # Δ(wood item count) — the real objective.
W_APPROACH = 0.3   # potential shaping: getting closer to the nearest log block.
W_FACE = 0.05      # per-step bonus for looking at an in-range log block.
W_ATTACK_LOG = 0.15  # per-step bonus for ATTACKING an in-range log; rewards the
                     # sustained mining a random policy never discovers on its own.
W_DEATH = 10.0       # one-time penalty applied when the agent dies (health<=0);
                     # death is a real, learnable negative signal, not engineered away.
W_CAMERA = 0.003     # per-step penalty per degree of |yaw_delta|+|pitch_delta|, to
                     # discourage chaotic spinning so the agent holds aim to finish a mine.
W_ANYITEM = 0.2    # Δ(total inventory count) — bootstraps "pick something up".


def _is_wood_name(name: str) -> bool:
    if name in _EXCLUDED:
        return False
    return name.startswith("minecraft:") and name.endswith(_WOOD_SUFFIXES)


def _wood_names_from_map(name_to_id: dict[str, int]) -> set[str]:
    return {name for name in name_to_id if _is_wood_name(name)}


LOG_ITEM_NAMES = (
    "minecraft:oak_log",
    "minecraft:spruce_log",
    "minecraft:birch_log",
    "minecraft:jungle_log",
    "minecraft:acacia_log",
    "minecraft:dark_oak_log",
    "minecraft:mangrove_log",
    "minecraft:cherry_log",
    "minecraft:pale_oak_log",
)


def log_item_ids(registry: Registry) -> set[int]:
    """ITEM ids of all wood-family items (for the inventory: inv_item_id)."""
    items = registry.item_ids()
    return {items[name] for name in _wood_names_from_map(items)}


def log_block_ids(registry: Registry) -> set[int]:
    """BLOCK ids of all wood-family blocks (for the voxel grid + target_block)."""
    blocks = registry.block_ids()
    return {blocks[name] for name in _wood_names_from_map(blocks)}


def wood_count(obs_record, log_item_ids_: set[int]) -> int:
    """Total count of wood ITEMS across the inventory of one observation record."""
    item_ids = np.asarray(obs_record["inv_item_id"])
    counts = np.asarray(obs_record["inv_count"])
    mask = np.isin(item_ids, list(log_item_ids_))
    return int(counts[mask].sum())


def total_item_count(obs_record) -> int:
    """Total stack size across all occupied inventory slots."""
    item_ids = np.asarray(obs_record["inv_item_id"])
    counts = np.asarray(obs_record["inv_count"]).astype(np.int64)
    return int(counts[item_ids != 0].sum())


def nearest_log_dist(obs_record, log_block_ids_: set[int]) -> float:
    """Min Euclidean distance (in block offsets) to a log BLOCK in the voxel grid.

    Returns NO_LOG_DIST if no log block is present. The agent sits at the grid
    center, so distance is the magnitude of the (dx, dy, dz) offset of the
    nearest matching cell.
    """
    voxel = np.asarray(obs_record["voxel_blocks"])
    matches = np.nonzero(np.isin(voxel, list(log_block_ids_)))[0]
    if matches.size == 0:
        return NO_LOG_DIST
    # Invert index = ((dy+R)*EDGE + (dz+R))*EDGE + (dx+R) -> (dx,dy,dz).
    rem, dx = np.divmod(matches, _EDGE)
    dy, dz = np.divmod(rem, _EDGE)
    off = np.stack(
        [dx - _RADIUS, dy - _RADIUS, dz - _RADIUS], axis=1
    ).astype(np.float64)
    dist = np.sqrt((off**2).sum(axis=1))
    return float(dist.min())


class WoodShapedReward:
    """Dense per-step, single-agent reward for the gather-wood task.

    Components (see module weights):
      +W_WOOD    * Δ(wood item count)            — real objective.
      +W_APPROACH* (prev_dist - cur_dist)         — potential shaping toward the
                                                    nearest log block.
      +W_FACE    if looking at an in-range log block.
      +W_ANYITEM * max(Δ total inventory count, 0) — bootstraps pickups. This
                  overlaps slightly with the wood term (wood pickups also raise
                  the total count); the overlap is intentional and tiny.
    """

    def __init__(self, log_item_ids_: set[int], log_block_ids_: set[int]) -> None:
        self._log_item_ids = log_item_ids_
        self._log_block_ids = log_block_ids_
        self._prev_wood = 0
        self._prev_dist = NO_LOG_DIST
        self._prev_total = 0

    def reset(self, obs_record) -> None:
        self._prev_wood = wood_count(obs_record, self._log_item_ids)
        self._prev_dist = nearest_log_dist(obs_record, self._log_block_ids)
        self._prev_total = total_item_count(obs_record)

    def _target_is_log(self, obs_record) -> bool:
        return int(obs_record["target_block"]) in self._log_block_ids

    def compute(self, obs_record, attacked: bool = False) -> float:
        wood = wood_count(obs_record, self._log_item_ids)
        dist = nearest_log_dist(obs_record, self._log_block_ids)
        total = total_item_count(obs_record)

        d_wood = wood - self._prev_wood
        d_total = max(total - self._prev_total, 0)
        approach = self._prev_dist - dist

        reward = W_WOOD * d_wood
        reward += W_APPROACH * approach
        reward += W_ANYITEM * d_total
        looking_at_log = (
            int(obs_record["target_in_range"]) == 1 and self._target_is_log(obs_record)
        )
        if looking_at_log:
            reward += W_FACE
            if attacked:
                reward += W_ATTACK_LOG

        self._prev_wood = wood
        self._prev_dist = dist
        self._prev_total = total
        return float(reward)


class GatherWoodReward:
    """Sparse Δ(wood) reward. Retained for tests/comparison."""

    def __init__(self, log_item_ids_: set[int]) -> None:
        self._log_ids = log_item_ids_
        self._prev = 0

    def reset(self, obs_record) -> None:
        self._prev = wood_count(obs_record, self._log_ids)

    def compute(self, obs_record) -> float:
        current = wood_count(obs_record, self._log_ids)
        delta = current - self._prev
        self._prev = current
        return float(delta)


class BatchWoodReward:
    """Vectorised gather-wood reward over all N agents at once.

    Same shaping as WoodShapedReward but computed with batched numpy on the full
    (N, ...) obs arrays, replacing the per-agent Python loop (the measured #2 cost
    in env.step). prev_* are (N,) arrays; compute() returns an (N,) reward.
    """

    def __init__(self, n_agents: int, log_item_ids_: set[int], log_block_ids_: set[int]) -> None:
        self.n = n_agents
        self._item_arr = np.array(sorted(log_item_ids_), dtype=np.int64)
        self._block_arr = np.array(sorted(log_block_ids_), dtype=np.int64)
        edge = _EDGE
        r = (edge - 1) // 2
        cell = np.empty(edge * edge * edge, dtype=np.float32)
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    cell[((dy + r) * edge + (dz + r)) * edge + (dx + r)] = (dx * dx + dy * dy + dz * dz) ** 0.5
        self._cell_dist = cell
        self._prev_wood = None
        self._prev_dist = None
        self._prev_total = None

    def _wood(self, obs):
        return (np.isin(obs["inv_item_id"], self._item_arr) * obs["inv_count"]).sum(axis=1).astype(np.float32)

    def _total(self, obs):
        ct = obs["inv_count"].astype(np.int64)
        return ((obs["inv_item_id"] != 0) * ct).sum(axis=1).astype(np.float32)

    def _nearest(self, obs):
        mask = np.isin(obs["voxel_blocks"], self._block_arr)
        return np.where(mask, self._cell_dist[None, :], NO_LOG_DIST).min(axis=1).astype(np.float32)

    def reset(self, obs) -> None:
        self._prev_wood = self._wood(obs)
        self._prev_dist = self._nearest(obs)
        self._prev_total = self._total(obs)

    def compute(self, obs, attacked) -> np.ndarray:
        wood = self._wood(obs)
        dist = self._nearest(obs)
        total = self._total(obs)
        r = (
            W_WOOD * (wood - self._prev_wood)
            + W_APPROACH * (self._prev_dist - dist)
            + W_ANYITEM * np.maximum(total - self._prev_total, 0.0)
        )
        looking = (obs["target_in_range"] == 1) & np.isin(obs["target_block"], self._block_arr)
        r = r + W_FACE * looking + W_ATTACK_LOG * (looking & np.asarray(attacked))
        self._prev_wood = wood
        self._prev_dist = dist
        self._prev_total = total
        return r.astype(np.float32)
