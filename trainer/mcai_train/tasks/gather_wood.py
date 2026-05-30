"""Gather-wood task: count log/wood items in the inventory and reward gains.

Operates on decoded observation records (a single agent row of OBS_DTYPE) whose
``inv_item_id`` carries ITEM ids (BuiltInRegistries.ITEM on the Java side) and
``inv_count`` the per-slot stack sizes.
"""
from __future__ import annotations

import numpy as np

from mcai_train.schema.registry import Registry

# Vanilla "wood" item families the agent gathers by mining: the natural logs and
# their bark (_wood) variants, the nether stems/hyphae, plus the stripped forms
# (stripping a log keeps it counted as gathered wood). mushroom_stem also ends in
# "_stem" but is a fungus block, not wood, so it is excluded by an explicit guard.
_WOOD_SUFFIXES = ("_log", "_wood", "_stem", "_hyphae")
_EXCLUDED = frozenset({"minecraft:mushroom_stem"})


def _is_wood_item_name(name: str) -> bool:
    if name in _EXCLUDED:
        return False
    return name.startswith("minecraft:") and name.endswith(_WOOD_SUFFIXES)


# Concrete set of vanilla wood item ids resolved against a registry at call time
# (see log_item_ids). LOG_ITEM_NAMES is the name-level source of truth, derived
# from the registry so it never drifts from the real id table.
def _log_item_names(registry: Registry) -> set[str]:
    return {name for name in registry._name_to_id if _is_wood_item_name(name)}


def log_item_ids(registry: Registry) -> set[int]:
    """ITEM ids of all wood-family items known to the registry.

    Uses Registry.id_of so the returned ids are ITEM ids (which override block
    ids of the same name in the merged registry) — the ids that appear in an
    observation's inv_item_id.
    """
    return {registry.id_of(name) for name in _log_item_names(registry)}


# Module-level name list for callers that want the human-readable set without a
# registry handle. Built lazily from the shipped registry.json on first import is
# avoided to keep this module import-light; expose the predicate-derived set via
# the registry instead. We still publish the suffix-matched names of the canonical
# overworld logs for documentation/tests.
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


def wood_count(obs_record, log_ids: set[int]) -> int:
    """Total count of wood items across the inventory of one observation record.

    Sums inv_count[i] for every slot whose inv_item_id[i] is a wood item id.
    """
    item_ids = np.asarray(obs_record["inv_item_id"])
    counts = np.asarray(obs_record["inv_count"])
    mask = np.isin(item_ids, list(log_ids))
    return int(counts[mask].sum())


class GatherWoodReward:
    """Per-step reward = change in total wood held since the previous step."""

    def __init__(self, log_ids: set[int]) -> None:
        self._log_ids = log_ids
        self._prev = 0

    def reset(self, obs_record) -> None:
        self._prev = wood_count(obs_record, self._log_ids)

    def compute(self, obs_record) -> float:
        current = wood_count(obs_record, self._log_ids)
        delta = current - self._prev
        self._prev = current
        return float(delta)
