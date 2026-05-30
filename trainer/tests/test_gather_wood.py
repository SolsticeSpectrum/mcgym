"""Fast unit tests for the gather-wood task (no gym, synthetic observations)."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from mcai_train.schema import spec
from mcai_train.schema.registry import Registry
from mcai_train.tasks import GatherWoodReward, log_item_ids, wood_count

REGISTRY_PATH = pathlib.Path("/home/user/github/mcai/schema/registry.json")


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load(REGISTRY_PATH)


@pytest.fixture(scope="module")
def log_ids(registry: Registry) -> set[int]:
    return log_item_ids(registry)


def _empty_obs() -> np.ndarray:
    return np.zeros(1, dtype=spec.OBS_DTYPE)


def test_log_item_ids_include_overworld_logs(registry, log_ids):
    for name in (
        "minecraft:oak_log",
        "minecraft:spruce_log",
        "minecraft:birch_log",
        "minecraft:jungle_log",
        "minecraft:acacia_log",
        "minecraft:dark_oak_log",
        "minecraft:mangrove_log",
        "minecraft:cherry_log",
    ):
        assert registry.id_of(name) in log_ids
    # mushroom_stem is a fungus, must not be counted as wood.
    assert registry.id_of("minecraft:mushroom_stem") not in log_ids
    # air must never be wood.
    assert registry.id_of("minecraft:air") not in log_ids


def test_wood_count_zero_when_empty(log_ids):
    obs = _empty_obs()[0]
    assert wood_count(obs, log_ids) == 0


def test_wood_count_sums_matching_slots(registry, log_ids):
    oak = registry.id_of("minecraft:oak_log")
    spruce = registry.id_of("minecraft:spruce_log")
    obs = _empty_obs()
    obs[0]["inv_item_id"][0] = oak
    obs[0]["inv_count"][0] = 3
    obs[0]["inv_item_id"][5] = spruce
    obs[0]["inv_count"][5] = 2
    # A non-wood item must be ignored.
    obs[0]["inv_item_id"][7] = registry.id_of("minecraft:dirt")
    obs[0]["inv_count"][7] = 9
    assert wood_count(obs[0], log_ids) == 5


def test_reward_returns_delta(registry, log_ids):
    oak = registry.id_of("minecraft:oak_log")
    reward = GatherWoodReward(log_ids)

    start = _empty_obs()[0]
    reward.reset(start)
    # No change yet.
    assert reward.compute(start) == 0.0

    gained = _empty_obs()
    gained[0]["inv_item_id"][0] = oak
    gained[0]["inv_count"][0] = 4
    # +4 wood since previous step.
    assert reward.compute(gained[0]) == 4.0
    # Holding steady -> 0 reward.
    assert reward.compute(gained[0]) == 0.0

    more = _empty_obs()
    more[0]["inv_item_id"][0] = oak
    more[0]["inv_count"][0] = 6
    assert reward.compute(more[0]) == 2.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
