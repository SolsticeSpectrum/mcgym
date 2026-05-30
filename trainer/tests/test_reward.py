"""Fast unit tests for the dense gather-wood shaping reward (no gym)."""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from mcai_train.schema import spec
from mcai_train.schema.registry import Registry
from mcai_train.tasks import (
    WoodShapedReward,
    log_block_ids,
    log_item_ids,
    nearest_log_dist,
)
from mcai_train.tasks.gather_wood import (
    NO_LOG_DIST,
    W_ANYITEM,
    W_APPROACH,
    W_FACE,
    W_WOOD,
)

REGISTRY_PATH = pathlib.Path("/home/user/github/mcai/schema/registry.json")

_R = spec.PARAMS["voxel_radius"]
_E = spec.VOXEL_EDGE


def _voxel_index(dx: int, dy: int, dz: int) -> int:
    return ((dy + _R) * _E + (dz + _R)) * _E + (dx + _R)


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load(REGISTRY_PATH)


@pytest.fixture(scope="module")
def item_ids(registry: Registry) -> set[int]:
    return log_item_ids(registry)


@pytest.fixture(scope="module")
def block_ids(registry: Registry) -> set[int]:
    return log_block_ids(registry)


def _empty_obs() -> np.ndarray:
    return np.zeros(1, dtype=spec.OBS_DTYPE)


def test_block_and_item_id_spaces_differ(registry):
    """oak_log must resolve to a different integer in the block vs item table."""
    block = registry.block_id_of("minecraft:oak_log")
    item = registry.item_id_of("minecraft:oak_log")
    assert block != item, (block, item)
    assert block in log_block_ids(registry)
    assert item in log_item_ids(registry)
    # The block id must NOT be treated as an item id and vice versa.
    assert block not in log_item_ids(registry)
    assert item not in log_block_ids(registry)


def test_nearest_log_dist_none(block_ids):
    obs = _empty_obs()[0]
    assert nearest_log_dist(obs, block_ids) == NO_LOG_DIST


def test_nearest_log_dist_euclidean(registry, block_ids):
    oak_block = registry.block_id_of("minecraft:oak_log")
    obs = _empty_obs()
    # Place a log at offset (3, 0, 0) -> distance 3.
    obs[0]["voxel_blocks"][_voxel_index(3, 0, 0)] = oak_block
    assert nearest_log_dist(obs[0], block_ids) == pytest.approx(3.0)
    # And one at (1, 2, 2) -> sqrt(9) = 3 too; add a closer one at (1,1,1).
    obs[0]["voxel_blocks"][_voxel_index(1, 1, 1)] = oak_block
    assert nearest_log_dist(obs[0], block_ids) == pytest.approx(math.sqrt(3))


def test_approach_reward_positive_when_closer(registry, item_ids, block_ids):
    oak_block = registry.block_id_of("minecraft:oak_log")
    reward = WoodShapedReward(item_ids, block_ids)

    far = _empty_obs()
    far[0]["voxel_blocks"][_voxel_index(5, 0, 0)] = oak_block
    reward.reset(far[0])

    near = _empty_obs()
    near[0]["voxel_blocks"][_voxel_index(2, 0, 0)] = oak_block
    r = reward.compute(near[0])
    # Only approach contributes: prev_dist 5 - cur_dist 2 = 3.
    assert r == pytest.approx(W_APPROACH * 3.0)
    assert r > 0


def test_face_bonus(registry, item_ids, block_ids):
    oak_block = registry.block_id_of("minecraft:oak_log")
    reward = WoodShapedReward(item_ids, block_ids)
    start = _empty_obs()[0]
    reward.reset(start)

    looking = _empty_obs()
    looking[0]["target_in_range"] = 1
    looking[0]["target_block"] = oak_block
    r = reward.compute(looking[0])
    assert r == pytest.approx(W_FACE)

    # In range but target is not a log -> no face bonus.
    reward.reset(start)
    dirt_block = registry.block_id_of("minecraft:dirt")
    other = _empty_obs()
    other[0]["target_in_range"] = 1
    other[0]["target_block"] = dirt_block
    assert reward.compute(other[0]) == pytest.approx(0.0)


def test_wood_item_gain_big_reward(registry, item_ids, block_ids):
    oak_item = registry.item_id_of("minecraft:oak_log")
    reward = WoodShapedReward(item_ids, block_ids)
    start = _empty_obs()[0]
    reward.reset(start)

    gained = _empty_obs()
    gained[0]["inv_item_id"][0] = oak_item
    gained[0]["inv_count"][0] = 2
    r = reward.compute(gained[0])
    # 2 wood items: W_WOOD*2 plus any-item bonus W_ANYITEM*2 (intentional overlap).
    assert r == pytest.approx(W_WOOD * 2 + W_ANYITEM * 2)
    assert r >= W_WOOD * 2


def test_non_log_item_gives_anyitem_not_wood(registry, item_ids, block_ids):
    dirt_item = registry.item_id_of("minecraft:dirt")
    reward = WoodShapedReward(item_ids, block_ids)
    start = _empty_obs()[0]
    reward.reset(start)

    gained = _empty_obs()
    gained[0]["inv_item_id"][0] = dirt_item
    gained[0]["inv_count"][0] = 3
    r = reward.compute(gained[0])
    # No wood term, only the any-item bootstrap bonus.
    assert r == pytest.approx(W_ANYITEM * 3)
    assert r < W_WOOD  # nowhere near a real wood pickup


def test_reward_never_nan(item_ids, block_ids):
    reward = WoodShapedReward(item_ids, block_ids)
    obs = _empty_obs()[0]
    reward.reset(obs)
    for _ in range(5):
        assert not math.isnan(reward.compute(obs))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
