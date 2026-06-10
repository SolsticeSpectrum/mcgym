"""wood task unit tests, synthetic observations, no gym."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from mcgym.schema import spec
from mcgym.schema.registry import Registry
from mcgym.tasks.wood import (
    Wood, log_block_ids, log_item_ids,
    W_WOOD, W_APPROACH, W_FACE, W_ATTACK, W_ANYITEM, W_DEATH, W_JERK,
)

REGISTRY = pathlib.Path(__file__).resolve().parents[2] / "schema" / "registry.json"

R = spec.PARAMS["voxel_radius"]
E = spec.VOXEL_EDGE

NOOP = np.zeros((1, 7), dtype=np.int64)
NOOP[:, 4] = 2  # camera bins, index 2 is 0 degrees
NOOP[:, 5] = 2

ALIVE = np.zeros(1, dtype=bool)


def cell(dx, dy, dz):
    return ((dy + R) * E + (dz + R)) * E + (dx + R)


def obs():
    o = np.zeros(1, dtype=spec.OBS_DTYPE)
    o["health"] = 20.0
    return o


@pytest.fixture(scope="module")
def registry():
    return Registry.load(REGISTRY)


@pytest.fixture()
def task(registry):
    return Wood(1, registry)


def test_id_spaces_differ(registry):
    # oak_log resolves to different ints in the block vs item table
    block = registry.block_id_of("minecraft:oak_log")
    item = registry.item_id_of("minecraft:oak_log")
    assert block != item
    assert block in log_block_ids(registry)
    assert item in log_item_ids(registry)
    assert block not in log_item_ids(registry)
    assert item not in log_block_ids(registry)


def test_overworld_logs_covered(registry):
    items = registry.item_ids()
    ids = log_item_ids(registry)
    for name in ("minecraft:oak_log", "minecraft:spruce_log", "minecraft:birch_log",
                 "minecraft:jungle_log", "minecraft:acacia_log", "minecraft:dark_oak_log",
                 "minecraft:mangrove_log", "minecraft:cherry_log"):
        assert items[name] in ids
    assert items["minecraft:mushroom_stem"] not in ids


def test_approach_positive_when_closer(registry, task):
    oak = registry.block_id_of("minecraft:oak_log")
    far = obs()
    far[0]["voxel_blocks"][cell(5, 0, 0)] = oak
    task.reset(far)

    near = obs()
    near[0]["voxel_blocks"][cell(2, 0, 0)] = oak
    r = task.reward(near, NOOP, ALIVE, ALIVE)
    assert r[0] == pytest.approx(W_APPROACH * 3.0, abs=1e-5)


def test_face_and_attack_bonus(registry, task):
    oak = registry.block_id_of("minecraft:oak_log")
    o = obs()
    o[0]["voxel_blocks"][cell(1, 0, 0)] = oak
    o[0]["target_block"] = oak
    o[0]["target_in_range"] = 1
    task.reset(o)

    r = task.reward(o, NOOP, ALIVE, ALIVE)
    assert r[0] == pytest.approx(W_FACE, abs=1e-5)

    attack = NOOP.copy()
    attack[:, 6] = 1
    r = task.reward(o, attack, ALIVE, ALIVE)
    assert r[0] == pytest.approx(W_FACE + W_ATTACK, abs=1e-5)


def test_wood_gain_dominates(registry, task):
    oak = registry.item_id_of("minecraft:oak_log")
    o = obs()
    task.reset(o)

    got = obs()
    got[0]["inv_item_id"][0] = oak
    got[0]["inv_count"][0] = 2
    r = task.reward(got, NOOP, ALIVE, ALIVE)
    assert r[0] == pytest.approx(W_WOOD * 2 + W_ANYITEM * 2, abs=1e-4)


def test_non_log_item_gives_anyitem_only(registry, task):
    dirt = registry.item_id_of("minecraft:dirt")
    o = obs()
    task.reset(o)

    got = obs()
    got[0]["inv_item_id"][0] = dirt
    got[0]["inv_count"][0] = 3
    r = task.reward(got, NOOP, ALIVE, ALIVE)
    assert r[0] == pytest.approx(W_ANYITEM * 3, abs=1e-4)


def test_death_and_respawn(registry, task):
    o = obs()
    task.reset(o)

    dead = obs()
    dead[0]["health"] = 0.0
    died = np.ones(1, dtype=bool)
    r = task.reward(dead, NOOP, died, ALIVE)
    assert r[0] == -W_DEATH

    # respawn frame is zeroed even though trackers see a cross episode jump
    back = obs()
    r = task.reward(back, NOOP, ALIVE, died)
    assert r[0] == 0.0


def test_camera_jerk_penalized(registry, task):
    o = obs()
    task.reset(o)

    swing = NOOP.copy()
    swing[:, 4] = 0  # -10 degrees after 0, jerk of 10
    r_swing = task.reward(o, swing, ALIVE, ALIVE)

    task.reset(o)
    r_still = task.reward(o, NOOP, ALIVE, ALIVE)
    assert r_swing[0] < r_still[0]
    assert r_still[0] - r_swing[0] >= W_JERK * 10 * 0.9


def test_never_nan(registry, task):
    o = obs()
    task.reset(o)
    for _ in range(5):
        r = task.reward(o, NOOP, ALIVE, ALIVE)
        assert np.isfinite(r).all()
