"""parkour task unit tests, synthetic observations, no gym"""
from __future__ import annotations

import numpy as np

from mcgym.tasks.parkour import (
    Parkour, GOAL, DONE_R,
    W_GOAL, W_FALL, W_LOW, W_DONE, W_DEATH,
)

NOOP = np.zeros((1, 7), dtype=np.int64)
NOOP[:, 4] = 2  # camera bins, index 2 is 0 degrees
NOOP[:, 5] = 2

ALIVE = np.zeros(1, dtype=bool)


def obs(x, y, z):
    o = np.zeros(1, dtype=np.dtype([("pos", "f4", 3), ("health", "f4")]))
    o["pos"]    = [x, y, z]
    o["health"] = 20.0
    return o


def task():
    t = Parkour(1, None)
    t.reset(obs(38.5, -61.0, -63.5))
    return t


def test_climbing_pays():
    t = task()
    r = t.reward(obs(38.5, -59.0, -63.5), NOOP, ALIVE, ALIVE)
    assert r[0] > W_GOAL * 1.5  # closed ~2 blocks of gap


def test_low_pressure_below_goal_height():
    t = task()
    r = t.reward(obs(38.5, -61.0, -63.5), NOOP, ALIVE, ALIVE)
    assert np.isclose(r[0], -W_LOW)  # standing still under 104 still bleeds


def test_falling_is_punished_once():
    t = task()
    t.reward(obs(38.5, -50.0, -63.5), NOOP, ALIVE, ALIVE)  # climbed to -50
    fell = t.reward(obs(38.5, -55.0, -63.5), NOOP, ALIVE, ALIVE)
    # 5 blocks of fresh drop on top of the reopened gap
    assert fell[0] < -(W_FALL * 5.0)
    # staying down is not the fall penalty again, only the low pressure
    stay = t.reward(obs(38.5, -55.0, -63.5), NOOP, ALIVE, ALIVE)
    assert np.isclose(stay[0], -W_LOW)


def test_goal_pays_and_ends():
    t = task()
    t.reward(obs(38.0, 103.0, -64.0), NOOP, ALIVE, ALIVE)
    r = t.reward(obs(38.0, 104.0, -64.0), NOOP, ALIVE, ALIVE)
    assert r[0] > W_DONE
    assert t.done(ALIVE)[0]


def test_death_overrides():
    t = task()
    died = np.ones(1, dtype=bool)
    r = t.reward(obs(38.5, -61.0, -63.5), NOOP, died, ALIVE)
    assert np.isclose(r[0], -W_DEATH)


def test_respawn_frame_is_zero():
    t = task()
    t.reward(obs(38.5, -40.0, -63.5), NOOP, ALIVE, ALIVE)
    respawned = np.ones(1, dtype=bool)
    r = t.reward(obs(38.5, -61.0, -63.5), NOOP, ALIVE, respawned)
    assert np.isclose(r[0], 0.0)
    # trackers rebased at the pool, no phantom fall on the next step
    after = t.reward(obs(38.5, -61.0, -63.5), NOOP, ALIVE, ALIVE)
    assert np.isclose(after[0], -W_LOW)


def test_metric_is_best_height():
    t = task()
    t.reward(obs(38.5, -45.0, -63.5), NOOP, ALIVE, ALIVE)
    t.reward(obs(38.5, -55.0, -63.5), NOOP, ALIVE, ALIVE)
    assert np.isclose(t.metric(obs(38.5, -55.0, -63.5))[0], -45.0)
