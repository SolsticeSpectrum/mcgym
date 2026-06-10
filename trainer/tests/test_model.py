"""Model + action-space unit tests (fast, CPU; one CUDA forward if available)."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import torch

from mcgym.models.actions import BINS, actions_to_records
from mcgym.models.policy import ActorCritic, tensors
from mcgym.schema import spec

REGISTRY_PATH = pathlib.Path("/home/user/github/mcai/schema/registry.json")


def _sizes():
    doc = json.loads(REGISTRY_PATH.read_text())
    return max(doc["blocks"].values()) + 2, max(doc["items"].values()) + 2


def _fake_obs(n=4):
    obs = np.zeros(n, dtype=spec.OBS_DTYPE)
    for i in range(n):
        obs[i]["voxel_blocks"][:10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        obs[i]["inv_item_id"][:3] = [11, 22, 33]
        obs[i]["inv_count"][:3] = [5, 1, 64]
        obs[i]["health"] = 20.0
        obs[i]["food"] = 18.0
        obs[i]["yaw"] = 45.0 * i
        obs[i]["target_face"] = i % 7  # includes 6 -> treated as invalid
        obs[i]["target_distance"] = 2.5
        obs[i]["target_in_range"] = 1
    return obs


def test_action_mapping():
    idx = np.array([[2, 0, 1, 0, 0, 4, 1]], dtype=np.int64)
    rec = actions_to_records(idx)
    assert rec.dtype == spec.ACTION_DTYPE
    assert rec.shape == (1,)
    assert rec[0]["forward"] == 1.0  # idx 2 -> 1.0
    assert rec[0]["strafe"] == -1.0  # idx 0 -> -1.0
    assert rec[0]["jump"] == 1
    assert rec[0]["sprint"] == 0
    assert rec[0]["yaw_delta"] == -10.0  # idx 0 -> -10
    assert rec[0]["pitch_delta"] == 10.0  # idx 4 -> 10
    assert rec[0]["attack"] == 1
    # Fixed fields stay zero.
    assert rec[0]["sneak"] == 0
    assert rec[0]["use"] == 0
    assert rec[0]["selected_slot"] == 0


def test_forward_shapes():
    num_blocks, num_items = _sizes()
    model = ActorCritic(num_blocks, num_items)
    obs = _fake_obs(4)
    t = tensors(obs, "cpu")

    latent = model.encoder.encode(t)
    assert latent.shape == (4, 256)

    logits, value = model.forward(t)
    assert logits.shape == (4, sum(BINS))
    assert value.shape == (4,)

    act, logprob, val = model.get_action(t)
    assert act.shape == (4, 7)
    assert logprob.shape == (4,)
    assert val.shape == (4,)
    for h, b in enumerate(BINS):
        assert (act[:, h] >= 0).all() and (act[:, h] < b).all()

    lp, ent, v = model.evaluate(t, act)
    assert lp.shape == (4,) and ent.shape == (4,) and v.shape == (4,)
    assert torch.isfinite(lp).all() and torch.isfinite(ent).all() and torch.isfinite(v).all()


def test_deterministic_action():
    num_blocks, num_items = _sizes()
    model = ActorCritic(num_blocks, num_items)
    t = tensors(_fake_obs(4), "cpu")
    a1, _, _ = model.get_action(t, deterministic=True)
    a2, _, _ = model.get_action(t, deterministic=True)
    assert torch.equal(a1, a2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_forward():
    num_blocks, num_items = _sizes()
    model = ActorCritic(num_blocks, num_items).to("cuda")
    t = tensors(_fake_obs(4), "cuda")
    act, logprob, value = model.get_action(t)
    assert act.shape == (4, 7)
    assert act.device.type == "cuda"
    assert torch.isfinite(logprob).all() and torch.isfinite(value).all()
