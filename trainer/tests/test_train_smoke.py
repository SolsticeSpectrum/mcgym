"""Slow end-to-end smoke test: real gym, a couple of PPO rollouts."""
from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest
import torch

from mcai_train import checkpoint
from mcai_train.env.wood_env import WoodEnv
from mcai_train.models.policy import ActorCritic, obs_to_tensors
from mcai_train.ppo.buffer import RolloutBuffer
from mcai_train.ppo.learner import PPOLearner
from mcai_train.schema.registry import Registry
from mcai_train.train import _model_sizes

REGISTRY_PATH = pathlib.Path("/home/user/github/mcai/schema/registry.json")


@pytest.mark.slow
def test_train_smoke():
    n_agents = 2
    rollout_len = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    registry = Registry.load(REGISTRY_PATH)
    num_blocks, num_items = _model_sizes(REGISTRY_PATH)
    model = ActorCritic(num_blocks, num_items).to(device)
    learner = PPOLearner(model, epochs=2, minibatch=16, device=device)

    env = None
    cumulative_timesteps = 0
    with tempfile.TemporaryDirectory() as ckpt_dir:
        try:
            env = WoodEnv(n_agents, seed=0, registry=registry, episode_len=50)
            obs_struct = env.reset()

            for _ in range(2):  # two rollouts
                buf = RolloutBuffer(rollout_len, n_agents)
                for _ in range(rollout_len):
                    with torch.no_grad():
                        idx, lp, val = model.get_action(obs_to_tensors(obs_struct, device))
                    a = idx.cpu().numpy()
                    nxt, rew, done = env.step(a)
                    assert np.isfinite(rew).all(), "non-finite reward"
                    buf.add(obs_struct, a, lp.cpu().numpy(), rew, val.cpu().numpy(), done)
                    obs_struct = nxt
                with torch.no_grad():
                    last_v = model.get_action(obs_to_tensors(obs_struct, device))[2]
                buf.compute_gae(last_v.cpu().numpy())
                metrics = learner.update(buf)
                assert np.isfinite(metrics["policy_loss"])
                assert np.isfinite(metrics["value_loss"])
                cumulative_timesteps += rollout_len * n_agents

            checkpoint.save(
                ckpt_dir, model, learner.optimizer,
                {"cumulative_timesteps": cumulative_timesteps, "schema_version": 0},
            )
            assert (pathlib.Path(ckpt_dir) / "latest" / "model.pt").exists()
            assert cumulative_timesteps == 2 * rollout_len * n_agents
        finally:
            if env is not None:
                env.close()
