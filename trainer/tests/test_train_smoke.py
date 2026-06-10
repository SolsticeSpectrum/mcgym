"""slow end to end smoke test, real gym, a couple of ppo rollouts"""
from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest
import torch

from mcgym import checkpoint
from mcgym.env import Env
from mcgym.tasks.wood import Wood
from mcgym.models.policy import ActorCritic, tensors
from mcgym.ppo.buffer import Buffer
from mcgym.ppo.learner import Learner
from mcgym.schema.registry import Registry
from mcgym.train import sizes

REGISTRY = pathlib.Path(__file__).resolve().parents[2] / "schema" / "registry.json"


@pytest.mark.slow
def test_train_smoke():
    agents  = 2
    rollout = 16
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    registry = Registry.load(REGISTRY)
    num_blocks, num_items = sizes(REGISTRY)
    model    = ActorCritic(num_blocks, num_items).to(device)
    learner  = Learner(model, epochs=2, minibatch=16, device=device)

    env = None
    steps = 0
    with tempfile.TemporaryDirectory() as ckpt_dir:
        try:
            task = Wood(agents, registry)
            task.eplen = 50
            env  = Env(agents, 0, task)
            obs  = env.reset()

            for _ in range(2):
                buf = Buffer(rollout, agents)
                for _ in range(rollout):
                    with torch.no_grad():
                        idx, lp, val = model.get_action(tensors(obs, device))
                    a = idx.cpu().numpy()
                    nxt, rew, done = env.step(a)
                    assert np.isfinite(rew).all(), "non-finite reward"
                    
                    buf.add(obs, a, lp.cpu().numpy(), rew, val.cpu().numpy(), done)
                    obs = nxt
                    
                with torch.no_grad():
                    last_v = model.get_action(tensors(obs, device))[2]
                    
                buf.gae(last_v.cpu().numpy())
                metrics = learner.update(buf)
                
                assert np.isfinite(metrics["policy_loss"])
                assert np.isfinite(metrics["value_loss"])
                
                steps += rollout * agents

            checkpoint.save(
                ckpt_dir, model, learner.optimizer,
                {"steps": steps, "schema_version": 0},
            )
            
            assert (pathlib.Path(ckpt_dir) / "latest" / "model.pt").exists()
            assert steps == 2 * rollout * agents
        finally:
            if env is not None:
                env.close()
