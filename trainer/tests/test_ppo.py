"""ppo unit tests, hand checkable gae and a synthetic bandit that must learn."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from mcgym.models.actions import BINS
from mcgym.ppo.buffer import Buffer
from mcgym.ppo.learner import Learner


def test_compute_gae_manual():
    # gamma 1 lam 1 makes gae plain discounted return minus value, easy by hand
    buf = Buffer(rollout=3, agents=1)
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 0.5, 0.5]
    for t in range(3):
        buf.reward[t, 0] = rewards[t]
        buf.value[t, 0] = values[t]
        buf.done[t, 0] = 0.0
    buf._t = 3

    last_value = 0.0
    adv, ret = buf.gae(np.array([last_value]), gamma=1.0, lam=1.0)

    # delta_t = r_t + V(t+1) - V(t), gae_t = delta_t + gae_{t+1}, V(3) = 0
    d2 = 3.0 + 0.0 - 0.5
    d1 = 2.0 + 0.5 - 0.5
    d0 = 1.0 + 0.5 - 0.5
    a2 = d2
    a1 = d1 + a2
    a0 = d0 + a1
    assert np.allclose(adv, [a0, a1, a2])
    assert np.allclose(ret, np.array([a0, a1, a2]) + np.array(values))


def test_compute_gae_with_terminal():
    # terminal at t=1 must cut the bootstrap from t=2
    buf = Buffer(rollout=3, agents=1)
    buf.reward[:, 0] = [1.0, 1.0, 1.0]
    buf.value[:, 0] = [0.0, 0.0, 0.0]
    buf.done[:, 0] = [0.0, 1.0, 0.0]
    buf._t = 3
    adv, _ = buf.gae(np.array([0.0]), gamma=0.99, lam=0.95)

    # t=2 delta = 1, gae2 = 1
    # t=1 done so nonterminal 0, delta = 1, gae1 = 1
    # t=0 gae0 = 1 + 0.99*0.95*gae1
    assert np.allclose(adv[2], 1.0)
    assert np.allclose(adv[1], 1.0)
    assert np.allclose(adv[0], 1.0 + 0.99 * 0.95 * 1.0)


class _Policy(nn.Module):
    """minimal flat obs actor critic with the real multi discrete interface."""

    def __init__(self, obs_dim=1):
        super().__init__()
        self.bins = list(BINS)
        self.body = nn.Sequential(nn.Linear(obs_dim, 32), nn.Tanh())
        self.policy_head = nn.Linear(32, sum(BINS))
        self.value_head = nn.Linear(32, 1)

    def _forward(self, obs):
        h = self.body(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def _dists(self, logits):
        return [Categorical(logits=c) for c in torch.split(logits, self.bins, dim=1)]

    def get_action(self, obs, deterministic=False):
        logits, value = self._forward(obs)
        dists = self._dists(logits)
        acts = [d.sample() for d in dists]
        idx = torch.stack(acts, dim=1)
        lp = sum(d.log_prob(a) for d, a in zip(dists, acts))
        return idx, lp, value

    def evaluate(self, obs, act):
        logits, value = self._forward(obs)
        dists = self._dists(logits)
        cols = act.unbind(dim=1)
        lp = sum(d.log_prob(a) for d, a in zip(dists, cols))
        ent = sum(d.entropy() for d in dists)
        return lp, ent, value


class _FlatBuf(Buffer):
    """buffer storing flat float obs instead of OBS_DTYPE records, no minecraft schema needed."""

    def __init__(self, rollout, agents, obs_dim):
        super().__init__(rollout, agents)
        self.flat = np.zeros((rollout, agents, obs_dim), dtype=np.float32)
        self.obs_dim = obs_dim

    def add(self, obs_flat, act, logprob, reward, value, done):
        self.flat[self._t] = obs_flat
        self.act[self._t] = act
        self.logprob[self._t] = logprob
        self.reward[self._t] = reward
        self.value[self._t] = value
        self.done[self._t] = np.asarray(done, dtype=np.float32)
        self._t += 1

    def to_device(self, device):
        return None

    def batches(self, batch_size, device, data=None):
        flat = self.flat.reshape(-1, self.obs_dim)
        actions = self.act.reshape(-1, self.n_heads)
        logp = self.logprob.reshape(-1)
        total = self.T * self.N
        order = np.random.permutation(total)
        for s in range(0, total, batch_size):
            mb = order[s : s + batch_size]
            yield (
                torch.from_numpy(flat[mb]).to(device),
                torch.from_numpy(actions[mb]).to(device),
                torch.from_numpy(logp[mb]).to(device),
                torch.from_numpy(self.advantages[mb]).to(device),
                torch.from_numpy(self.returns[mb]).to(device),
            )


def test_bandit_learns():
    torch.manual_seed(0)
    np.random.seed(0)

    agents = 16
    rollout = 16
    obs_dim = 1
    model = _Policy(obs_dim)
    learner = Learner(
        model, lr=3e-3, ent_coef=0.0, epochs=4, minibatch=64, device="cpu"
    )

    obs_const = np.ones((agents, obs_dim), dtype=np.float32)

    def mean_reward():
        with torch.no_grad():
            idx, _, _ = model.get_action(torch.from_numpy(obs_const))
        return float((idx[:, 0].numpy() == 2).mean())

    start_rew = np.mean([mean_reward() for _ in range(20)])

    for _ in range(60):
        buf = _FlatBuf(rollout, agents, obs_dim)
        for _ in range(rollout):
            obs_t = torch.from_numpy(obs_const)
            with torch.no_grad():
                idx, lp, val = model.get_action(obs_t)
            reward = (idx[:, 0].numpy() == 2).astype(np.float32)
            buf.add(
                obs_const,
                idx.numpy(),
                lp.numpy(),
                reward,
                val.numpy(),
                np.ones(agents),  # bandit, every step terminal
            )
        with torch.no_grad():
            last_v = model.get_action(torch.from_numpy(obs_const))[2].numpy()
        buf.gae(last_v, gamma=0.99, lam=0.95)
        learner.update(buf)

    final_rew = np.mean([mean_reward() for _ in range(50)])
    assert start_rew < 0.6, f"start already high: {start_rew}"
    assert final_rew > 0.8, f"bandit did not learn: {final_rew}"
