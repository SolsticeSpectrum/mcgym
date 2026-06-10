"""On-policy rollout storage with GAE.

Stores a (T, N) grid of transitions. Observations are kept as raw structured
records (one (N,) OBS_DTYPE array per timestep) and re-encoded to tensors at
minibatch time, so the buffer is device-agnostic and cheap to hold.
"""
from __future__ import annotations

import numpy as np
import torch

from mcgym.schema import spec


class Buffer:
    def __init__(self, rollout: int, agents: int, n_heads: int = 7) -> None:
        self.T = rollout
        self.N = agents
        self.n_heads = n_heads

        self.obs = np.zeros((rollout, agents), dtype=spec.OBS_DTYPE)
        self.act = np.zeros((rollout, agents, n_heads), dtype=np.int64)
        self.logprob = np.zeros((rollout, agents), dtype=np.float32)
        self.reward = np.zeros((rollout, agents), dtype=np.float32)
        self.value = np.zeros((rollout, agents), dtype=np.float32)
        self.done = np.zeros((rollout, agents), dtype=np.float32)

        self._t = 0
        self.advantages = None
        self.returns = None

    def reset(self) -> None:
        self._t = 0
        self.advantages = None
        self.returns = None

    def add(self, obs, act, logprob, reward, value, done) -> None:
        t = self._t
        self.obs[t] = obs
        self.act[t] = np.asarray(act)
        self.logprob[t] = np.asarray(logprob)
        self.reward[t] = np.asarray(reward)
        self.value[t] = np.asarray(value)
        self.done[t] = np.asarray(done, dtype=np.float32)
        self._t += 1

    def gae(self, last_value, gamma: float = 0.99, lam: float = 0.95):
        """Generalised advantage estimation over the (T, N) grid.

        ``last_value`` is the critic's bootstrap value (N,) for the state after
        the final stored step. Returns flattened (T*N,) advantages and returns.
        """
        last_value = np.asarray(last_value, dtype=np.float32)
        adv = np.zeros((self.T, self.N), dtype=np.float32)
        next_value = last_value
        next_nonterminal = 1.0 - self.done[self.T - 1]
        gae = np.zeros(self.N, dtype=np.float32)
        for t in reversed(range(self.T)):
            if t < self.T - 1:
                next_nonterminal = 1.0 - self.done[t]
                next_value = self.value[t + 1]
            delta = self.reward[t] + gamma * next_value * next_nonterminal - self.value[t]
            gae = delta + gamma * lam * next_nonterminal * gae
            adv[t] = gae
        returns = adv + self.value
        self.advantages = adv.reshape(-1)
        self.returns = returns.reshape(-1)
        return self.advantages, self.returns

    def to_device(self, device) -> dict:
        """Encode the whole rollout to device tensors ONCE for an update.

        The CPU-side strided gather out of the structured obs array + the H2D upload is the
        expensive part (~GBs); epochs only need a fresh shuffle, so the learner encodes once
        and passes the result to batches for every epoch.
        """
        from mcgym.models.policy import tensors

        return {
            "obs": tensors(self.obs.reshape(-1), device),  # dict of (total, ...) tensors
            "actions": torch.from_numpy(self.act.reshape(-1, self.n_heads)).to(device),
            "logprob": torch.from_numpy(self.logprob.reshape(-1)).to(device),
            "adv": torch.from_numpy(self.advantages).to(device),
            "ret": torch.from_numpy(self.returns).to(device),
            "value": torch.from_numpy(self.value.reshape(-1)).to(device),
            "total": self.T * self.N,
        }

    def batches(self, batch_size: int, device, data: dict | None = None):
        """Yield shuffled minibatches of (obs_tensors, act, old_logprob,
        advantages, returns, old_value).

        Pass ``data`` (from to_device) to reuse one encode across all epochs; otherwise this
        encodes the rollout itself. Minibatch gathers are cheap on-device indexing either way.
        """
        if data is None:
            data = self.to_device(device)

        order = torch.randperm(data["total"], device=device)
        for start in range(0, data["total"], batch_size):
            mb = order[start : start + batch_size]
            yield (
                {k: v[mb] for k, v in data["obs"].items()},
                data["actions"][mb],
                data["logprob"][mb],
                data["adv"][mb],
                data["ret"][mb],
                data["value"][mb],
            )
