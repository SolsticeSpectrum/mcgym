"""on policy rollout storage with gae"""
from __future__ import annotations

import numpy as np
import torch

from mcgym.schema import spec
from mcgym.models.policy import tensors


class Buffer:
    # stores a (T, N) grid, obs kept as raw structured records and
    # encoded to tensors at update time so the buffer is cheap to hold
    def __init__(self, rollout: int, agents: int, n_heads: int = 7) -> None:
        self.T       = rollout
        self.N       = agents
        self.n_heads = n_heads

        self.obs     = np.zeros((rollout, agents), dtype=spec.OBS_DTYPE)
        self.act     = np.zeros((rollout, agents, n_heads), dtype=np.int64)
        self.logprob = np.zeros((rollout, agents), dtype=np.float32)
        self.reward  = np.zeros((rollout, agents), dtype=np.float32)
        self.value   = np.zeros((rollout, agents), dtype=np.float32)
        self.done    = np.zeros((rollout, agents), dtype=np.float32)

        self._t         = 0
        self.advantages = None
        self.returns    = None

    def reset(self) -> None:
        self._t         = 0
        self.advantages = None
        self.returns    = None

    def add(self, obs, act, logprob, reward, value, done) -> None:
        t = self._t
        self.obs    [t] = obs
        self.act    [t] = np.asarray(act)
        self.logprob[t] = np.asarray(logprob)
        self.reward [t] = np.asarray(reward)
        self.value  [t] = np.asarray(value)
        self.done   [t] = np.asarray(done, dtype=np.float32)
        self._t += 1

    def gae(self, last_value, gamma: float = 0.99, lam: float = 0.95):
        """gae over the grid, last_value bootstraps the state after the final step"""
        last_value       = np.asarray(last_value, dtype=np.float32)
        adv              = np.zeros((self.T, self.N), dtype=np.float32)
        next_value       = last_value
        next_nonterminal = 1.0 - self.done[self.T - 1]
        gae              = np.zeros(self.N, dtype=np.float32)

        for t in reversed(range(self.T)):
            if t < self.T - 1:
                next_nonterminal = 1.0 - self.done[t]
                next_value       = self.value[t + 1]
            delta  = self.reward[t] + gamma * next_value * next_nonterminal - self.value[t]
            gae    = delta + gamma * lam * next_nonterminal * gae
            adv[t] = gae

        returns         = adv + self.value
        self.advantages = adv.reshape(-1)
        self.returns    = returns.reshape(-1)

        return self.advantages, self.returns

    def to_device(self, device) -> dict:
        # encode whole rollout once per update, the structured gather plus h2d
        # upload is the expensive part, epochs only need a fresh shuffle
        return {
            "obs":     tensors(self.obs.reshape(-1), device),
            "actions": torch.from_numpy(self.act.reshape(-1, self.n_heads)).to(device),
            "logprob": torch.from_numpy(self.logprob.reshape(-1)).to(device),
            "adv":     torch.from_numpy(self.advantages).to(device),
            "ret":     torch.from_numpy(self.returns).to(device),
            "total":   self.T * self.N,
        }

    def batches(self, batch_size: int, device, data: dict | None = None):
        """yield shuffled minibatches, pass data from to_device to reuse one encode"""
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
            )
