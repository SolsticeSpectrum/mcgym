"""On-policy rollout storage with GAE.

Stores a (T, N) grid of transitions. Observations are kept as raw structured
records (one (N,) OBS_DTYPE array per timestep) and re-encoded to tensors at
minibatch time, so the buffer is device-agnostic and cheap to hold.
"""
from __future__ import annotations

import numpy as np
import torch

from mcai_train.schema import spec


class RolloutBuffer:
    def __init__(self, rollout_len: int, n_agents: int, n_heads: int = 7) -> None:
        self.T = rollout_len
        self.N = n_agents
        self.n_heads = n_heads

        self.obs = np.zeros((rollout_len, n_agents), dtype=spec.OBS_DTYPE)
        self.action_idx = np.zeros((rollout_len, n_agents, n_heads), dtype=np.int64)
        self.logprob = np.zeros((rollout_len, n_agents), dtype=np.float32)
        self.reward = np.zeros((rollout_len, n_agents), dtype=np.float32)
        self.value = np.zeros((rollout_len, n_agents), dtype=np.float32)
        self.done = np.zeros((rollout_len, n_agents), dtype=np.float32)

        self._t = 0
        self.advantages = None
        self.returns = None

    def reset(self) -> None:
        self._t = 0
        self.advantages = None
        self.returns = None

    def add(self, obs_struct, action_idx, logprob, reward, value, done) -> None:
        t = self._t
        self.obs[t] = obs_struct
        self.action_idx[t] = np.asarray(action_idx)
        self.logprob[t] = np.asarray(logprob)
        self.reward[t] = np.asarray(reward)
        self.value[t] = np.asarray(value)
        self.done[t] = np.asarray(done, dtype=np.float32)
        self._t += 1

    def compute_gae(self, last_value, gamma: float = 0.99, lam: float = 0.95):
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

    def iter_minibatches(self, batch_size: int, device):
        """Yield shuffled minibatches of (obs_tensors, action_idx, old_logprob,
        advantages, returns, old_value).

        Encodes the whole rollout's obs to GPU ONCE per call and indexes minibatches on-device,
        instead of re-encoding + re-transferring the obs for every minibatch (which moved the full
        obs H2D batch_count times). The big tensors stay resident on the GPU for the call; the
        minibatch gather is a cheap on-device index.
        """
        from mcai_train.models.policy import obs_to_tensors

        total = self.T * self.N

        all_obs = obs_to_tensors(self.obs.reshape(-1), device)  # dict of (total, ...) GPU tensors
        all_actions = torch.from_numpy(self.action_idx.reshape(-1, self.n_heads)).to(device)
        all_logprob = torch.from_numpy(self.logprob.reshape(-1)).to(device)
        all_adv = torch.from_numpy(self.advantages).to(device)
        all_ret = torch.from_numpy(self.returns).to(device)
        all_value = torch.from_numpy(self.value.reshape(-1)).to(device)

        order = torch.randperm(total, device=device)
        for start in range(0, total, batch_size):
            mb = order[start : start + batch_size]
            yield (
                {k: v[mb] for k, v in all_obs.items()},
                all_actions[mb],
                all_logprob[mb],
                all_adv[mb],
                all_ret[mb],
                all_value[mb],
            )
