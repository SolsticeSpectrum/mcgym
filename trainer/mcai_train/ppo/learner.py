"""PPO update: clipped surrogate policy loss, value MSE, entropy bonus.

Mirrors the mechanics of rlgym-ppo: ratio clipping, advantage normalisation,
value-function coefficient, entropy coefficient, global grad-norm clip, and
shuffled minibatch epochs.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PPOLearner:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        clip: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        epochs: int = 3,
        minibatch: int = 2048,
        grad_clip: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.clip = clip
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.epochs = epochs
        self.minibatch = minibatch
        self.grad_clip = grad_clip
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def update(self, buffer) -> dict:
        pol_losses, val_losses, entropies, clip_fracs, approx_kls = [], [], [], [], []

        for _ in range(self.epochs):
            for (
                obs_tensors,
                action_idx,
                old_logprob,
                advantages,
                returns,
                old_value,
            ) in buffer.iter_minibatches(self.minibatch, self.device):
                adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                logprob, entropy, value = self.model.evaluate(obs_tensors, action_idx)

                ratio = torch.exp(logprob - old_logprob)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (returns - value).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    - self.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.clip).float().mean()
                    approx_kl = (old_logprob - logprob).mean()
                pol_losses.append(policy_loss.item())
                val_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())
                clip_fracs.append(clip_frac.item())
                approx_kls.append(approx_kl.item())

        return {
            "policy_loss": float(np.mean(pol_losses)),
            "value_loss": float(np.mean(val_losses)),
            "entropy": float(np.mean(entropies)),
            "clip_frac": float(np.mean(clip_fracs)),
            "approx_kl": float(np.mean(approx_kls)),
        }
