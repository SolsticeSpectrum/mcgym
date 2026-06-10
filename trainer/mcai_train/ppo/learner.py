"""PPO update: clipped surrogate policy loss, value MSE, entropy bonus.

Mirrors the mechanics of rlgym-ppo: ratio clipping, advantage normalisation,
value-function coefficient, entropy coefficient, global grad-norm clip, and
shuffled minibatch epochs.
"""
from __future__ import annotations

import os
import time

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
        # Fused Adam steps all parameters in one kernel instead of one launch per tensor —
        # the model is many small tensors, so the launch overhead dominates the eager step.
        fused = str(device).startswith("cuda")
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=fused)
        # bf16 autocast for the SGD forward/backward: the encoder is bandwidth-bound 3D conv +
        # embedding-gather work, so halving the bytes (and hitting tensor cores) is the win.
        # Loss math stays fp32 (computed outside the autocast region); weights/optimizer fp32.
        self.autocast = fused and bool(os.environ.get("MCAI_BF16"))

    def update(self, buffer) -> dict:
        # Metrics stay 0-dim GPU tensors until the end: a .item() per minibatch is a full
        # device sync, which stalls the SGD pipeline hundreds of times per update.
        pol_losses, val_losses, entropies, clip_fracs, approx_kls = [], [], [], [], []

        t0 = time.perf_counter()
        data = buffer.to_device(self.device)  # encode the rollout once; epochs reshuffle on-device
        enc_s = time.perf_counter() - t0

        for _ in range(self.epochs):
            for (
                obs_tensors,
                action_idx,
                old_logprob,
                advantages,
                returns,
                old_value,
            ) in buffer.iter_minibatches(self.minibatch, self.device, data=data):
                adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
                    logprob, entropy, value = self.model.evaluate(obs_tensors, action_idx)
                logprob, entropy, value = logprob.float(), entropy.float(), value.float()

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
                    clip_fracs.append(((ratio - 1.0).abs() > self.clip).float().mean())
                    approx_kls.append((old_logprob - logprob).mean())
                pol_losses.append(policy_loss.detach())
                val_losses.append(value_loss.detach())
                entropies.append(entropy_loss.detach())

        metrics = {
            "policy_loss": torch.stack(pol_losses).mean().item(),
            "value_loss": torch.stack(val_losses).mean().item(),
            "entropy": torch.stack(entropies).mean().item(),
            "clip_frac": torch.stack(clip_fracs).mean().item(),
            "approx_kl": torch.stack(approx_kls).mean().item(),
        }
        if os.environ.get("MCAI_PROFILE"):
            print(f"[profile] update encode={enc_s:.1f}s sgd={time.perf_counter() - t0 - enc_s:.1f}s",
                  flush=True)
        return metrics
