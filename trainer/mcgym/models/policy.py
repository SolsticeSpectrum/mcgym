"""actor critic with shared encoder for the multi discrete wood policy."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from mcgym.schema import spec

from .actions import BINS

VOXEL_EDGE = spec.VOXEL_EDGE

EMBED_DIM = 8
# vel 3 + yaw sincos 2 + pitch sincos 2 + ground 1 + health 1 + food 1
# + dist 1 + in range 1 + face onehot 6
SCALAR_DIM = 3 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 6
LATENT_DIM = 256


def tensors(obs: np.ndarray, device) -> dict:
    """extract encoder tensors from an (N,) OBS_DTYPE batch."""
    # id grids stay i32 for h2d transfer, encoder upcasts to long on gpu
    # .copy() also fixes misaligned strides on batch of 1 views which
    # torch.from_numpy rejects (hit by onnx export)
    voxel = obs["voxel_blocks"].copy()
    far = obs["voxel_far"].copy()
    target = obs["target_block"].copy()

    vel = np.ascontiguousarray(obs["vel"]).astype(np.float32)
    yaw = np.ascontiguousarray(obs["yaw"]).astype(np.float32)
    pitch = np.ascontiguousarray(obs["pitch"]).astype(np.float32)
    ground = np.ascontiguousarray(obs["on_ground"]).astype(np.float32)
    health = np.ascontiguousarray(obs["health"]).astype(np.float32)
    food = np.ascontiguousarray(obs["food"]).astype(np.float32)
    dist = np.ascontiguousarray(obs["target_distance"]).astype(np.float32)
    in_range = np.ascontiguousarray(obs["target_in_range"]).astype(np.float32)
    face = np.ascontiguousarray(obs["target_face"]).astype(np.int64)

    inv_id = obs["inv_item_id"].copy()
    inv_count = np.ascontiguousarray(obs["inv_count"]).astype(np.float32)

    yaw_r = np.deg2rad(yaw)
    pitch_r = np.deg2rad(pitch)

    # valid faces 0-5, sentinel 255 maps to all zeros
    n = obs.shape[0]
    onehot = np.zeros((n, 6), dtype=np.float32)
    valid = face < 6
    onehot[np.arange(n)[valid], face[valid]] = 1.0

    scalars = np.concatenate(
        [
            vel,
            np.stack([np.sin(yaw_r), np.cos(yaw_r)], axis=1),
            np.stack([np.sin(pitch_r), np.cos(pitch_r)], axis=1),
            ground[:, None],
            (health / 20.0)[:, None],
            (food / 20.0)[:, None],
            (dist / 8.0)[:, None],
            in_range[:, None],
            onehot,
        ],
        axis=1,
    ).astype(np.float32)

    return {
        "voxel": torch.from_numpy(voxel).to(device),
        "voxel_far": torch.from_numpy(far).to(device),
        "target_block": torch.from_numpy(target).to(device),
        "scalars": torch.from_numpy(scalars).to(device),
        "inv_item_id": torch.from_numpy(inv_id).to(device),
        "inv_count": torch.from_numpy(inv_count).to(device),
    }


class Encoder(nn.Module):
    def __init__(self, num_blocks: int, num_items: int, scale: int = 1) -> None:
        super().__init__()
        # scale multiplies every width, 1 is the 646k param model, raise on a big gpu
        self.scale = scale
        embed = EMBED_DIM * scale
        c1, c2 = 8 * scale, 16 * scale
        self.embed_dim = embed
        self.block_embed = nn.Embedding(num_blocks, embed)
        self.item_embed = nn.Embedding(num_items, embed)
        self.num_blocks = num_blocks
        self.num_items = num_items

        # near and far grids get own cnns but share the block embedding
        # 17 to 9 to 5 with stride 2 convs
        def cnn():
            return nn.Sequential(
                nn.Conv3d(embed, c1, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv3d(c1, c2, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )

        self.voxel_conv = cnn()
        self.voxel_conv_far = cnn()
        conv_out = c2 * 5 * 5 * 5
        self.voxel_fc = nn.Sequential(nn.Linear(conv_out, 128 * scale), nn.ReLU())
        self.voxel_fc_far = nn.Sequential(nn.Linear(conv_out, 128 * scale), nn.ReLU())

        self.scalar_fc = nn.Sequential(nn.Linear(SCALAR_DIM, 64 * scale), nn.ReLU())
        self.inv_fc = nn.Sequential(nn.Linear(embed, 32 * scale), nn.ReLU())
        # block under crosshair so policy can tell trunk from leaves
        self.target_fc = nn.Sequential(nn.Linear(embed, 16 * scale), nn.ReLU())

        self.latent_dim = LATENT_DIM * scale
        self.fuse = nn.Sequential(
            nn.Linear((128 + 128 + 64 + 32 + 16) * scale, self.latent_dim), nn.ReLU()
        )

    def encode(self, obs: dict) -> torch.Tensor:
        voxel = obs["voxel"].clamp(0, self.num_blocks - 1).long()
        b = voxel.shape[0]
        v = self.block_embed(voxel)
        v = v.permute(0, 2, 1).reshape(b, self.embed_dim, VOXEL_EDGE, VOXEL_EDGE, VOXEL_EDGE)
        v = self.voxel_conv(v).reshape(b, -1)
        v = self.voxel_fc(v)

        far = obs["voxel_far"].clamp(0, self.num_blocks - 1).long()
        vf = self.block_embed(far)
        vf = vf.permute(0, 2, 1).reshape(b, self.embed_dim, VOXEL_EDGE, VOXEL_EDGE, VOXEL_EDGE)
        vf = self.voxel_conv_far(vf).reshape(b, -1)
        vf = self.voxel_fc_far(vf)

        s = self.scalar_fc(obs["scalars"])

        # count weighted pool over inventory slots
        ids = obs["inv_item_id"].clamp(0, self.num_items - 1).long()
        weight = torch.log1p(obs["inv_count"]).unsqueeze(-1)
        inv = (self.item_embed(ids) * weight).sum(dim=1)
        inv = self.inv_fc(inv)

        tb = obs["target_block"].clamp(0, self.num_blocks - 1).long()
        tgt = self.target_fc(self.block_embed(tb))

        return self.fuse(torch.cat([v, vf, s, inv, tgt], dim=1))


class ActorCritic(nn.Module):
    def __init__(self, num_blocks: int, num_items: int, scale: int = 1) -> None:
        super().__init__()
        self.encoder = Encoder(num_blocks, num_items, scale=scale)
        latent = self.encoder.latent_dim
        self.policy_head = nn.Linear(latent, sum(BINS))
        self.value_head = nn.Linear(latent, 1)
        self.bins = list(BINS)

    def forward(self, obs: dict):
        latent = self.encoder.encode(obs)
        return self.policy_head(latent), self.value_head(latent).squeeze(-1)

    def dists(self, logits: torch.Tensor):
        return [Categorical(logits=c) for c in torch.split(logits, self.bins, dim=1)]

    def get_action(self, obs: dict, deterministic: bool = False):
        logits, value = self.forward(obs)
        dists = self.dists(logits)
        if deterministic:
            acts = [d.probs.argmax(dim=-1) for d in dists]
        else:
            acts = [d.sample() for d in dists]
        act = torch.stack(acts, dim=1)
        logprob = sum(d.log_prob(a) for d, a in zip(dists, acts))
        return act, logprob, value

    def evaluate(self, obs: dict, act: torch.Tensor):
        logits, value = self.forward(obs)
        dists = self.dists(logits)
        cols = act.unbind(dim=1)
        logprob = sum(d.log_prob(a) for d, a in zip(dists, cols))
        entropy = sum(d.entropy() for d in dists)
        return logprob, entropy, value
