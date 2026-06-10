"""Shared-backbone actor-critic for the gather-wood multi-discrete policy.

The encoder fuses four observation streams:

* a 17x17x17 NEAR voxel grid (stride 1, radius 8) of block ids, embedded + 3D CNN,
* a 17x17x17 FAR voxel grid (stride N, radius 8*N) — a foveated render-distance
  field sharing the block embedding, through its own 3D CNN, so the policy can see
  and path toward distant trees/terrain,
* per-agent scalars (velocity, look angles, health/food, target geometry),
* the 41-slot inventory of item ids, count-weighted and pooled.

Absolute world position is deliberately excluded so the policy generalises
across spawn locations.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from mcai_train.schema import spec

from .action_space import BINS

VOXEL_EDGE = spec.VOXEL_EDGE
VOXEL_COUNT = VOXEL_EDGE ** 3
INV_SLOTS = spec.PARAMS["inventory_slots"]

EMBED_DIM = 8
# Scalar feature width: vel(3) + sin/cos yaw(2) + sin/cos pitch(2) + on_ground(1)
# + health(1) + food(1) + target_distance(1) + target_in_range(1) + face onehot(6).
SCALAR_DIM = 3 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 6
LATENT_DIM = 256


def obs_to_tensors(obs_struct_batch: np.ndarray, device) -> dict:
    """Extract the tensors the encoder needs from an (N,) OBS_DTYPE batch.

    Returns voxel ids and inventory ids as long tensors (for the embeddings),
    everything else as float tensors, all on ``device``.
    """
    obs = obs_struct_batch

    # Keep the voxel/id grids as int32 for the CPU->GPU transfer (they are i32 in the schema);
    # upcast to long on the GPU inside the encoder. Transferring int32 instead of int64 halves
    # the H2D traffic, which dominates both collect and the per-minibatch update re-encode.
    # .copy(): structured-array field views always need the copy anyway, and unlike
    # ascontiguousarray it also fixes the misaligned strides a batch-of-1 view reports
    # (torch.from_numpy rejects those — hit by the ONNX export path).
    voxel = obs["voxel_blocks"].copy()
    voxel_far = obs["voxel_far"].copy()
    target_block = obs["target_block"].copy()

    vel = np.ascontiguousarray(obs["vel"]).astype(np.float32)
    yaw = np.ascontiguousarray(obs["yaw"]).astype(np.float32)
    pitch = np.ascontiguousarray(obs["pitch"]).astype(np.float32)
    on_ground = np.ascontiguousarray(obs["on_ground"]).astype(np.float32)
    health = np.ascontiguousarray(obs["health"]).astype(np.float32)
    food = np.ascontiguousarray(obs["food"]).astype(np.float32)
    target_distance = np.ascontiguousarray(obs["target_distance"]).astype(np.float32)
    target_in_range = np.ascontiguousarray(obs["target_in_range"]).astype(np.float32)
    target_face = np.ascontiguousarray(obs["target_face"]).astype(np.int64)

    inv_item_id = obs["inv_item_id"].copy()
    inv_count = np.ascontiguousarray(obs["inv_count"]).astype(np.float32)

    yaw_r = np.deg2rad(yaw)
    pitch_r = np.deg2rad(pitch)

    # Face one-hot: valid faces are 0..5; the sentinel 255 maps to all-zeros.
    n = obs.shape[0]
    face_onehot = np.zeros((n, 6), dtype=np.float32)
    valid = target_face < 6
    face_onehot[np.arange(n)[valid], target_face[valid]] = 1.0

    scalars = np.concatenate(
        [
            vel,
            np.stack([np.sin(yaw_r), np.cos(yaw_r)], axis=1),
            np.stack([np.sin(pitch_r), np.cos(pitch_r)], axis=1),
            on_ground[:, None],
            (health / 20.0)[:, None],
            (food / 20.0)[:, None],
            (target_distance / 8.0)[:, None],
            target_in_range[:, None],
            face_onehot,
        ],
        axis=1,
    ).astype(np.float32)

    return {
        "voxel": torch.from_numpy(voxel).to(device),
        "voxel_far": torch.from_numpy(voxel_far).to(device),
        "target_block": torch.from_numpy(target_block).to(device),
        "scalars": torch.from_numpy(scalars).to(device),
        "inv_item_id": torch.from_numpy(inv_item_id).to(device),
        "inv_count": torch.from_numpy(inv_count).to(device),
    }


class ObsEncoder(nn.Module):
    def __init__(self, num_blocks: int, num_items: int, scale: int = 1) -> None:
        super().__init__()
        # `scale` multiplies every width. scale=1 is the original 646k-param model (tuned for a
        # GTX 1060); on a big GPU raise it (e.g. 4-8) for more capacity + far more GPU work per
        # forward, which both fills VRAM and closes the collect-phase GPU-idle gaps.
        self.scale = scale
        embed_dim = EMBED_DIM * scale
        c1, c2 = 8 * scale, 16 * scale
        self.embed_dim = embed_dim
        self.block_embed = nn.Embedding(num_blocks, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.num_blocks = num_blocks
        self.num_items = num_items

        # Near and far voxel grids each get their own 3D CNN (distinct weights) but share the
        # block embedding table above. 17 -> 9 -> 5 with stride-2 convs.
        def _voxel_cnn():
            return nn.Sequential(
                nn.Conv3d(embed_dim, c1, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv3d(c1, c2, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )

        self.voxel_conv = _voxel_cnn()
        self.voxel_conv_far = _voxel_cnn()
        conv_out = c2 * 5 * 5 * 5
        self.voxel_fc = nn.Sequential(nn.Linear(conv_out, 128 * scale), nn.ReLU())
        self.voxel_fc_far = nn.Sequential(nn.Linear(conv_out, 128 * scale), nn.ReLU())

        self.scalar_fc = nn.Sequential(nn.Linear(SCALAR_DIM, 64 * scale), nn.ReLU())
        self.inv_fc = nn.Sequential(nn.Linear(embed_dim, 32 * scale), nn.ReLU())
        # Identity of the block under the crosshair (shares the block embedding) so the
        # policy can tell a trunk from leaves when it has something in range.
        self.target_fc = nn.Sequential(nn.Linear(embed_dim, 16 * scale), nn.ReLU())

        self.latent_dim = LATENT_DIM * scale
        self.fuse = nn.Sequential(
            nn.Linear((128 + 128 + 64 + 32 + 16) * scale, self.latent_dim), nn.ReLU()
        )

    def encode(self, obs_tensors: dict) -> torch.Tensor:
        voxel = obs_tensors["voxel"].clamp(0, self.num_blocks - 1).long()
        b = voxel.shape[0]
        v = self.block_embed(voxel)  # (B, 4913, 8)
        v = v.permute(0, 2, 1).reshape(b, self.embed_dim, VOXEL_EDGE, VOXEL_EDGE, VOXEL_EDGE)
        v = self.voxel_conv(v).reshape(b, -1)
        v = self.voxel_fc(v)

        voxel_far = obs_tensors["voxel_far"].clamp(0, self.num_blocks - 1).long()
        vf = self.block_embed(voxel_far)
        vf = vf.permute(0, 2, 1).reshape(b, self.embed_dim, VOXEL_EDGE, VOXEL_EDGE, VOXEL_EDGE)
        vf = self.voxel_conv_far(vf).reshape(b, -1)
        vf = self.voxel_fc_far(vf)

        s = self.scalar_fc(obs_tensors["scalars"])

        inv_ids = obs_tensors["inv_item_id"].clamp(0, self.num_items - 1).long()
        inv_emb = self.item_embed(inv_ids)  # (B, 41, 8)
        weight = torch.log1p(obs_tensors["inv_count"]).unsqueeze(-1)  # (B, 41, 1)
        inv = (inv_emb * weight).sum(dim=1)  # (B, 8)
        inv = self.inv_fc(inv)

        tb = obs_tensors["target_block"].clamp(0, self.num_blocks - 1).long()
        tgt = self.target_fc(self.block_embed(tb))  # (B, 16)

        return self.fuse(torch.cat([v, vf, s, inv, tgt], dim=1))


class ActorCritic(nn.Module):
    def __init__(self, num_blocks: int, num_items: int, scale: int = 1) -> None:
        super().__init__()
        self.encoder = ObsEncoder(num_blocks, num_items, scale=scale)
        latent = self.encoder.latent_dim
        self.policy_head = nn.Linear(latent, sum(BINS))
        self.value_head = nn.Linear(latent, 1)
        self.bins = list(BINS)

    def forward(self, obs_tensors: dict):
        latent = self.encoder.encode(obs_tensors)
        return self.policy_head(latent), self.value_head(latent).squeeze(-1)

    def _dists(self, logits: torch.Tensor):
        return [Categorical(logits=chunk) for chunk in torch.split(logits, self.bins, dim=1)]

    def get_action(self, obs_tensors: dict, deterministic: bool = False):
        logits, value = self.forward(obs_tensors)
        dists = self._dists(logits)
        if deterministic:
            actions = [d.probs.argmax(dim=-1) for d in dists]
        else:
            actions = [d.sample() for d in dists]
        action_idx = torch.stack(actions, dim=1)  # (B, 7)
        logprob = sum(d.log_prob(a) for d, a in zip(dists, actions))
        return action_idx, logprob, value

    def evaluate(self, obs_tensors: dict, action_idx: torch.Tensor):
        logits, value = self.forward(obs_tensors)
        dists = self._dists(logits)
        cols = action_idx.unbind(dim=1)
        logprob = sum(d.log_prob(a) for d, a in zip(dists, cols))
        entropy = sum(d.entropy() for d in dists)
        return logprob, entropy, value
