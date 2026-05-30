"""Multi-discrete action space for the gather-wood policy.

Seven categorical heads map to a subset of the gym's ACTION_DTYPE fields. The
remaining action fields (sneak, use, hotbar selection, inventory ops) are held
fixed at zero: the gather-wood task only needs locomotion, looking, and attack.

Head order and bin sizes::

    0 forward     {-1, 0, 1}        BINS[0] = 3
    1 strafe      {-1, 0, 1}        BINS[1] = 3
    2 jump        {0, 1}            BINS[2] = 2
    3 sprint      {0, 1}            BINS[3] = 2
    4 yaw_delta   {-10,-3,0,3,10}   BINS[4] = 5
    5 pitch_delta {-10,-3,0,3,10}   BINS[5] = 5
    6 attack      {0, 1}            BINS[6] = 2
"""
from __future__ import annotations

import numpy as np

from mcai_train.schema import spec

# Categorical sizes per head (sum = 22 = policy logit width).
BINS = [3, 3, 2, 2, 5, 5, 2]

# Per-head value tables: index into these to recover the gym action value.
FORWARD_VALUES = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
STRAFE_VALUES = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
JUMP_VALUES = np.array([0, 1], dtype=np.uint8)
SPRINT_VALUES = np.array([0, 1], dtype=np.uint8)
YAW_VALUES = np.array([-10.0, -3.0, 0.0, 3.0, 10.0], dtype=np.float32)
PITCH_VALUES = np.array([-10.0, -3.0, 0.0, 3.0, 10.0], dtype=np.float32)
ATTACK_VALUES = np.array([0, 1], dtype=np.uint8)


def actions_to_records(idx: np.ndarray) -> np.ndarray:
    """Map (N, 7) per-head action indices to an (N,) ACTION_DTYPE array.

    Fixed fields (sneak, use, selected_slot, inv_*) stay zero from the zeroed
    structured array; only the seven policy-controlled fields are filled.
    """
    idx = np.asarray(idx)
    if idx.ndim != 2 or idx.shape[1] != len(BINS):
        raise ValueError(f"expected (N, {len(BINS)}) action indices, got {idx.shape}")
    n = idx.shape[0]
    out = np.zeros(n, dtype=spec.ACTION_DTYPE)
    out["forward"] = FORWARD_VALUES[idx[:, 0]]
    out["strafe"] = STRAFE_VALUES[idx[:, 1]]
    out["jump"] = JUMP_VALUES[idx[:, 2]]
    out["sprint"] = SPRINT_VALUES[idx[:, 3]]
    out["yaw_delta"] = YAW_VALUES[idx[:, 4]]
    out["pitch_delta"] = PITCH_VALUES[idx[:, 5]]
    out["attack"] = ATTACK_VALUES[idx[:, 6]]
    return out
