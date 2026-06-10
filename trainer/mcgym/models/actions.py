"""multi discrete action space for the wood policy."""
from __future__ import annotations

import numpy as np

from mcgym.schema import spec

# heads forward strafe jump sprint yaw pitch attack, rest of ACTION_DTYPE stays zero
BINS = [3, 3, 2, 2, 5, 5, 2]

# per head value tables, index recovers the gym action value
FORWARD = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
STRAFE = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
JUMP = np.array([0, 1], dtype=np.uint8)
SPRINT = np.array([0, 1], dtype=np.uint8)
YAW = np.array([-10.0, -3.0, 0.0, 3.0, 10.0], dtype=np.float32)
PITCH = np.array([-10.0, -3.0, 0.0, 3.0, 10.0], dtype=np.float32)
ATTACK = np.array([0, 1], dtype=np.uint8)


def actions_to_records(idx: np.ndarray) -> np.ndarray:
    """map (N, 7) head indices to an (N,) ACTION_DTYPE array."""
    idx = np.asarray(idx)
    if idx.ndim != 2 or idx.shape[1] != len(BINS):
        raise ValueError(f"expected (N, {len(BINS)}) action indices, got {idx.shape}")
    out = np.zeros(idx.shape[0], dtype=spec.ACTION_DTYPE)
    out["forward"] = FORWARD[idx[:, 0]]
    out["strafe"] = STRAFE[idx[:, 1]]
    out["jump"] = JUMP[idx[:, 2]]
    out["sprint"] = SPRINT[idx[:, 3]]
    out["yaw_delta"] = YAW[idx[:, 4]]
    out["pitch_delta"] = PITCH[idx[:, 5]]
    out["attack"] = ATTACK[idx[:, 6]]
    return out
