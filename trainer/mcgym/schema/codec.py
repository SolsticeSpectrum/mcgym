"""pack/unpack obs and action records to/from canonical bytes."""
import numpy as np

from . import spec


def encode_obs(rec: dict) -> bytes:
    arr = np.zeros(1, dtype=spec.OBS_DTYPE)
    for key, value in rec.items():
        arr[0][key] = value
    return arr.tobytes()


def decode_obs(buf: bytes) -> np.void:
    return np.frombuffer(buf, dtype=spec.OBS_DTYPE, count=1)[0]


def decode_obs_batch(buf, n: int) -> np.ndarray:
    # zero copy view of n records
    return np.frombuffer(buf, dtype=spec.OBS_DTYPE, count=n)


def encode_action(rec: dict) -> bytes:
    arr = np.zeros(1, dtype=spec.ACTION_DTYPE)
    for key, value in rec.items():
        arr[0][key] = value
    return arr.tobytes()


def decode_action(buf: bytes) -> np.void:
    return np.frombuffer(buf, dtype=spec.ACTION_DTYPE, count=1)[0]


def encode_action_batch(recs) -> bytes:
    arr = np.zeros(len(recs), dtype=spec.ACTION_DTYPE)
    for i, rec in enumerate(recs):
        for key, value in rec.items():
            arr[i][key] = value
    return arr.tobytes()
