"""Pack/unpack observation & action records to/from the canonical byte layout."""
import numpy as np

from . import spec


def encode_obs(record: dict) -> bytes:
    arr = np.zeros(1, dtype=spec.OBS_DTYPE)
    for key, value in record.items():
        arr[0][key] = value
    return arr.tobytes()


def decode_obs(buf: bytes) -> np.void:
    return np.frombuffer(buf, dtype=spec.OBS_DTYPE, count=1)[0]


def decode_obs_batch(buf, n: int) -> np.ndarray:
    """Zero-copy structured-array view of N observation records from a buffer."""
    return np.frombuffer(buf, dtype=spec.OBS_DTYPE, count=n)


def encode_action(record: dict) -> bytes:
    arr = np.zeros(1, dtype=spec.ACTION_DTYPE)
    for key, value in record.items():
        arr[0][key] = value
    return arr.tobytes()


def decode_action(buf: bytes) -> np.void:
    return np.frombuffer(buf, dtype=spec.ACTION_DTYPE, count=1)[0]


def encode_action_batch(records) -> bytes:
    arr = np.zeros(len(records), dtype=spec.ACTION_DTYPE)
    for i, record in enumerate(records):
        for key, value in record.items():
            arr[i][key] = value
    return arr.tobytes()
