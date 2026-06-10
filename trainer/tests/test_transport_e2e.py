"""real round trip against the live gym, agents walk and voxels report real blocks."""
from __future__ import annotations

import pathlib
import tempfile
import uuid

import numpy as np
import pytest

from mcgym.env import Transport, launch
from mcgym.schema import codec, spec
from mcgym.schema.registry import Registry

REGISTRY = pathlib.Path(__file__).resolve().parents[2] / "schema" / "registry.json"


def _walk_forward_actions(n: int) -> np.ndarray:
    records = [
        {
            "forward": 1.0,
            "strafe": 0.0,
            "jump": 0,
            "sneak": 0,
            "sprint": 1,
            "yaw_delta": 0.0,
            "pitch_delta": 0.0,
            "attack": 0,
            "use": 0,
            "selected_slot": 0,
            "inv_op_type": 0,
            "inv_slot_a": 0,
            "inv_slot_b": 0,
        }
        for _ in range(n)
    ]
    return np.frombuffer(codec.encode_action_batch(records), dtype=spec.ACTION_DTYPE).copy()


@pytest.mark.slow
def test_transport_round_trip():
    agents = 2
    seed = 0
    registry = Registry.load(REGISTRY)
    air_id = registry.id_of("minecraft:air")
    assert air_id == 0

    tmpdir = tempfile.mkdtemp(prefix="mcai_sock_")
    shm_path = f"/dev/shm/mcai_shm_{uuid.uuid4().hex}.bin"
    sock_path = str(pathlib.Path(tmpdir) / "gym.sock")

    proc = launch(agents, seed, shm_path, sock_path)
    transport = None
    try:
        transport = Transport(shm_path, sock_path, agents)

        obs0 = transport.reset()
        assert obs0.shape == (agents,)
        assert obs0.dtype == spec.OBS_DTYPE
        assert obs0["schema_version"].tolist() == [spec.SCHEMA_VERSION] * agents
        assert obs0["agent_id"].tolist() == [0, 1]

        start_pos = obs0["pos"].copy()
        actions = _walk_forward_actions(agents)
        obs = obs0
        for _ in range(20):
            obs = transport.step(actions)

        assert obs.dtype == spec.OBS_DTYPE
        assert obs["agent_id"].tolist() == [0, 1]

        # terrain can block an agent at some seeds, physics is proven if any agent walks
        moved = np.abs(obs["pos"] - start_pos).max(axis=1)
        assert (moved > 1.0).any(), f"no agent moved, physics dead: {moved.tolist()}"

        for agent in range(agents):
            voxels = obs["voxel_blocks"][agent]
            assert (voxels != air_id).any(), f"agent {agent} voxel grid is all air"
    finally:
        if transport is not None:
            transport.close()
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        pathlib.Path(shm_path).unlink(missing_ok=True)
        pathlib.Path(sock_path).unlink(missing_ok=True)


if __name__ == "__main__":
    test_transport_round_trip()
    print("OK")
