"""Real Python -> Java -> Python round-trip against the live Minecraft gym.

Launches the gym with 2 agents, resets, then walks forward for several steps and
asserts the agents physically move and that the voxel grid reports real (non-air)
blocks. No stubs: this is the end-to-end proof the transport works.
"""
from __future__ import annotations

import pathlib
import tempfile
import uuid

import numpy as np
import pytest

from mcai_train.env import ShmTransport, launch_gym
from mcai_train.schema import codec, spec
from mcai_train.schema.registry import Registry

REGISTRY_PATH = pathlib.Path("/home/user/github/mcai/schema/registry.json")


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
    n_agents = 2
    seed = 0
    registry = Registry.load(REGISTRY_PATH)
    air_id = registry.id_of("minecraft:air")
    assert air_id == 0

    tmpdir = tempfile.mkdtemp(prefix="mcai_sock_")
    shm_path = f"/dev/shm/mcai_shm_{uuid.uuid4().hex}.bin"
    sock_path = str(pathlib.Path(tmpdir) / "gym.sock")

    proc = launch_gym(n_agents, seed, shm_path, sock_path)
    transport = None
    try:
        transport = ShmTransport(shm_path, sock_path, n_agents)

        obs0 = transport.reset()
        assert obs0.shape == (n_agents,)
        assert obs0.dtype == spec.OBS_DTYPE
        assert obs0["schema_version"].tolist() == [0, 0]
        assert obs0["agent_id"].tolist() == [0, 1]

        start_pos = obs0["pos"].copy()
        actions = _walk_forward_actions(n_agents)
        obs = obs0
        for _ in range(20):
            obs = transport.step(actions)

        assert obs.dtype == spec.OBS_DTYPE
        assert obs["agent_id"].tolist() == [0, 1]

        moved = np.abs(obs["pos"] - start_pos).max(axis=1)
        assert (moved > 1.0).all(), f"agents did not move >1 block: {moved.tolist()}"

        for agent in range(n_agents):
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
