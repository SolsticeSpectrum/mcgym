"""end to end proof of real block mining through the live gym.

drives a single agent over the shm transport and asserts a real item enters the
inventory obs (break, drop, pickup) and that target_* populates while attacking.
"""
from __future__ import annotations

import json
import math
import pathlib
import tempfile
import uuid

import numpy as np
import pytest

from mcgym.env import Transport, launch
from mcgym.schema import codec, spec
from mcgym.schema.registry import Registry

REGISTRY = pathlib.Path(__file__).resolve().parents[2] / "schema" / "registry.json"
R = spec.PARAMS["voxel_radius"]
E = spec.VOXEL_EDGE


def _action(
    *,
    forward: float = 0.0,
    pitch_delta: float = 0.0,
    yaw_delta: float = 0.0,
    attack: int = 0,
) -> np.ndarray:
    rec = {
        "forward": forward,
        "strafe": 0.0,
        "jump": 0,
        "sneak": 0,
        "sprint": 0,
        "yaw_delta": yaw_delta,
        "pitch_delta": pitch_delta,
        "attack": attack,
        "use": 0,
        "selected_slot": 0,
        "inv_op_type": 0,
        "inv_slot_a": 0,
        "inv_slot_b": 0,
    }
    return np.frombuffer(codec.encode_action_batch([rec]), dtype=spec.ACTION_DTYPE).copy()


def _offset(idx: int) -> tuple[int, int, int]:
    # voxel layout is ((dy+R)*E + (dz+R))*E + (dx+R)
    dx = idx % E - R
    rem = idx // E
    dz = rem % E - R
    dy = rem // E - R
    return dx, dy, dz


def _has_item(row) -> bool:
    ids = np.asarray(row["inv_item_id"])
    counts = np.asarray(row["inv_count"])
    return bool(((ids != 0) & (counts > 0)).any())


def _aim(dx: int, dy: int, dz: int) -> tuple[float, float]:
    # absolute yaw and pitch to look at voxel center, eyes sit ~1.62 above feet
    tx = dx + 0.5
    ty = dy + 0.5 - 1.62
    tz = dz + 0.5
    horiz = math.sqrt(tx * tx + tz * tz)
    # vanilla yaw 0 faces +Z and increases toward -X
    yaw = -math.degrees(math.atan2(tx, tz))
    pitch = -math.degrees(math.atan2(ty, horiz))
    return yaw, pitch


def _run(transport, build_action, steps):
    # returns last obs, first obs with inventory, and whether target was ever in range
    last = None
    saw_target = False
    gained = None
    for i in range(steps):
        obs = transport.step(build_action(i))
        row = obs[0]
        last = obs
        if int(row["target_in_range"]) == 1 and int(row["target_block"]) != 0:
            saw_target = True
        if gained is None and _has_item(row):
            gained = obs.copy()
    return last, gained, saw_target


@pytest.mark.slow
def test_mining_e2e():
    agents = 1
    seed = 0
    registry = Registry.load(REGISTRY)

    tmpdir = tempfile.mkdtemp(prefix="mcai_sock_")
    shm_path = f"/dev/shm/mcai_shm_{uuid.uuid4().hex}.bin"
    sock_path = str(pathlib.Path(tmpdir) / "gym.sock")

    proc = launch(agents, seed, shm_path, sock_path)
    transport = None
    try:
        transport = Transport(shm_path, sock_path, agents)
        obs0 = transport.reset()
        assert obs0.shape == (agents,)

        reg_doc = json.loads(REGISTRY.read_text())
        # voxel_blocks uses block ids, the drop in inv_item_id uses item ids
        oak_block = reg_doc["blocks"]["minecraft:oak_log"]
        oak_item = reg_doc["items"]["minecraft:oak_log"]

        # scan pristine spawn voxels for a reachable oak_log before mining alters the
        # world (reset restores pose but not broken blocks). only logs at body level
        # within bare hand reach are practically mineable from spawn
        voxels = obs0[0]["voxel_blocks"]
        log_idx = np.where(np.asarray(voxels) == oak_block)[0]
        log_offset = None
        best = None
        for idx in log_idx:
            dx, dy, dz = _offset(int(idx))
            if abs(dy) > 1:
                continue
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist <= 3.5 and (best is None or dist < best[0]):
                best = (dist, dx, dy, dz)
        if best is not None:
            log_offset = best[1:]

        # block below proof, runs first on the pristine world
        # dig straight down in bursts, ~22 ticks attack to break then ~10 idle so the
        # drop clears its pickup delay and gets collected before the agent descends
        cur_pitch = float(obs0[0]["pitch"])
        period = 32
        dig_ticks = 22

        def _down(i):
            nonlocal cur_pitch
            dpitch = max(-15.0, min(15.0, 90.0 - cur_pitch))
            cur_pitch += dpitch
            digging = (i % period) < dig_ticks
            return _action(forward=0.0, pitch_delta=dpitch, attack=1 if digging else 0)

        last, gained, saw_target = _run(transport, _down, 6 * period)

        assert saw_target, "agent never had a valid target_in_range while attacking down"

        proof = gained if gained is not None else last
        assert _has_item(proof[0]), (
            "no item entered the inventory obs after mining the block below, "
            f"inv_item_id nonzero={np.asarray(last[0]['inv_item_id']).nonzero()[0].tolist()}"
        )

        items = np.asarray(proof[0]["inv_item_id"])
        counts = np.asarray(proof[0]["inv_count"])
        gathered = [
            (registry.name_of(int(it)), int(c))
            for it, c in zip(items, counts)
            if int(it) != 0 and int(c) > 0
        ]
        print(f"block below proof gathered {gathered}")

        # best effort oak_log path, runs after and may disturb the world
        oak_done = False
        if log_offset is not None:
            dx, dy, dz = log_offset
            # trunk still stands, the block below proof dug a pit not toward the tree
            base = transport.reset()
            yaw, pitch = _aim(dx, dy, dz)
            print(f"oak_log reachable at offset ({dx},{dy},{dz}), aiming yaw={yaw:.1f} pitch={pitch:.1f}")

            cyaw = float(base[0]["yaw"])
            cpitch = float(base[0]["pitch"])

            def _trunk(i, _yaw=yaw, _pitch=pitch):
                nonlocal cyaw, cpitch
                dyaw = max(-30.0, min(30.0, _yaw - cyaw))
                dpitch = max(-30.0, min(30.0, _pitch - cpitch))
                cyaw += dyaw
                cpitch += dpitch
                aligned = abs(_yaw - cyaw) < 15.0 and abs(_pitch - cpitch) < 15.0
                # creep forward and hop to clear foliage while digging, then idle so the
                # dropped log clears its pickup delay and gets collected
                digging = i < 150
                forward = 0.5 if (aligned and digging) else 0.0
                jump = 1 if (aligned and digging and i % 20 < 3) else 0
                rec = {
                    "forward": forward, "strafe": 0.0, "jump": jump, "sneak": 0, "sprint": 0,
                    "yaw_delta": dyaw, "pitch_delta": dpitch, "attack": 1 if digging else 0,
                    "use": 0, "selected_slot": 0, "inv_op_type": 0, "inv_slot_a": 0, "inv_slot_b": 0,
                }
                return np.frombuffer(codec.encode_action_batch([rec]), dtype=spec.ACTION_DTYPE).copy()

            last, gained, _ = _run(transport, _trunk, 190)
            li = np.asarray(last[0]["inv_item_id"])
            lc = np.asarray(last[0]["inv_count"])
            if oak_item in li[lc > 0].tolist():
                oak_done = True
                print("oak_log path succeeded, oak_log item in inventory obs")
            if not oak_done:
                print("oak_log path ran but collected no oak_log, "
                      "block below proof already established real mining works")
        else:
            print("no oak_log reachable from spawn voxels, relying on block below proof only")

        print(f"oak_log specific path succeeded: {oak_done}")
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
    test_mining_e2e()
    print("OK")
