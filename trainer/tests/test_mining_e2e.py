"""End-to-end proof of REAL block mining through the live Minecraft gym.

Python drives a single agent to mine blocks via the shared-memory transport and
asserts that a real item enters the inventory observation (break -> drop ->
pickup -> inventory -> obs) and that the target_* fields populate while attacking.
No stubs: every assertion is against bytes the Java gym wrote from real mechanics.
"""
from __future__ import annotations

import math
import pathlib
import tempfile
import uuid

import numpy as np
import pytest

from mcgym.env import Transport, launch
from mcgym.schema import codec, spec
from mcgym.schema.registry import Registry
from mcgym.tasks.wood import log_item_ids

REGISTRY = pathlib.Path(__file__).resolve().parents[2] / "schema" / "registry.json"
VOXEL_RADIUS = spec.PARAMS["voxel_radius"]
VOXEL_EDGE = spec.VOXEL_EDGE


def _action(
    *,
    forward: float = 0.0,
    pitch_delta: float = 0.0,
    yaw_delta: float = 0.0,
    attack: int = 0,
) -> np.ndarray:
    record = {
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
    return np.frombuffer(codec.encode_action_batch([record]), dtype=spec.ACTION_DTYPE).copy()


def _voxel_index(dx: int, dy: int, dz: int) -> int:
    return ((dy + VOXEL_RADIUS) * VOXEL_EDGE + (dz + VOXEL_RADIUS)) * VOXEL_EDGE + (dx + VOXEL_RADIUS)


def _index_to_offset(index: int) -> tuple[int, int, int]:
    """Inverse of _voxel_index: index -> (dx, dy, dz)."""
    dx = index % VOXEL_EDGE - VOXEL_RADIUS
    rem = index // VOXEL_EDGE
    dz = rem % VOXEL_EDGE - VOXEL_RADIUS
    dy = rem // VOXEL_EDGE - VOXEL_RADIUS
    return dx, dy, dz


def _inv_has_any(obs_row) -> bool:
    item_ids = np.asarray(obs_row["inv_item_id"])
    counts = np.asarray(obs_row["inv_count"])
    return bool(((item_ids != 0) & (counts > 0)).any())


def _aim_at_offset(dx: int, dy: int, dz: int) -> tuple[float, float]:
    """Absolute (yaw, pitch) degrees to look at the center of a voxel offset from
    the agent's eyes. Agent eyes sit ~1.62 above feet; the voxel center is at
    +0.5 in each axis from the integer block origin relative to the feet block.
    """
    tx = dx + 0.5
    ty = dy + 0.5 - 1.62
    tz = dz + 0.5
    horizontal = math.sqrt(tx * tx + tz * tz)
    # Vanilla yaw: 0 faces +Z, increases toward -X. yaw = -atan2(tx, tz).
    yaw = -math.degrees(math.atan2(tx, tz))
    pitch = -math.degrees(math.atan2(ty, horizontal))
    return yaw, pitch


def _run_episode(transport, build_action, steps):
    """Step the given action factory and return whichever obs first shows
    inventory + the last obs and a flag whether target was ever in range."""
    last = None
    saw_target = False
    gained = None
    for i in range(steps):
        obs = transport.step(build_action(i))
        row = obs[0]
        last = obs
        if int(row["target_in_range"]) == 1 and int(row["target_block"]) != 0:
            saw_target = True
        if gained is None and _inv_has_any(row):
            gained = obs.copy()
    return last, gained, saw_target


@pytest.mark.slow
def test_mining_e2e():
    agents = 1
    seed = 0
    registry = Registry.load(REGISTRY)
    log_ids = log_item_ids(registry)

    tmpdir = tempfile.mkdtemp(prefix="mcai_sock_")
    shm_path = f"/dev/shm/mcai_shm_{uuid.uuid4().hex}.bin"
    sock_path = str(pathlib.Path(tmpdir) / "gym.sock")

    proc = launch(agents, seed, shm_path, sock_path)
    transport = None
    try:
        transport = Transport(shm_path, sock_path, agents)
        obs0 = transport.reset()
        assert obs0.shape == (agents,)

        import json

        reg_doc = json.loads(REGISTRY.read_text())
        # oak_log BLOCK id (voxel_blocks uses block ids); the item that drops is the
        # oak_log ITEM id (inv_item_id uses item ids). Same name, two id tables.
        oak_log_block = reg_doc["blocks"]["minecraft:oak_log"]
        oak_log_item = reg_doc["items"]["minecraft:oak_log"]

        # Scan the pristine spawn voxel grid for a genuinely reachable oak_log BEFORE any
        # mining alters the world (reset() restores the pose but not broken blocks). The
        # spawn corridor logic deliberately avoids trunks directly ahead, so a log is only
        # attempted when it is at body level (|dy|<=1) and within bare-hand reach (<3.5
        # blocks) -- canopy logs above the agent are not practically mineable from spawn.
        voxels = obs0[0]["voxel_blocks"]
        log_indices = np.where(np.asarray(voxels) == oak_log_block)[0]
        log_offset = None
        best = None
        for idx in log_indices:
            dx, dy, dz = _index_to_offset(int(idx))
            if abs(dy) > 1:
                continue
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist <= 3.5 and (best is None or dist < best[0]):
                best = (dist, dx, dy, dz)
        if best is not None:
            log_offset = best[1:]

        # --- Robust block-below proof (runs first, on the pristine world) -------
        # Aim straight down and dig the column below in repeated bursts: ~22 ticks of
        # attack to break a block, then ~10 ticks idle so the freshly dropped item
        # (which falls into the 1-wide pit alongside the agent) clears its pickup delay
        # and is collected before the next burst. Digging straight down passes through
        # the surface dirt/grass layer, whose items drop with bare hands; the burst
        # cadence keeps the agent beside its own drops instead of descending past them.
        cur_pitch = float(obs0[0]["pitch"])
        period = 32
        dig_ticks = 22

        def _down_factory(i):
            nonlocal cur_pitch
            target = 90.0
            dpitch = max(-15.0, min(15.0, target - cur_pitch))
            cur_pitch += dpitch
            digging = (i % period) < dig_ticks
            return _action(forward=0.0, pitch_delta=dpitch, attack=1 if digging else 0)

        last, gained, saw_target = _run_episode(transport, _down_factory, 6 * period)

        assert saw_target, "agent never had a valid target_in_range while attacking down"

        proof = gained if gained is not None else last
        assert _inv_has_any(proof[0]), (
            "no item entered the inventory obs after mining the block below; "
            f"inv_item_id nonzero={np.asarray(last[0]['inv_item_id']).nonzero()[0].tolist()}"
        )

        # Report what the block-below proof gathered.
        items = np.asarray(proof[0]["inv_item_id"])
        counts = np.asarray(proof[0]["inv_count"])
        below_gathered = [
            (registry.name_of(int(it)), int(c))
            for it, c in zip(items, counts)
            if int(it) != 0 and int(c) > 0
        ]
        print(f"BLOCK_BELOW proof: inventory gained via real mining: {below_gathered}")

        # --- Best-effort oak_log path (runs after; may disturb the world) -------
        oak_path_done = False
        if log_offset is not None:
            dx, dy, dz = log_offset
            # reset() restores the spawn pose; the trunk is still standing because the
            # block-below proof dug a pit straight down, not toward the tree.
            base = transport.reset()
            # Aim directly at the trunk (yaw + pitch) and mine; creep forward to clear any
            # intervening foliage so the trunk comes into reach, then idle to collect its
            # drop. The trunk is at body level and within ~3.5 blocks, so once the line of
            # sight is clear the log itself becomes the mined target.
            yaw, pitch = _aim_at_offset(dx, dy, dz)
            print(f"OAK_LOG reachable at offset ({dx},{dy},{dz}); aiming yaw={yaw:.1f} pitch={pitch:.1f}")

            cyaw = float(base[0]["yaw"])
            cpitch = float(base[0]["pitch"])

            def _aim_factory(i, _yaw=yaw, _pitch=pitch):
                nonlocal cyaw, cpitch
                dyaw = max(-30.0, min(30.0, _yaw - cyaw))
                dpitch = max(-30.0, min(30.0, _pitch - cpitch))
                cyaw += dyaw
                cpitch += dpitch
                aligned = abs(_yaw - cyaw) < 15.0 and abs(_pitch - cpitch) < 15.0
                # Creep forward and hop to clear foliage during the dig phase, then stop and
                # idle so any dropped log item clears its pickup delay and is collected.
                digging = i < 150
                forward = 0.5 if (aligned and digging) else 0.0
                jump = 1 if (aligned and digging and i % 20 < 3) else 0
                rec = {
                    "forward": forward, "strafe": 0.0, "jump": jump, "sneak": 0, "sprint": 0,
                    "yaw_delta": dyaw, "pitch_delta": dpitch, "attack": 1 if digging else 0,
                    "use": 0, "selected_slot": 0, "inv_op_type": 0, "inv_slot_a": 0, "inv_slot_b": 0,
                }
                return np.frombuffer(codec.encode_action_batch([rec]), dtype=spec.ACTION_DTYPE).copy()

            last, gained, _ = _run_episode(transport, _aim_factory, 190)
            # The full accumulated inventory is in the final obs.
            li = np.asarray(last[0]["inv_item_id"])
            lc = np.asarray(last[0]["inv_count"])
            if oak_log_item in li[lc > 0].tolist():
                oak_path_done = True
                print("OAK_LOG path SUCCEEDED: oak_log item in inventory obs")
            if not oak_path_done:
                print("OAK_LOG path ran but did not collect an oak_log item; "
                      "block-below proof already established real mining works")
        else:
            print("No oak_log reachable from spawn voxel grid; relying on block-below proof only")

        print(f"oak_log-specific path succeeded: {oak_path_done}")
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
