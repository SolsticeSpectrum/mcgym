"""Deterministic golden records — the immutable cross-language byte contract."""
import numpy as np

from . import spec


def golden_obs_record() -> dict:
    edge3 = spec.VOXEL_EDGE**3
    return {
        "schema_version": spec.SCHEMA_VERSION,
        "tick": 7,
        "agent_id": 1,
        "pos": [10.5, 80.0, -3.25],
        "vel": [0.1, -0.05, 0.2],
        "yaw": 45.0,
        "pitch": -10.0,
        "on_ground": 1,
        "health": 20.0,
        "food": 18.0,
        "selected_slot": 4,
        # Deterministic, varied, bounded block ids.
        "voxel_blocks": (np.arange(edge3, dtype="<i4") % 97),
        "voxel_far": (np.arange(edge3, dtype="<i4") % 89),
        "target_block": 42,
        "target_face": 2,
        "target_distance": 2.75,
        "target_in_range": 1,
        "entity_type_id": np.arange(spec.PARAMS["max_entities"], dtype="<i4"),
        "entity_health": np.full(spec.PARAMS["max_entities"], 20.0, dtype="<f4"),
        "inv_item_id": np.arange(spec.PARAMS["inventory_slots"], dtype="<i4"),
        "inv_count": np.ones(spec.PARAMS["inventory_slots"], dtype="u1"),
    }


def golden_action_record() -> dict:
    return {
        "forward": 1.0, "strafe": 0.0, "jump": 0, "sneak": 0, "sprint": 1,
        "yaw_delta": 3.0, "pitch_delta": -1.0, "attack": 1, "use": 0,
        "selected_slot": 4, "inv_op_type": 0, "inv_slot_a": 0, "inv_slot_b": 0,
    }
