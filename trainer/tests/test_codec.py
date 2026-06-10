import numpy as np

from mcgym.schema import codec, spec


def _sample_obs():
    return {
        "schema_version":  0,
        "tick":            1234,
        "agent_id":        3,
        "pos":             [130000.0, 80.0, 0.0],
        "vel":             [0.13, 0.0, 0.0],
        "yaw":             -90.0,
        "pitch":           0.0,
        "on_ground":       1,
        "health":          20.0,
        "food":            20.0,
        "selected_slot":   2,
        "voxel_blocks":    np.arange(spec.VOXEL_EDGE**3, dtype="<i4"),
        "target_block":    42,
        "target_face":     1,
        "target_distance": 3.5,
        "target_in_range": 1,
    }


def test_obs_roundtrip():
    rec = _sample_obs()
    buf = codec.encode_obs(rec)
    assert len(buf) == spec.OBS_NBYTES
    
    out = codec.decode_obs(buf)
    assert out["tick"]     == 1234
    assert out["agent_id"] == 3
    assert np.allclose(out["pos"], [130000.0, 80.0, 0.0])
    
    assert out["target_block"] == 42
    assert np.array_equal(out["voxel_blocks"], np.arange(spec.VOXEL_EDGE**3))


def test_action_roundtrip():
    act = {
        "forward": 1.0, "strafe": -1.0, "jump": 1, "sneak": 0, "sprint": 1,
        "yaw_delta": 5.0, "pitch_delta": -2.5, "attack": 1, "use": 0,
        "selected_slot": 3, "inv_op_type": 0, "inv_slot_a": 9, "inv_slot_b": 36,
    }
    
    buf = codec.encode_action(act)
    assert len(buf) == spec.ACTION_NBYTES
    
    out = codec.decode_action(buf)
    assert out["forward"]    == 1.0
    assert out["sprint"]     == 1
    assert out["inv_slot_b"] == 36


def test_decode_obs_batch_zero_copy_shape():
    recs = b"".join(codec.encode_obs(_sample_obs()) for _ in range(4))
    batch = codec.decode_obs_batch(recs, 4)
    assert batch.shape == (4,)
    assert batch["agent_id"].tolist() == [3, 3, 3, 3]
