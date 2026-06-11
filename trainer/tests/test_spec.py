import numpy as np

from mcgym.schema import spec


def test_obs_dtype_is_packed_little_endian():
    d = spec.OBS_DTYPE
    
    # packed means itemsize == sum of field nbytes, no alignment padding
    total = sum(d.fields[n][0].itemsize for n in d.names)
    
    assert d.itemsize == total
    assert d.fields["voxel_blocks"][0].shape == (4913,)
    assert d.fields["voxel_blocks"][0].base == np.dtype("<i4")


def test_obs_itemsize_matches_hand_count():
    # 4 + 8 + 4 + 12 + 12 + 4 + 4 + 1 + 4 + 4 + 1  = 58    header+self
    # + 4913*4                                     = 19652 voxel_blocks near
    # + 4913*4                                     = 19652 voxel_far
    # + 4913*6                                     = 29478 bounds
    # + 4 + 1 + 4 + 1                              = 10    target
    # + 16*4 + 16*12 + 16*12 + 16*4 + 16*4 + 16*1  = 592   entities
    # + 41*4 + 41*1                                = 205   inventory
    assert spec.OBS_DTYPE.itemsize == 58 + 19652 + 19652 + 29478 + 10 + 592 + 205


def test_action_itemsize_matches_hand_count():
    # 4 + 4 + 1 + 1 + 1 + 4 + 4 + 1 + 1 + 1 + 1 + 2 + 2 = 27
    assert spec.ACTION_DTYPE.itemsize == 27


def test_params_exposed():
    assert spec.SCHEMA_VERSION == 2
    assert spec.PARAMS["voxel_radius"] == 8
    assert spec.VOXEL_EDGE == 17
    assert spec.VOXEL_FAR_STRIDE == 4
