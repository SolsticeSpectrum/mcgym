"""parse schema/mcai_schema.yaml into packed little endian dtypes, the canonical wire format"""
import pathlib

import numpy as np
import yaml

_SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[3] / "schema" / "mcai_schema.yaml"

_DTYPE_MAP = {
    "i16": "<i2",
    "i32": "<i4",
    "i64": "<i8",
    "f32": "<f4",
    "u8":  "u1",
}


def _build_dtype(fields):
    names, formats = [], []
    for f in fields:
        base  = _DTYPE_MAP[f["dtype"]]
        shape = f.get("shape")
        formats.append((base, tuple(shape)) if shape else base)
        names.append(f["name"])

    # align false keeps fields tightly packed
    return np.dtype({"names": names, "formats": formats}, align=False)


_DOC = yaml.safe_load(_SCHEMA_PATH.read_text())

SCHEMA_VERSION   = _DOC["schema_version"]
PARAMS           = _DOC["params"]
VOXEL_EDGE       = 2 * PARAMS["voxel_radius"] + 1
VOXEL_FAR_STRIDE = PARAMS["voxel_far_stride"]
OBS_DTYPE        = _build_dtype(_DOC["observation"])
ACTION_DTYPE     = _build_dtype(_DOC["action"])
OBS_NBYTES       = OBS_DTYPE.itemsize
ACTION_NBYTES    = ACTION_DTYPE.itemsize
