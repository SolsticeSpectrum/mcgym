# MCAI Schema Implementation Plan (Plan 1 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the single canonical, versioned observation/action schema and its Python codec, with a frozen cross-language golden contract that the Java gym (Plan 2) and Fabric mod (later) must reproduce byte-for-byte.

**Architecture:** The schema is one language-neutral YAML file (`schema/mcai_schema.yaml`) describing an ordered, packed, little-endian binary record for observations and actions. A Python module parses it into a packed NumPy structured dtype — that dtype *is* the wire format and the shared-memory layout. A committed golden fixture (known field values → known bytes) is the contract every language implementation is tested against.

**Tech Stack:** Python 3.11+, NumPy, PyYAML, pytest.

---

### Task 0: Python package scaffold

**Files:**
- Create: `trainer/pyproject.toml`
- Create: `trainer/mcai_train/__init__.py`
- Create: `trainer/tests/__init__.py`
- Create: `trainer/tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

`trainer/tests/test_smoke.py`:
```python
import mcai_train


def test_package_imports():
    assert mcai_train.__version__ == "0.0.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd trainer && python -m pytest tests/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mcai_train'`

- [ ] **Step 3: Write minimal implementation**

`trainer/pyproject.toml`:
```toml
[project]
name = "mcai-train"
version = "0.0.0"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "pyyaml>=6.0"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["mcai_train*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`trainer/mcai_train/__init__.py`:
```python
__version__ = "0.0.0"
```

`trainer/tests/__init__.py`: (empty file)

- [ ] **Step 4: Install and run test to verify it passes**

Run: `cd trainer && pip install -e ".[dev]" && python -m pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trainer/pyproject.toml trainer/mcai_train/__init__.py trainer/tests/__init__.py trainer/tests/test_smoke.py
git commit -m "feat(schema): scaffold mcai_train python package"
```

---

### Task 1: Schema YAML spec file

**Files:**
- Create: `schema/mcai_schema.yaml`
- Test: `trainer/tests/test_schema_yaml.py`

This file is the single source of truth. `shape` absent means a scalar. `voxel_blocks`
length is `(2*voxel_radius+1)**3 = 17**3 = 4913`. `hidden_block_id` is the `legit`-mode
sentinel for fully-enclosed blocks. dtypes: `i32 i64 f32 u8 i16`.

- [ ] **Step 1: Write the failing test**

`trainer/tests/test_schema_yaml.py`:
```python
import pathlib
import yaml

SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[2] / "schema" / "mcai_schema.yaml"


def test_schema_yaml_loads_and_has_expected_top_keys():
    doc = yaml.safe_load(SCHEMA_PATH.read_text())
    assert doc["schema_version"] == 0
    assert set(doc.keys()) == {"schema_version", "params", "observation", "action"}
    assert doc["params"]["voxel_radius"] == 8
    assert doc["params"]["max_entities"] == 16
    assert doc["params"]["inventory_slots"] == 41


def test_voxel_field_matches_radius():
    doc = yaml.safe_load(SCHEMA_PATH.read_text())
    radius = doc["params"]["voxel_radius"]
    edge = 2 * radius + 1
    voxel = next(f for f in doc["observation"] if f["name"] == "voxel_blocks")
    assert voxel["shape"] == [edge**3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd trainer && python -m pytest tests/test_schema_yaml.py -v`
Expected: FAIL — `FileNotFoundError` for `schema/mcai_schema.yaml`

- [ ] **Step 3: Write minimal implementation**

`schema/mcai_schema.yaml`:
```yaml
schema_version: 0

params:
  voxel_radius: 8          # voxel edge = 2*r+1 = 17, cells = 4913
  max_entities: 16
  inventory_slots: 41      # 36 main + 4 armor + 1 offhand
  hidden_block_id: -1      # legit-mode sentinel for enclosed blocks

observation:
  - {name: schema_version, dtype: i32}
  - {name: tick, dtype: i64}
  - {name: agent_id, dtype: i32}
  - {name: pos, dtype: f32, shape: [3]}
  - {name: vel, dtype: f32, shape: [3]}
  - {name: yaw, dtype: f32}
  - {name: pitch, dtype: f32}
  - {name: on_ground, dtype: u8}
  - {name: health, dtype: f32}
  - {name: food, dtype: f32}
  - {name: selected_slot, dtype: u8}
  - {name: voxel_blocks, dtype: i32, shape: [4913]}
  - {name: target_block, dtype: i32}
  - {name: target_face, dtype: u8}
  - {name: target_distance, dtype: f32}
  - {name: target_in_range, dtype: u8}
  - {name: entity_type_id, dtype: i32, shape: [16]}
  - {name: entity_rel_pos, dtype: f32, shape: [16, 3]}
  - {name: entity_vel, dtype: f32, shape: [16, 3]}
  - {name: entity_yaw, dtype: f32, shape: [16]}
  - {name: entity_health, dtype: f32, shape: [16]}
  - {name: entity_flags, dtype: u8, shape: [16]}
  - {name: inv_item_id, dtype: i32, shape: [41]}
  - {name: inv_count, dtype: u8, shape: [41]}

action:
  - {name: forward, dtype: f32}
  - {name: strafe, dtype: f32}
  - {name: jump, dtype: u8}
  - {name: sneak, dtype: u8}
  - {name: sprint, dtype: u8}
  - {name: yaw_delta, dtype: f32}
  - {name: pitch_delta, dtype: f32}
  - {name: attack, dtype: u8}
  - {name: use, dtype: u8}
  - {name: selected_slot, dtype: u8}
  - {name: inv_op_type, dtype: u8}
  - {name: inv_slot_a, dtype: i16}
  - {name: inv_slot_b, dtype: i16}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd trainer && python -m pytest tests/test_schema_yaml.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add schema/mcai_schema.yaml trainer/tests/test_schema_yaml.py
git commit -m "feat(schema): add canonical mcai_schema.yaml spec"
```

---

### Task 2: Spec loader → packed NumPy dtype

**Files:**
- Create: `trainer/mcai_train/schema/__init__.py`
- Create: `trainer/mcai_train/schema/spec.py`
- Test: `trainer/tests/test_spec.py`

`spec.py` parses the YAML into two packed (no padding), little-endian structured dtypes.
`align=False` with explicit `names`/`formats` produces a tightly packed record whose
`itemsize` equals the sum of field byte sizes — this is the exact wire layout Java must
match.

- [ ] **Step 1: Write the failing test**

`trainer/tests/test_spec.py`:
```python
import numpy as np

from mcai_train.schema import spec


def test_obs_dtype_is_packed_little_endian():
    d = spec.OBS_DTYPE
    # Packed: itemsize == sum of field nbytes (no alignment padding).
    total = sum(d.fields[n][0].itemsize for n in d.names)
    assert d.itemsize == total
    # voxel_blocks is int32 with 4913 cells.
    assert d.fields["voxel_blocks"][0].shape == (4913,)
    assert d.fields["voxel_blocks"][0].base == np.dtype("<i4")


def test_obs_itemsize_matches_hand_count():
    # 4 + 8 + 4 + 12 + 12 + 4 + 4 + 1 + 4 + 4 + 1            = 58  (header+self)
    # + 4913*4                                               = 19652 (voxel)
    # + 4 + 1 + 4 + 1                                        = 10  (target)
    # + 16*4 + 16*12 + 16*12 + 16*4 + 16*4 + 16*1            = 592  (entities)
    # + 41*4 + 41*1                                          = 205  (inventory)
    assert spec.OBS_DTYPE.itemsize == 58 + 19652 + 10 + 592 + 205


def test_action_itemsize_matches_hand_count():
    # 4 + 4 + 1 + 1 + 1 + 4 + 4 + 1 + 1 + 1 + 1 + 2 + 2 = 27
    assert spec.ACTION_DTYPE.itemsize == 27


def test_params_exposed():
    assert spec.SCHEMA_VERSION == 0
    assert spec.PARAMS["voxel_radius"] == 8
    assert spec.VOXEL_EDGE == 17
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd trainer && python -m pytest tests/test_spec.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mcai_train.schema'`

- [ ] **Step 3: Write minimal implementation**

`trainer/mcai_train/schema/__init__.py`:
```python
from . import spec  # noqa: F401
```

`trainer/mcai_train/schema/spec.py`:
```python
"""Parses schema/mcai_schema.yaml into packed little-endian NumPy dtypes.

The dtypes produced here ARE the canonical wire format and shared-memory layout.
Any other language implementation (Java gym, Fabric mod) must reproduce these bytes.
"""
import pathlib

import numpy as np
import yaml

_SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[3] / "schema" / "mcai_schema.yaml"

_DTYPE_MAP = {
    "i16": "<i2",
    "i32": "<i4",
    "i64": "<i8",
    "f32": "<f4",
    "u8": "u1",
}


def _build_dtype(fields):
    names, formats = [], []
    for f in fields:
        base = _DTYPE_MAP[f["dtype"]]
        shape = f.get("shape")
        formats.append((base, tuple(shape)) if shape else base)
        names.append(f["name"])
    # align=False -> tightly packed, no padding between fields.
    return np.dtype({"names": names, "formats": formats}, align=False)


_DOC = yaml.safe_load(_SCHEMA_PATH.read_text())

SCHEMA_VERSION = _DOC["schema_version"]
PARAMS = _DOC["params"]
VOXEL_EDGE = 2 * PARAMS["voxel_radius"] + 1
OBS_DTYPE = _build_dtype(_DOC["observation"])
ACTION_DTYPE = _build_dtype(_DOC["action"])
OBS_NBYTES = OBS_DTYPE.itemsize
ACTION_NBYTES = ACTION_DTYPE.itemsize
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd trainer && python -m pytest tests/test_spec.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add trainer/mcai_train/schema/__init__.py trainer/mcai_train/schema/spec.py trainer/tests/test_spec.py
git commit -m "feat(schema): parse yaml into packed numpy dtypes"
```

---

### Task 3: Observation & action codec

**Files:**
- Create: `trainer/mcai_train/schema/codec.py`
- Modify: `trainer/mcai_train/schema/__init__.py`
- Test: `trainer/tests/test_codec.py`

The codec converts between Python dicts and packed bytes, and decodes whole batches of
N records straight out of a shared-memory buffer as a zero-copy structured array view.

- [ ] **Step 1: Write the failing test**

`trainer/tests/test_codec.py`:
```python
import numpy as np

from mcai_train.schema import codec, spec


def _sample_obs():
    return {
        "schema_version": 0,
        "tick": 1234,
        "agent_id": 3,
        "pos": [130000.0, 80.0, 0.0],
        "vel": [0.13, 0.0, 0.0],
        "yaw": -90.0,
        "pitch": 0.0,
        "on_ground": 1,
        "health": 20.0,
        "food": 20.0,
        "selected_slot": 2,
        "voxel_blocks": np.arange(spec.VOXEL_EDGE**3, dtype="<i4"),
        "target_block": 42,
        "target_face": 1,
        "target_distance": 3.5,
        "target_in_range": 1,
    }


def test_obs_roundtrip():
    rec = _sample_obs()
    buf = codec.encode_obs(rec)
    assert len(buf) == spec.OBS_NBYTES
    out = codec.decode_obs(buf)
    assert out["tick"] == 1234
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
    assert out["forward"] == 1.0
    assert out["sprint"] == 1
    assert out["inv_slot_b"] == 36


def test_decode_obs_batch_zero_copy_shape():
    recs = b"".join(codec.encode_obs(_sample_obs()) for _ in range(4))
    batch = codec.decode_obs_batch(recs, 4)
    assert batch.shape == (4,)
    assert batch["agent_id"].tolist() == [3, 3, 3, 3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd trainer && python -m pytest tests/test_codec.py -v`
Expected: FAIL — `AttributeError: module 'mcai_train.schema.codec'` / ImportError

- [ ] **Step 3: Write minimal implementation**

`trainer/mcai_train/schema/codec.py`:
```python
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
```

Modify `trainer/mcai_train/schema/__init__.py`:
```python
from . import codec, spec  # noqa: F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd trainer && python -m pytest tests/test_codec.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add trainer/mcai_train/schema/codec.py trainer/mcai_train/schema/__init__.py trainer/tests/test_codec.py
git commit -m "feat(schema): add obs/action codec with batch decode"
```

---

### Task 4: Frozen cross-language golden contract

**Files:**
- Create: `trainer/mcai_train/schema/golden.py`
- Create: `schema/fixtures/golden_obs.json`
- Create: `schema/fixtures/golden_obs.bin` (generated, committed)
- Create: `schema/fixtures/golden_action.json`
- Create: `schema/fixtures/golden_action.bin` (generated, committed)
- Test: `trainer/tests/test_golden.py`

The golden `.bin` files are the immutable contract: Plan 2's Java codec and the Fabric mod
must emit these exact bytes for the values in the matching `.json`. `golden.py` builds the
records deterministically so the JSON and the bytes can never drift. Once committed, the
`.bin` files change ONLY with an intentional `schema_version` bump.

- [ ] **Step 1: Write the failing test**

`trainer/tests/test_golden.py`:
```python
import json
import pathlib

from mcai_train.schema import codec, golden, spec

FIX = pathlib.Path(__file__).resolve().parents[2] / "schema" / "fixtures"


def test_golden_obs_bytes_match_committed_file():
    rec = golden.golden_obs_record()
    produced = codec.encode_obs(rec)
    committed = (FIX / "golden_obs.bin").read_bytes()
    assert produced == committed, "Obs byte layout changed without a schema_version bump"
    assert len(committed) == spec.OBS_NBYTES


def test_golden_action_bytes_match_committed_file():
    rec = golden.golden_action_record()
    produced = codec.encode_action(rec)
    committed = (FIX / "golden_action.bin").read_bytes()
    assert produced == committed
    assert len(committed) == spec.ACTION_NBYTES


def test_golden_json_matches_records():
    obs_json = json.loads((FIX / "golden_obs.json").read_text())
    assert obs_json["tick"] == golden.golden_obs_record()["tick"]
    act_json = json.loads((FIX / "golden_action.json").read_text())
    assert act_json["forward"] == golden.golden_action_record()["forward"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd trainer && python -m pytest tests/test_golden.py -v`
Expected: FAIL — `ModuleNotFoundError: mcai_train.schema.golden`

- [ ] **Step 3: Write the record builder, then generate the committed fixtures**

`trainer/mcai_train/schema/golden.py`:
```python
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
```

Generate and commit the fixtures (one-time; rerun only on an intentional schema bump):
```bash
cd trainer && python - <<'PY'
import json, pathlib
from mcai_train.schema import codec, golden
fix = pathlib.Path("..") / "schema" / "fixtures"
fix.mkdir(parents=True, exist_ok=True)
obs = golden.golden_obs_record()
act = golden.golden_action_record()
(fix / "golden_obs.bin").write_bytes(codec.encode_obs(obs))
(fix / "golden_action.bin").write_bytes(codec.encode_action(act))
# JSON: scalars only, for the human-readable contract + Java test inputs.
(fix / "golden_obs.json").write_text(json.dumps(
    {k: v for k, v in obs.items() if not hasattr(v, "shape")}, indent=2))
(fix / "golden_action.json").write_text(json.dumps(act, indent=2))
print("wrote fixtures")
PY
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd trainer && python -m pytest tests/test_golden.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add trainer/mcai_train/schema/golden.py schema/fixtures/ trainer/tests/test_golden.py
git commit -m "feat(schema): freeze cross-language golden byte contract"
```

---

### Task 5: Block/item registry table

**Files:**
- Create: `trainer/mcai_train/schema/registry.py`
- Create: `schema/registry.sample.json`
- Test: `trainer/tests/test_registry.py`

Maps stable integer IDs ⇄ names (e.g. `42 ⇄ "minecraft:oak_log"`). The full table is
dumped from `BuiltInRegistries` by the Java gym in Plan 2; this task defines the loader,
the JSON format, and a small sample so the trainer is independently testable. The loader
fails loudly on a missing file (no silent fallback).

- [ ] **Step 1: Write the failing test**

`trainer/tests/test_registry.py`:
```python
import pathlib

import pytest

from mcai_train.schema import registry

SAMPLE = pathlib.Path(__file__).resolve().parents[2] / "schema" / "registry.sample.json"


def test_load_and_lookup_roundtrip():
    reg = registry.Registry.load(SAMPLE)
    assert reg.name_of(reg.id_of("minecraft:oak_log")) == "minecraft:oak_log"
    assert reg.id_of("minecraft:air") == 0
    assert reg.size >= 3


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        registry.Registry.load(pathlib.Path("/no/such/registry.json"))


def test_unknown_name_raises():
    reg = registry.Registry.load(SAMPLE)
    with pytest.raises(KeyError):
        reg.id_of("minecraft:does_not_exist")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd trainer && python -m pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: mcai_train.schema.registry`

- [ ] **Step 3: Write minimal implementation**

`schema/registry.sample.json`:
```json
{
  "schema_version": 0,
  "blocks": {
    "minecraft:air": 0,
    "minecraft:stone": 1,
    "minecraft:dirt": 2,
    "minecraft:oak_log": 42
  },
  "items": {
    "minecraft:air": 0,
    "minecraft:oak_log": 42,
    "minecraft:oak_planks": 43
  }
}
```

`trainer/mcai_train/schema/registry.py`:
```python
"""ID <-> name registry for blocks and items (single shared vocabulary)."""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass


@dataclass
class Registry:
    _name_to_id: dict[str, int]
    _id_to_name: dict[int, str]

    @classmethod
    def load(cls, path: pathlib.Path) -> "Registry":
        if not path.exists():
            raise FileNotFoundError(f"registry not found: {path}")
        doc = json.loads(path.read_text())
        merged: dict[str, int] = {}
        merged.update(doc.get("blocks", {}))
        merged.update(doc.get("items", {}))
        id_to_name = {v: k for k, v in merged.items()}
        return cls(merged, id_to_name)

    def id_of(self, name: str) -> int:
        return self._name_to_id[name]

    def name_of(self, ident: int) -> str:
        return self._id_to_name[ident]

    @property
    def size(self) -> int:
        return len(self._name_to_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd trainer && python -m pytest tests/test_registry.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add trainer/mcai_train/schema/registry.py schema/registry.sample.json trainer/tests/test_registry.py
git commit -m "feat(schema): add block/item id registry loader"
```

---

### Task 6: Full suite green + README

**Files:**
- Create: `schema/README.md`
- Test: (run the whole suite)

- [ ] **Step 1: Run the entire test suite**

Run: `cd trainer && python -m pytest -v`
Expected: PASS — all tests from Tasks 0–5 green.

- [ ] **Step 2: Write the schema README**

`schema/README.md`:
```markdown
# MCAI Schema

`mcai_schema.yaml` is the single source of truth for the observation/action binary
layout shared by the Java gym (training) and the Fabric mod (inference).

- Layout: ordered, packed (no padding), little-endian. Python parses it into a NumPy
  structured dtype in `trainer/mcai_train/schema/spec.py`; that dtype is the wire format
  and the shared-memory layout.
- Golden contract: `fixtures/golden_*.json` (values) + `fixtures/golden_*.bin` (bytes).
  Every language implementation MUST reproduce the `.bin` bytes for the `.json` values.
  Regenerate the `.bin` files ONLY when intentionally bumping `schema_version`.
- Registry: `registry.sample.json` is a small fixture; the full block/item id table is
  dumped from `BuiltInRegistries` by the gym (Plan 2).

`block_visibility` (`xray` vs `legit`) is applied server-side at emit time; `legit`
replaces enclosed blocks with `params.hidden_block_id`.
```

- [ ] **Step 3: Commit**

```bash
git add schema/README.md
git commit -m "docs(schema): document schema layout and golden contract"
```

---

## Self-Review

**Spec coverage (against design Section 3 + 12):**
- Versioned schema, `schema_version` header, assert-on-mismatch → Task 1/2 (version in record + params). ✓
- Self / voxel / target / entities / inventory obs fields → Task 1 YAML, Task 2 dtype. ✓
- `block_visibility` toggle (`xray`/`legit`) + `hidden_block_id` sentinel → encoded as a param + documented (the server applies it at emit; codec just carries ids). ✓
- Action incl. vanilla container-click op → Task 1 (`inv_op_type/slot_a/slot_b`). ✓
- Block/item ID registry from `BuiltInRegistries` → Task 5 (loader + format; full dump deferred to Plan 2 as designed). ✓
- Byte-identical cross-language contract → Task 4 golden `.bin`. ✓
- Defaults R=8 (4913), K=16, 41 inv slots → Task 1. ✓
- Zero-copy shared-memory batch decode (needed by Plan 3 transport) → Task 3 `decode_obs_batch`. ✓

**Placeholder scan:** No TBD/TODO; every code step is complete and runnable. The only
deferred item (full registry dump) is an explicit Plan 2 dependency, not a placeholder —
Task 5 ships a working loader + sample so this plan stands alone. ✓

**Type consistency:** `OBS_DTYPE`/`ACTION_DTYPE`/`OBS_NBYTES`/`ACTION_NBYTES`/`VOXEL_EDGE`/
`SCHEMA_VERSION`/`PARAMS` defined in Task 2 and used identically in Tasks 3–4.
`encode_obs/decode_obs/decode_obs_batch/encode_action/decode_action` names consistent
across Tasks 3–4. `Registry.load/id_of/name_of/size` consistent in Task 5. ✓
