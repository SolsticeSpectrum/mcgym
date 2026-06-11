import pathlib

import yaml

SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[2] / "schema" / "mcgym_schema.yaml"


def test_schema_yaml_loads_and_has_expected_top_keys():
    doc = yaml.safe_load(SCHEMA_PATH.read_text())
    assert doc["schema_version"] == 2
    assert set(doc.keys()) == {"schema_version", "params", "observation", "action"}
    assert doc["params"]["voxel_radius"] == 8
    assert doc["params"]["voxel_far_stride"] == 4
    assert doc["params"]["max_entities"] == 16
    assert doc["params"]["inventory_slots"] == 41


def test_voxel_field_matches_radius():
    doc = yaml.safe_load(SCHEMA_PATH.read_text())
    radius = doc["params"]["voxel_radius"]
    edge = 2 * radius + 1
    
    # near and far shells are both edge**3 grids of block ids
    for name in ("voxel_blocks", "voxel_far"):
        voxel = next(f for f in doc["observation"] if f["name"] == name)
        assert voxel["shape"] == [edge**3]
