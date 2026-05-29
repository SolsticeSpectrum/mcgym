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
