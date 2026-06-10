import pathlib

import pytest

from mcgym.schema import registry

SAMPLE = pathlib.Path(__file__).resolve().parents[2] / "schema" / "registry.sample.json"


def test_load_and_lookup():
    reg = registry.Registry.load(SAMPLE)

    assert reg.block_id_of("minecraft:air") == 0
    assert reg.block_ids()["minecraft:oak_log"] == reg.block_id_of("minecraft:oak_log")
    assert len(reg.block_ids()) >= 3


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        registry.Registry.load(pathlib.Path("/no/such/registry.json"))


def test_unknown_name_raises():
    reg = registry.Registry.load(SAMPLE)
    with pytest.raises(KeyError):
        reg.block_id_of("minecraft:does_not_exist")
