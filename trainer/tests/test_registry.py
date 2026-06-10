import pathlib

import pytest

from mcgym.schema import registry

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
