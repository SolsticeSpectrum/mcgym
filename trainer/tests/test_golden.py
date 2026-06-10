import json
import pathlib

from mcgym.schema import codec, golden, spec

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
