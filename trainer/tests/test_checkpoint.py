"""Checkpoint save/load + keep-N retention (fast, CPU)."""
from __future__ import annotations

import json
import pathlib
import tempfile

import torch
import torch.nn as nn

from mcai_train import checkpoint


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        model = _Tiny()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        meta = {"cumulative_timesteps": 1024, "schema_version": 0, "hyperparams": {}}
        checkpoint.save(d, model, opt, meta)

        orig = model.fc.weight.detach().clone()
        with torch.no_grad():
            model.fc.weight.add_(1.0)
        assert not torch.equal(model.fc.weight, orig)

        loaded_meta = checkpoint.load_latest(d, model, opt)
        assert torch.equal(model.fc.weight, orig)
        assert loaded_meta["cumulative_timesteps"] == 1024


def test_latest_symlink():
    with tempfile.TemporaryDirectory() as d:
        model = _Tiny()
        opt = torch.optim.Adam(model.parameters())
        checkpoint.save(d, model, opt, {"cumulative_timesteps": 10})
        checkpoint.save(d, model, opt, {"cumulative_timesteps": 20})
        latest = pathlib.Path(d) / "latest"
        assert latest.is_symlink()
        meta = json.loads((latest / "meta.json").read_text())
        assert meta["cumulative_timesteps"] == 20


def test_keep_n_deletes_oldest():
    with tempfile.TemporaryDirectory() as d:
        model = _Tiny()
        opt = torch.optim.Adam(model.parameters())
        for step in (100, 200, 300, 400):
            checkpoint.save(d, model, opt, {"cumulative_timesteps": step}, keep_n=2)
        dirs = sorted(p.name for p in pathlib.Path(d).iterdir() if p.name.startswith("step_") and not p.is_symlink())
        assert dirs == ["step_300", "step_400"]


def test_load_latest_missing_returns_none():
    with tempfile.TemporaryDirectory() as d:
        model = _Tiny()
        assert checkpoint.load_latest(d, model, None) is None
