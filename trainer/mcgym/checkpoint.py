"""checkpoint save/load with keep n retention and onnx export"""
from __future__ import annotations

import json
import pathlib

import torch


def _checkpoint_dirs(root: pathlib.Path):
    dirs = [
        d
        for d in root.iterdir()
        if d.is_dir() and d.name.startswith("step_") and not d.is_symlink()
    ]
    return sorted(dirs, key=lambda d: int(d.name.split("_")[1]))


def save(directory, model, optimizer, meta: dict, keep_n: int = 5) -> pathlib.Path:
    """write step_<steps>/ with state dicts and meta, point latest at it"""
    root = pathlib.Path(directory)
    root.mkdir(parents=True, exist_ok=True)

    step = int(meta["steps"])
    ckpt = root / f"step_{step}"
    ckpt.mkdir(exist_ok=True)

    torch.save(model.state_dict(), ckpt / "model.pt")
    torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")
    (ckpt / "meta.json").write_text(json.dumps(meta, indent=2))

    latest = root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(ckpt.name)

    existing = _checkpoint_dirs(root)
    for old in existing[:-keep_n] if keep_n > 0 else []:
        for f in old.iterdir():
            f.unlink()
        old.rmdir()

    return ckpt


def load_latest(directory, model, optimizer) -> dict | None:
    """load newest checkpoint into model and optimizer, returns meta or None"""
    root   = pathlib.Path(directory)
    latest = root / "latest"
    if not latest.exists():
        dirs = _checkpoint_dirs(root) if root.exists() else []
        if not dirs:
            return None
        ckpt = dirs[-1]
    else:
        ckpt = latest.resolve()

    loc = next(model.parameters()).device
    model.load_state_dict(
        torch.load(ckpt / "model.pt", map_location=loc, weights_only=True)
    )
    
    if optimizer is not None and (ckpt / "optimizer.pt").exists():
        optimizer.load_state_dict(
            torch.load(ckpt / "optimizer.pt", map_location=loc, weights_only=True)
        )

    return json.loads((ckpt / "meta.json").read_text())


def export_onnx(directory, model) -> pathlib.Path:
    """export forward as (logits, value) onnx, returns the path"""
    import numpy as np

    from mcgym.models.policy import tensors
    from mcgym.schema import spec

    root = pathlib.Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    out = root / "model.onnx"

    device = next(model.parameters()).device
    dummy  = np.zeros(1, dtype=spec.OBS_DTYPE)
    t      = tensors(dummy, device)

    # onnx wants positional inputs, rewrap into the obs dict
    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, voxel, voxel_far, target_block, scalars, inv_item_id, inv_count):
            return self.m.forward({
                "voxel":        voxel,
                "voxel_far":    voxel_far,
                "target_block": target_block,
                "scalars":      scalars,
                "inv_item_id":  inv_item_id,
                "inv_count":    inv_count,
            })

    torch.onnx.export(
        Wrap(model),
        (
            t["voxel"],
            t["voxel_far"],
            t["target_block"],
            t["scalars"],
            t["inv_item_id"],
            t["inv_count"],
        ),
        str(out),
        input_names=["voxel", "voxel_far", "target_block", "scalars", "inv_item_id", "inv_count"],
        output_names=["logits", "value"],
        dynamic_axes={
            "voxel":        {0: "batch"},
            "voxel_far":    {0: "batch"},
            "target_block": {0: "batch"},
            "scalars":      {0: "batch"},
            "inv_item_id":  {0: "batch"},
            "inv_count":    {0: "batch"},
        },
        opset_version=17,
    )

    return out
