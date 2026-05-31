"""Checkpoint save/load with keep-N retention and best-effort ONNX export.

A checkpoint is a directory ``step_<cumulative_timesteps>/`` holding model and
optimizer state_dicts plus a meta.json. A ``latest`` symlink always points at
the newest checkpoint for easy resume.
"""
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
    root = pathlib.Path(directory)
    root.mkdir(parents=True, exist_ok=True)

    step = int(meta["cumulative_timesteps"])
    ckpt_dir = root / f"step_{step}"
    ckpt_dir.mkdir(exist_ok=True)

    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    latest = root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(ckpt_dir.name)

    existing = _checkpoint_dirs(root)
    for old in existing[:-keep_n] if keep_n > 0 else []:
        for f in old.iterdir():
            f.unlink()
        old.rmdir()

    return ckpt_dir


def load_latest(directory, model, optimizer) -> dict | None:
    root = pathlib.Path(directory)
    latest = root / "latest"
    if not latest.exists():
        dirs = _checkpoint_dirs(root) if root.exists() else []
        if not dirs:
            return None
        ckpt_dir = dirs[-1]
    else:
        ckpt_dir = latest.resolve()

    map_location = next(model.parameters()).device
    model.load_state_dict(
        torch.load(ckpt_dir / "model.pt", map_location=map_location, weights_only=True)
    )
    if optimizer is not None and (ckpt_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(
            torch.load(
                ckpt_dir / "optimizer.pt", map_location=map_location, weights_only=True
            )
        )
    return json.loads((ckpt_dir / "meta.json").read_text())


def export_onnx(directory, model) -> pathlib.Path | None:
    """Best-effort ONNX export of forward -> (logits, value).

    The 3D conv + embedding stack may not export cleanly on every opset; on
    failure we log a warning and return None rather than raising.
    """
    import numpy as np

    from mcai_train.schema import spec

    root = pathlib.Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    out = root / "model.onnx"

    device = next(model.parameters()).device
    dummy = np.zeros(1, dtype=spec.OBS_DTYPE)
    from mcai_train.models.policy import obs_to_tensors

    tensors = obs_to_tensors(dummy, device)

    class _Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, voxel, voxel_far, target_block, scalars, inv_item_id, inv_count):
            obs = {
                "voxel": voxel,
                "voxel_far": voxel_far,
                "target_block": target_block,
                "scalars": scalars,
                "inv_item_id": inv_item_id,
                "inv_count": inv_count,
            }
            return self.m.forward(obs)

    wrapper = _Wrapper(model)
    try:
        torch.onnx.export(
            wrapper,
            (
                tensors["voxel"],
                tensors["voxel_far"],
                tensors["target_block"],
                tensors["scalars"],
                tensors["inv_item_id"],
                tensors["inv_count"],
            ),
            str(out),
            input_names=["voxel", "voxel_far", "target_block", "scalars", "inv_item_id", "inv_count"],
            output_names=["logits", "value"],
            dynamic_axes={
                "voxel": {0: "batch"},
                "voxel_far": {0: "batch"},
                "target_block": {0: "batch"},
                "scalars": {0: "batch"},
                "inv_item_id": {0: "batch"},
                "inv_count": {0: "batch"},
            },
            opset_version=17,
        )
        return out
    except Exception as exc:  # noqa: BLE001 - export is best-effort
        print(f"[checkpoint] ONNX export skipped: {exc}")
        return None
