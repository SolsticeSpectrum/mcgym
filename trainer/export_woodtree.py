"""One-off: export the latest woodtree checkpoint to weights/woodtree.onnx.

Preserves the realistic-tree policy (avg ~18 wood, 100/100 agents) as a
deployable ONNX before the gym is reworked for real-terrain spawn.
"""
import pathlib
import shutil

import torch

from mcai_train.checkpoint import export_onnx, load_latest
from mcai_train.models.policy import ActorCritic
from mcai_train.train import _model_sizes, REGISTRY_PATH

OUT = pathlib.Path("/home/user/github/mcai/weights/woodtree.onnx")

num_blocks, num_items = _model_sizes(REGISTRY_PATH)
model = ActorCritic(num_blocks, num_items)
meta = load_latest("runs/woodtree", model, None)
if meta is None:
    raise SystemExit("no woodtree checkpoint found")
model.eval()
print(f"loaded woodtree latest meta={meta}")

tmp = pathlib.Path("runs/woodtree/_export")
path = export_onnx(tmp, model)
if path is None:
    raise SystemExit("ONNX export failed")
OUT.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(path, OUT)
print(f"wrote {OUT} ({OUT.stat().st_size} B)")
