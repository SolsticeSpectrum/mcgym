"""export the latest checkpoint of a run to onnx, usage: export_ckpt.py <run-dir> <out.onnx>"""
import pathlib
import shutil
import sys

import torch

from mcgym.checkpoint import _checkpoint_dirs, export_onnx, load_latest
from mcgym.models.policy import EMBED_DIM, ActorCritic
from mcgym.train import sizes, REGISTRY

run = sys.argv[1]
out = pathlib.Path(sys.argv[2])

blocks, items = sizes(REGISTRY)

# infer scale from the checkpoint itself, embed width is EMBED_DIM * scale,
# so exports work for any run without knowing its training args
ckpt = _checkpoint_dirs(pathlib.Path(run))[-1] / "model.pt"
state = torch.load(ckpt, map_location="cpu", weights_only=True)

scale = state["encoder.block_embed.weight"].shape[1] // EMBED_DIM
print(f"inferred model scale={scale}")

model = ActorCritic(blocks, items, scale=scale)

meta = load_latest(run, model, None)
if meta is None:
    raise SystemExit(f"no checkpoint in {run}")

model.eval()
print(f"loaded {run} @ {meta.get('steps')} steps")

path = export_onnx(pathlib.Path(run) / "_export", model)
out.parent.mkdir(parents=True, exist_ok=True)

shutil.copy(path, out)
print(f"wrote {out} ({out.stat().st_size} B)")
