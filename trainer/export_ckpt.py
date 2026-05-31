"""Export the latest checkpoint of a run to ONNX. Usage: export_ckpt.py <run-dir> <out.onnx>"""
import pathlib
import shutil
import sys

from mcai_train.checkpoint import export_onnx, load_latest
from mcai_train.models.policy import ActorCritic
from mcai_train.train import _model_sizes, REGISTRY_PATH

run_dir = sys.argv[1]
out = pathlib.Path(sys.argv[2])

num_blocks, num_items = _model_sizes(REGISTRY_PATH)
model = ActorCritic(num_blocks, num_items)
meta = load_latest(run_dir, model, None)
if meta is None:
    raise SystemExit(f"no checkpoint in {run_dir}")
model.eval()
print(f"loaded {run_dir} @ {meta.get('cumulative_timesteps')} steps")

tmp = pathlib.Path(run_dir) / "_export"
path = export_onnx(tmp, model)
if path is None:
    raise SystemExit("ONNX export failed")
out.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(path, out)
print(f"wrote {out} ({out.stat().st_size} B)")
