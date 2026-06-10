#!/usr/bin/env bash
# Scaled training launch for the RTX 6000 (96 GB) box: Rust gym + large batch + GPU pipeline.
#
# The old defaults (4-32 agents, minibatch 2048, 646k-param model) were sized for a GTX 1060 6GB
# dev card and the slow Java gym — which left the RTX 6000 at ~5 GB VRAM and GPU-starved. The Rust
# gym is ~7x faster, so we can now feed a much larger batch and actually use the hardware. Batch
# scaling (agents + minibatch) does NOT invalidate an existing checkpoint; only growing the model
# would. Override any knob via env var.
set -euo pipefail
cd "$(dirname "$0")/.."  # -> trainer/

REPO="$(cd .. && pwd)"
# Build once: (cd "$REPO/rust-gym" && cargo build --release)
export MCAI_RUST_GYM="${MCAI_RUST_GYM:-$REPO/rust-gym/target/release/mcai-gym}"
export MCAI_GYM_SPACING="${MCAI_GYM_SPACING:-512}"   # blocks between agents within a gym
export PYTHONUNBUFFERED=1

if [[ ! -x "$MCAI_RUST_GYM" ]]; then
  echo "Rust gym binary not found at $MCAI_RUST_GYM — build it: (cd $REPO/rust-gym && cargo build --release)" >&2
  exit 1
fi

# ~512 agents (8 gyms x 64), big minibatch to fill VRAM and cut update-loop overhead, pipelined so
# the update overlaps the next collect. Tune MINIBATCH up (32k+) to push VRAM toward full.
NUM_ENVS="${NUM_ENVS:-8}"
N_AGENTS="${N_AGENTS:-64}"
ROLLOUT="${ROLLOUT:-256}"
MINIBATCH="${MINIBATCH:-16384}"
EPOCHS="${EPOCHS:-3}"
RUN="${RUN:-woodrust}"

echo "[train_box] gym=$MCAI_RUST_GYM agents=$((NUM_ENVS*N_AGENTS)) minibatch=$MINIBATCH rollout=$ROLLOUT pipeline=on"
exec .venv/bin/python -m mcai_train.train \
  --num-envs "$NUM_ENVS" \
  --n-agents "$N_AGENTS" \
  --rollout-len "$ROLLOUT" \
  --minibatch "$MINIBATCH" \
  --model-scale "${MODEL_SCALE:-1}" \
  --epochs "$EPOCHS" \
  --lr "${LR:-3e-4}" \
  --device cuda \
  --arena wild \
  ${ASYNC:+--async-collect} \
  ${PIPELINE:+--pipeline} \
  --run-name "$RUN" \
  --checkpoint-dir "runs/$RUN" \
  --checkpoint-every "${CKPT:-1000000}" \
  --monitor-port "${MONITOR_PORT:-9080}" \
  --total-timesteps "${TOTAL:-1000000000}" \
  ${RESUME:+--resume}
