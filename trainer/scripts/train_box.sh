#!/usr/bin/env bash
# Scaled training launch for the GPU box: Rust gym + K-cohort async collect + GPU pipeline.
#
# Defaults are the measured-best config for the RTX 6000 box (32 gyms x 64 agents = 2048,
# rollout 128, minibatch 16384, 2 epochs, 4 cohorts, scale-3 model, bf16+compile via
# MCAI_BF16/MCAI_COMPILE): ~14.7k steps/s with the two rollout buffers fitting a 64 GiB
# memory cap. Batch scaling (agents/rollout/minibatch) does NOT invalidate an existing
# checkpoint; only growing the model (MODEL_SCALE) would. Override any knob via env var.
set -euo pipefail
cd "$(dirname "$0")/.."  # -> trainer/

REPO="$(cd .. && pwd)"
# Build once: (cd "$REPO/rust-gym" && cargo build --release)
export MCAI_RUST_GYM="${MCAI_RUST_GYM:-$REPO/rust-gym/target/release/mcai-gym}"
export MCAI_GYM_SPACING="${MCAI_GYM_SPACING:-128}"   # blocks between agents within a gym
export PYTHONUNBUFFERED=1

if [[ ! -x "$MCAI_RUST_GYM" ]]; then
  echo "Rust gym binary not found at $MCAI_RUST_GYM — build it: (cd $REPO/rust-gym && cargo build --release)" >&2
  exit 1
fi

NUM_ENVS="${NUM_ENVS:-32}"
N_AGENTS="${N_AGENTS:-64}"
ROLLOUT="${ROLLOUT:-128}"
MINIBATCH="${MINIBATCH:-16384}"
EPOCHS="${EPOCHS:-2}"
MODEL_SCALE="${MODEL_SCALE:-3}"
ASYNC="${ASYNC:-1}"          # K-cohort async collect on by default (ASYNC= to disable)
COHORTS="${COHORTS:-4}"
RUN="${RUN:-woodopt}"

echo "[train_box] gym=$MCAI_RUST_GYM agents=$((NUM_ENVS*N_AGENTS)) minibatch=$MINIBATCH rollout=$ROLLOUT pipeline=on"
exec .venv/bin/python -m mcai_train.train \
  --num-envs "$NUM_ENVS" \
  --n-agents "$N_AGENTS" \
  --rollout-len "$ROLLOUT" \
  --minibatch "$MINIBATCH" \
  --model-scale "$MODEL_SCALE" \
  --epochs "$EPOCHS" \
  --lr "${LR:-3e-4}" \
  --device cuda \
  --arena wild \
  ${ASYNC:+--async-collect} \
  ${COHORTS:+--async-cohorts "$COHORTS"} \
  ${PIPELINE:+--pipeline} \
  --run-name "$RUN" \
  --checkpoint-dir "${CKPT_DIR:-runs/$RUN}" \
  --checkpoint-every "${CKPT:-1000000}" \
  --monitor-port "${MONITOR_PORT:-9080}" \
  --total-timesteps "${TOTAL:-1000000000}" \
  ${RESUME:+--resume}
