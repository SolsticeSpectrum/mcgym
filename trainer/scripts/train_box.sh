#!/usr/bin/env bash
# training launch for the gpu box, defaults are the measured best config for
# the rtx 6000 (2048 agents at ~14.7k steps/s under a 64g memory cap), batch
# knobs dont invalidate a checkpoint, only SCALE would
set -euo pipefail
cd "$(dirname "$0")/.."  # -> trainer/

REPO="$(cd .. && pwd)"
export MCGYM="${MCGYM:-$REPO/mcgym/target/release/mcgym}"
export MCGYM_SPACING="${MCGYM_SPACING:-128}"   # blocks between agents within a gym
export MCGYM_BF16="${MCGYM_BF16:-1}"
export MCGYM_COMPILE="${MCGYM_COMPILE:-1}"
export PYTHONUNBUFFERED=1

if [[ ! -x "$MCGYM" ]]; then
  echo "mcgym binary not found at $MCGYM, build it with cargo build --release" >&2
  exit 1
fi

TASK="${TASK:-wood}"
RUN="${RUN:-$TASK}"

exec .venv/bin/python -m mcgym.train \
  --task "$TASK" \
  --num-envs "${NUM_ENVS:-32}" \
  --agents "${AGENTS:-64}" \
  --rollout "${ROLLOUT:-128}" \
  --minibatch "${MINIBATCH:-16384}" \
  --scale "${SCALE:-3}" \
  --epochs "${EPOCHS:-2}" \
  --lr "${LR:-3e-4}" \
  --device cuda \
  --cohorts "${COHORTS:-4}" \
  --run "$RUN" \
  --ckpt-dir "${CKPT_DIR:-runs/$RUN}" \
  --ckpt-every "${CKPT_EVERY:-1000000}" \
  --monitor "${MONITOR_PORT:-9080}" \
  --total "${TOTAL:-1000000000}" \
  ${RESUME:+--resume}
