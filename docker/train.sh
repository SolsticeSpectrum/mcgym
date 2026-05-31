#!/usr/bin/env bash
# Scale-tunable training launcher. The gym is CPU-bound, so throughput scales with
# NUM_ENVS (parallel gym processes) — set it toward the box's usable core count. The
# RTX 6000 handles the PPO update + big batches easily, so the GPU is no longer the limit.
#
#   AGENTS=64 NUM_ENVS=8 RUN=woodscale ./train.sh        # 8 gyms x 64 = 512 agents
#   EXTRA="--resume" RUN=woodreal ./train.sh             # resume a run
set -euo pipefail
cd "${MCAI_HOME:-/workspace/mcai}/trainer"

AGENTS=${AGENTS:-64}          # real ServerPlayers per gym process
NUM_ENVS=${NUM_ENVS:-1}       # parallel gym processes; raise toward usable cores
TOTAL=${TOTAL:-50000000}
RUN=${RUN:-woodscale}
EXTRA=${EXTRA:-}

echo "[train.sh] agents/env=$AGENTS  num_envs=$NUM_ENVS  total=$TOTAL  run=$RUN  cores=$(nproc)"
exec .venv/bin/python -m mcai_train.train \
    --n-agents "$AGENTS" --num-envs "$NUM_ENVS" \
    --total-timesteps "$TOTAL" --rollout-len 128 --episode-len 256 --epochs 2 \
    --device cuda --arena wild --run-name "$RUN" \
    --checkpoint-dir "runs/$RUN" --checkpoint-every 300000 \
    --monitor-port 8088 $EXTRA
