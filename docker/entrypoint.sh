#!/usr/bin/env bash
# Container entrypoint: install the SSH key, start sshd, keep the container alive.
# Connect with:  ssh -p <hostport> root@<box>   (key-based; password auth is off)
set -euo pipefail

# Authorize the operator's public key (passed at `docker run` via -e SSH_PUBKEY="ssh-ed25519 ...").
if [ -n "${SSH_PUBKEY:-}" ]; then
    mkdir -p /root/.ssh
    echo "$SSH_PUBKEY" > /root/.ssh/authorized_keys
    chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
else
    echo "[entrypoint] WARNING: no SSH_PUBKEY provided — you won't be able to ssh in." >&2
fi

# Quick GPU sanity line in the container log.
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
    echo "[entrypoint] nvidia-smi not found — is the container run with --gpus and host drivers?" >&2
fi

cat <<EOF
[entrypoint] MCAI training container ready.
  repo:     ${MCAI_HOME}
  trainer:  ${MCAI_HOME}/trainer/.venv/bin/python -m mcai_train.train ...
  helper:   ${MCAI_HOME}/train.sh   (AGENTS=.. NUM_ENVS=.. RUN=.. ./train.sh)
  checkpoints persist in trainer/runs/ (mount a volume there to keep/sync them).
EOF

# Foreground sshd so the container stays up and logs go to docker.
exec /usr/sbin/sshd -D -e
