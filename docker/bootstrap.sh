#!/bin/bash
# one script behind both supervisord programs the compose init writes,
# bootstrap.sh ssh runs dropbear, bootstrap.sh train sets up and trains
set -euo pipefail

MODE="${1:?usage: bootstrap.sh ssh|train}"

DRIVE=/drive2
INIT=$DRIVE/mcgym-init  # written by the compose init service on every up
SSH=$DRIVE/xgl-ssh      # dropbear host key + deb cache
TOOLS=$DRIVE/tools      # rustup, cargo, venv, caches
SRC=$DRIVE/mcai         # repo clone, no state inside
RUNS=$DRIVE/runs        # checkpoints, outside the clone

ssh_mode() {
    mkdir -p "$SSH/debs" "$HOME/.ssh"
    cp "$INIT/authorized_keys" "$HOME/.ssh/authorized_keys"
    chmod 700 "$HOME/.ssh"
    chmod 600 "$HOME/.ssh/authorized_keys"

    # no real root in this image, apt only works under fakeroot, and openssh
    # cannot run at all (privsep chroot gets EPERM) hence dropbear
    if [ ! -x /usr/sbin/dropbear ]; then
        fakeroot dpkg -i "$SSH"/debs/*.deb 2>/dev/null || {
            fakeroot apt-get update
            fakeroot apt-get install -y dropbear-bin
            (cd "$SSH/debs" && apt-get download dropbear-bin libtomcrypt1 libtommath1) || true
        }
    fi

    # persisted host key so clients never see a host key change
    [ -f "$SSH/dropbear_ed25519_host_key" ] \
        || dropbearkey -t ed25519 -f "$SSH/dropbear_ed25519_host_key"

    exec /usr/sbin/dropbear -F -E -s -p 2222 -r "$SSH/dropbear_ed25519_host_key"
}

train_mode() {
    : "${REPO_URL:?REPO_URL not set, configure docker/.env}"
    mkdir -p "$TOOLS" "$RUNS"

    export RUSTUP_HOME=$TOOLS/rustup CARGO_HOME=$TOOLS/cargo
    export PATH="$TOOLS/cargo/bin:$PATH"
    export PIP_CACHE_DIR=$TOOLS/pip-cache

    # image ships python without ensurepip
    python3 -c 'import ensurepip' 2>/dev/null || {
        fakeroot apt-get update
        fakeroot apt-get install -y python3-venv
    }

    command -v cargo >/dev/null \
        || curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path

    # clone on first boot, hard reset to origin afterwards, offline keeps the
    # old checkout
    if [ ! -d "$SRC/.git" ]; then
        [ -e "$SRC" ] && mv "$SRC" "$SRC.bak.$(date +%s)"
        git clone "$REPO_URL" "$SRC"
    else
        git -C "$SRC" fetch origin && git -C "$SRC" reset --hard origin/main \
            || echo "[bootstrap] git update failed, training with the existing checkout"
    fi

    # cuda torch wheels bundle the runtime, only the driver comes from the host
    VENV=$TOOLS/venv
    [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet torch numpy pyyaml onnx
    "$VENV/bin/pip" install --quiet -e "$SRC/trainer"
    ln -sfn "$VENV" "$SRC/trainer/.venv"

    (cd "$SRC/mcgym" && cargo build --release)

    # frontend for the monitor, dist is a build artifact and not in git
    command -v npm >/dev/null || {
        fakeroot apt-get update
        fakeroot apt-get install -y nodejs npm
    }
    (cd "$SRC/frontend" && npm install --silent && npm run build)

    # train forever, train_box.sh carries the tuned defaults, knobs come from
    # the supervisord environment, resume is safe with no checkpoint
    export MCGYM_BF16=1 MCGYM_COMPILE=1 MCGYM_PROFILE=1
    while true; do
        pkill -9 -f "mcgym --shm" 2>/dev/null || true
        RESUME=1 RUN="${TASK:-wood}" CKPT_DIR="$RUNS/${TASK:-wood}" bash "$SRC/trainer/scripts/train_box.sh" \
            || echo "[bootstrap] train exited ($?)"
        echo "[bootstrap] restarting in 15s"
        sleep 15
    done
}

case "$MODE" in
    ssh) ssh_mode ;;
    train) train_mode ;;
    *) echo "unknown mode $MODE, use ssh|train" >&2; exit 1 ;;
esac
