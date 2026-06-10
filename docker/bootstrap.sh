#!/bin/bash
# MCAI box bootstrap — the one script behind both supervisord programs (see
# docker-compose.yml, which writes the program configs via the mcgym-init service):
#
#   bootstrap.sh ssh     dropbear SSH on :2222, keys from docker/.env
#   bootstrap.sh train   toolchain + repo clone + gym build + supervised wood task
#
# Image quirks this is written around (ghcr.io/selkies-project/nvidia-glx-desktop):
#   - /usr/bin/sudo is a fakeroot symlink (shells look like root but are uid 1000);
#     apt/dpkg work under `fakeroot`, real setuid sudo is /usr/bin/sudo-root.
#   - OpenSSH sshd cannot run (privsep chroot gets EPERM) — dropbear instead.
#   - The container is disposable: everything that must survive a recreate goes
#     under /drive2 (toolchains, repo clone, checkpoints, dropbear host key).
set -euo pipefail

MODE="${1:?usage: bootstrap.sh ssh|train}"

DRIVE=/drive2
INIT_DIR=/opt/mcgym-init        # written by the compose mcgym-init service on every `up`
SSH_DIR=$DRIVE/xgl-ssh         # persisted dropbear state (host key + deb cache)
TOOLS=$DRIVE/tools             # rustup/cargo/venv/pip-cache — survive recreates
SRC=$DRIVE/mcai                # git clone of the repo (disposable; no state inside)
RUNS=$DRIVE/runs               # checkpoints, outside the clone

ssh_mode() {
    mkdir -p "$SSH_DIR/debs" "$HOME/.ssh"
    cp "$INIT_DIR/authorized_keys" "$HOME/.ssh/authorized_keys"
    chmod 700 "$HOME/.ssh"
    chmod 600 "$HOME/.ssh/authorized_keys"

    if [ ! -x /usr/sbin/dropbear ]; then
        # Offline install from cached debs; fall back to apt and refill the cache
        fakeroot dpkg -i "$SSH_DIR"/debs/*.deb 2>/dev/null || {
            fakeroot apt-get update
            fakeroot apt-get install -y dropbear-bin
            (cd "$SSH_DIR/debs" && apt-get download dropbear-bin libtomcrypt1 libtommath1) || true
        }
    fi

    # Persisted host key so clients never see a host-key change across recreates
    [ -f "$SSH_DIR/dropbear_ed25519_host_key" ] \
        || dropbearkey -t ed25519 -f "$SSH_DIR/dropbear_ed25519_host_key"

    exec /usr/sbin/dropbear -F -E -s -p 2222 -r "$SSH_DIR/dropbear_ed25519_host_key"
}

train_mode() {
    : "${REPO_URL:?REPO_URL not set — configure docker/.env}"
    mkdir -p "$TOOLS" "$RUNS"

    export RUSTUP_HOME=$TOOLS/rustup CARGO_HOME=$TOOLS/cargo
    export PATH="$TOOLS/cargo/bin:$PATH"
    export PIP_CACHE_DIR=$TOOLS/pip-cache

    # python3-venv is not in the image (the venv module is there but ensurepip is not)
    python3 -c 'import ensurepip' 2>/dev/null || {
        fakeroot apt-get update
        fakeroot apt-get install -y python3-venv
    }

    # Rust toolchain, persisted under /drive2/tools
    command -v cargo >/dev/null \
        || curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path

    # Code: clone on first boot, fast-forward to origin afterwards. The clone holds
    # no state (checkpoints in $RUNS, venv in $TOOLS), so a hard reset is safe; if
    # the network is down we train with whatever is already there.
    if [ ! -d "$SRC/.git" ]; then
        [ -e "$SRC" ] && mv "$SRC" "$SRC.bak.$(date +%s)"
        git clone "$REPO_URL" "$SRC"
    else
        git -C "$SRC" fetch origin && git -C "$SRC" reset --hard origin/main \
            || echo "[bootstrap] git update failed; training with the existing checkout"
    fi

    # Python venv + torch (CUDA wheels bundle the runtime; only the driver comes
    # from the host). Persisted in $TOOLS so recreates skip the multi-GB download.
    VENV=$TOOLS/venv
    [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet torch numpy pyyaml onnx
    "$VENV/bin/pip" install --quiet -e "$SRC/trainer"
    ln -sfn "$VENV" "$SRC/trainer/.venv"   # train_box.sh runs .venv/bin/python

    (cd "$SRC/mcgym" && cargo build --release)

    # frontend for the monitor, dist is a build artifact and not in git
    command -v npm >/dev/null || {
        fakeroot apt-get update
        fakeroot apt-get install -y nodejs npm
    }
    (cd "$SRC/frontend" && npm install --silent && npm run build)

    # The wood task, forever: train_box.sh carries the tuned defaults; the compose
    # passes knob overrides via the supervisord program environment. RESUME=1 is
    # safe with no checkpoint (fresh start). Crashes restart after a pause.
    export MCGYM_BF16=1 MCGYM_COMPILE=1 MCGYM_PROFILE=1
    while true; do
        pkill -9 -f "mcgym --shm" 2>/dev/null || true
        RESUME=1 RUN="${TASK:-wood}" CKPT_DIR="$RUNS/${TASK:-wood}" bash "$SRC/trainer/scripts/train_box.sh" \
            || echo "[bootstrap] train exited ($?)"
        echo "[bootstrap] restarting training in 15s"
        sleep 15
    done
}

case "$MODE" in
    ssh) ssh_mode ;;
    train) train_mode ;;
    *) echo "unknown mode: $MODE (use ssh|train)" >&2; exit 1 ;;
esac
