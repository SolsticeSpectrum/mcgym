#!/usr/bin/env bash
# Bundle the Docker build context into one tarball to ship to the GPU box.
#
# WHY: the gym's compilable sources (minecraft-decomp/server/src) are GENERATED and gitignored,
# and minecraft-decomp is a separate repo the umbrella ignores — so `git clone` does NOT produce
# a buildable tree. This tar captures the on-disk working tree (with the generated sources) minus
# the big stuff the image rebuilds or doesn't need. Ship it, then `docker build` on the box.
#
#   ./docker/pack.sh                       # -> mcai-context.tar.gz
#   scp -P <port> mcai-context.tar.gz root@<box>:/root/
#   # on the box:
#   mkdir mcai && tar -xzf mcai-context.tar.gz -C mcai && cd mcai
#   docker build -f docker/Dockerfile -t mcai-train .
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/mcai-context.tar.gz}"

# NOTE: exclude patterns must match the STORED member paths (no leading ./), and we list
# explicit top-level dirs so reference/ and fabric-mod/ are simply never archived.
tar -czf "$OUT" -C "$ROOT" \
    --exclude='*/.git' \
    --exclude='*/.gradle' \
    --exclude='*/__pycache__' \
    --exclude='*.log' \
    --exclude='trainer/.venv' \
    --exclude='trainer/runs' \
    --exclude='trainer/build' \
    --exclude='minecraft-decomp/decompSrc' \
    --exclude='minecraft-decomp/libs' \
    --exclude='minecraft-decomp/patchSrc' \
    --exclude='minecraft-decomp/run' \
    --exclude='minecraft-decomp/run_server' \
    --exclude='minecraft-decomp/server/build' \
    --exclude='minecraft-decomp/buildSrc/build' \
    minecraft-decomp schema trainer docker weights .dockerignore

echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "ship:  scp -P <port> $OUT root@<box>:/root/"
echo "build: tar -xzf $(basename "$OUT") && docker build -f docker/Dockerfile -t mcai-train ."
