# MCAI training — GPU Docker box

Packages the full stack so it runs anywhere with an NVIDIA GPU:
- **Gym**: patched vanilla Minecraft server (JDK 21), headless, real physics/worldgen/mining.
- **Trainer**: Python 3.11 + PyTorch 2.5.1 (CUDA 12.1) PPO, ONNX export.
- **SSH** so you can connect and drive it.

The decompiled+patched server sources are baked in; the image only compiles the gym, it does
**not** re-run the heavy decompile. The Python venv is rebuilt fresh with pinned torch-cu121.

## Host prerequisites (Ricman)
- NVIDIA driver (recent enough for CUDA 12.1) + **nvidia-container-toolkit** installed.
  Verify: `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`
- ~10 GB disk for the image; a few GB more for checkpoints.

## Build
From the repo root (the build context must be the repo root, not `docker/`):
```bash
docker build -f docker/Dockerfile -t mcai-train .
```
(Needs network during build: gradle 8.10 + Mojang/maven deps + the torch cu121 wheel.)

## Run
**Key gotcha:** the gym uses `/dev/shm` for its Python↔gym transport, and Docker's default
`/dev/shm` is only 64 MB — you **must** pass `--shm-size`.

```bash
docker run -d --name mcai \
  --gpus '"device=0"' \          # one RTX 6000; use --gpus all for both
  --shm-size=4g \                 # REQUIRED for the shm transport
  -p 2222:22 -p 8088:8088 \       # ssh + web monitor
  -e SSH_PUBKEY="$(cat ~/.ssh/id_ed25519.pub)" \
  -v $PWD/runs:/workspace/mcai/trainer/runs \   # checkpoints on the host
  mcai-train
```
or `SSH_PUBKEY="$(cat ~/.ssh/id_ed25519.pub)" docker compose -f docker/docker-compose.yml up -d --build`.

## Connect & train (you)
```bash
ssh -p 2222 root@<box>            # key-based (password auth is off)
# optional: tunnel the live monitor ->  ssh -p 2222 -L 8088:localhost:8088 root@<box>

cd /workspace/mcai
AGENTS=64 NUM_ENVS=8 RUN=woodscale ./train.sh    # 8 gyms x 64 = 512 agents
# resume:  EXTRA="--resume" RUN=woodscale ./train.sh
```
Open http://localhost:8088 (through the tunnel) for the live agent monitor.

## Scaling
The **gym is CPU-bound** (real ServerPlayers), so throughput scales with **`NUM_ENVS`** —
parallel gym processes. Set it toward the box's usable core count; `AGENTS` is players per gym.
The RTX 6000 removes the old GPU-update bottleneck, so push agents until the cores saturate
(watch `sps` and `collect=` in the log). RAM: each agent loads ~a 5×5 chunk island; budget a
few GB per ~50 agents. Start e.g. `AGENTS=64 NUM_ENVS=8`, then tune.

## Checkpoints / syncing home
Checkpoints land in `trainer/runs/<run>/` (mounted to `./runs` on the host). To pull them home
hourly (Ricman's suggestion), from your machine:
```bash
while true; do rsync -az -e 'ssh -p 2222' root@<box>:/workspace/mcai/trainer/runs/ ./runs-home/; sleep 3600; done
```
Export a checkpoint to ONNX (for the Fabric mod):
```bash
cd /workspace/mcai/trainer
.venv/bin/python export_ckpt.py runs/woodscale /workspace/mcai/weights/woodscale.onnx
```

## Native (GraalVM) build — optional throughput lever
The image ships **GraalVM CE 21 with `native-image`**, so the gym can be AOT-compiled to a native
binary on the box (faster startup, ~1.5× steady-state — one of the scaling levers):
```bash
cd /workspace/mcai/minecraft-decomp
./gradlew nativeServer        # GraalVM native-image build (slow + RAM-hungry; run on the box, not in `docker build`)
```
(Plain JVM training works without this; native is just the extra speedup.)

## Notes
- Current trainable task is **gather-wood in wild real-terrain** (the proven pipeline). PvP and
  parkour need new task/reward modules (`mcai_train/tasks/`) + spawn modes — the environment here
  is ready for them; they're the next code to write.
- One RTX 6000 (48 GB) is far more than the ~0.6 M-param model needs; the win is removing the
  update stall, not VRAM. Multi-GPU isn't wired up yet (use one).
