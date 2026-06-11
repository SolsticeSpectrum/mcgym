# MCGym

RLGym but for Minecraft. A from scratch PPO stack with a fast headless sim,
a real SteelMC server and real azalea clients lockstepped in one process. Trained
policies run in a Fabric mod, the agent sees only structured observations, no pixels.

Currently learning to gather wood. The architecture is task driven.

## How it works

```
sim/         rust sim, steel server + azalea client swarm behind a shm + unix socket transport
trainer/     pytorch PPO
schema/      single source of truth for the obs/action binary layout
fabric-mod/  runs an exported onnx policy in the live client
frontend/    react + three.js training monitor
docker/      fresh gpu box to running training
```

One sim process is one server world with N azalea clients stepped in lockstep,
client and server exchange real protocol bytes in memory, no sockets, no disk. The
trainer talks to many sims over shared memory, ticks them in parallel across cores
and trains on gpu. Observations are a 17x17x17 voxel grid around the agent with per
block collision boxes, a strided far shell, scalars and the inventory.

## Train

```bash
cd sim && cargo build --release
cd ../trainer && python -m venv .venv && .venv/bin/pip install torch -e .
SIM=../sim/target/release/sim .venv/bin/python -m mcgym.train --task wood --agents 4 --monitor 9080
```

The monitor at http://localhost:9080 shows every agent live, per env minimaps,
 `trainer/scripts/train_box.sh` carries the tuned big gpu config.

## Run a policy

```bash
.venv/bin/python export_ckpt.py runs/wood weights/wood.onnx
cd ../fabric-mod && ./gradlew runClient
```

Then in chat: `.run wood`, `.stop`, `.status`. The mod builds the exact training
observation from the live client and drives the player with valid inputs only,
movement through the input object, mining through the interaction manager with
vanilla packet timing.

## Schema

The gym, trainer and mod share one binary contract, `schema/mcgym_schema.yaml`.
Golden fixtures pin the bytes, every implementation must reproduce them, see
`schema/README.md`. Ids map by resource name so client registry order never matters.

## Deploy a training box

Copy `docker/` to a gpu host, fill `.env`, `docker compose up -d`. The container
bootstraps toolchains, clones this repo, builds the gym and starts training with
the monitor exposed, checkpoints survive restarts. See `docker/README.md`.
