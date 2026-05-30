# MCAI — current status & how to run

A from-scratch RL stack that trains a Minecraft agent to gather wood in a fast headless
gym and runs the trained policy as a Fabric client mod. Built over several sessions.

## What works (committed)

| Layer | State |
|---|---|
| Canonical obs/action schema + codec + golden byte contract | done (`schema/`, `trainer/mcai_train/schema/`) |
| Headless vanilla-physics gym (real `ServerPlayer`s) | done (`minecraft-decomp/patches/gym/benchmark.patch`) |
| Shared-memory transport (Python↔gym) | done |
| Mining + inventory/target obs + **death-as-penalty** | done |
| Clustered flat **arena** (100+ agents, fast boot) | done (`-Parena=flat` / `--arena flat`) |
| Curriculum: `tree_ahead` (log 2 blocks ahead + axe) | done |
| PPO trainer (ActorCritic, GAE, checkpoints) on GPU | done (`trainer/mcai_train/`) |
| Web viewer: per-agent voxel 3D FOV + minimap, real map colors | done (`--monitor-port 8088`) |
| Fabric inference mod (MC 1.21.11, ONNX policy) | builds + `runClient`; **in-world loop unverified** |
| ONNX export of the trained policy | done (`weights/woodcurr3.onnx`) |

**Proven:** the agent learns to chop wood — `wood/agent` rose 0 → ~6 over training
(`trainer/runs/woodcurr3`, 1.07M+ steps).

## How to run

Python env: `trainer/.venv` (python3.11, torch 2.5.1+cu121 for the Pascal GTX 1060 — do NOT
`pip install` plain torch, it drops Pascal). All commands from `trainer/`.

```bash
# Train (single-process is the supported path) + watch in a browser:
.venv/bin/python -m mcai_train.train --n-agents 100 --arena flat --curriculum tree_ahead \
    --device cuda --run-name woodcurr3 --checkpoint-dir runs/woodcurr3 \
    --resume --monitor-port 8088
# then open http://127.0.0.1:8088/   (left: voxel 3D FOV, right: minimap, status below)

# Tests:  .venv/bin/python -m pytest -q -m "not slow"   (fast)   |   add slow ones for gym e2e

# Fabric mod (the "true benchmark"):
cd ../fabric-mod && ./gradlew runClient
#   join a world, type:  .run woodcurr3     (loads weights/woodcurr3.onnx; chat is swallowed
#   client-side). .stop / .status. THIS is the unverified bit — confirm it mines via valid packets.
```

Regenerate gym artifacts (in `minecraft-decomp/`): `./gradlew dumpRegistry dumpMapColors`,
edit `server/src/.../mcai/*.java`, then `./gradlew genGymPatch -PpatchName=benchmark`.

## Throughput (measured, honest)

Single-process: ~1.0–1.5k agent-steps/s at 100 agents (arena). Profiled bottleneck = the Java
tick (real per-agent physics + obs pack), not the GPU. Optimizations applied: bulk chunk-section
obs, block→id memo, vectorized reward. **The box (6 cores / GTX 1060) is CPU-bound on per-agent
vanilla physics** — `ParallelVecEnv` (`--num-envs`) is correct but gives no win here (can't run
enough heavy gyms on 6 cores). **100k tps needs more cores/machines or cheaper per-agent physics
(smaller obs radius / lighter entity ticking / GraalVM native build).**

## Open items / next

- Verify the Fabric mod in-world loop (`runClient` → `.run woodcurr3`): does it mine with valid packets?
- Throughput: GraalVM native build; smaller voxel radius (retrain); run `ParallelVecEnv` across machines.
- Curriculum annealing: axe→bare-handed, log farther→natural forests; fade shaping toward sparse Δwood.
- xray vs legit obs variants (two models).

## Branches
- `minecraft-decomp`: `feature/real-player-agents`
- umbrella `mcai`: `feature/shm-transport`
