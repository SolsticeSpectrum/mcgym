# mcai-gym (Rust) — status

A fast, headless, vanilla-faithful Minecraft RL gym built on **Pumpkin**, replacing the
decompiled-vanilla Java gym. Drop-in for the existing PyTorch PPO trainer over shm + UDS.

## Why

The Java gym is CPU-bound at ~14 TPS/world (~71 ms/tick) — slower than real-time — because each
tick runs the full JVM server plus heavy observation extraction. The GPU sits ~90% idle. This
crate rebuilds the gym in Rust on Pumpkin's vanilla-parity world sim, stripping everything that
isn't vanilla simulation (no networking, no disk I/O, no async chunk-system, no `pumpkin` server
binary — only `pumpkin-world`/`pumpkin-data`/`pumpkin-util`).

## Done (built + tested — 13 tests green; wood loop + curriculum functional, trainer-validated)

- **Schema codec** (`src/schema.rs`) — packed little-endian obs (40169 B) / action (27 B), the
  cross-language contract. Reproduces `schema/fixtures/golden_*.bin` **byte-identical** + round-trips
  (`tests/golden.rs`, 4 tests).
- **Transport** (`src/transport.rs`) — shm mmap (64 B header) + UDS RESET/STEP/CLOSE↔OK loop behind
  a `Gym` trait. Validated against a real socket + shm, incl. action→obs plumbing
  (`tests/transport.rs`).
- **Headless worldgen + reads** (`src/world.rs`) — `VanillaGenerator(seed, OVERWORLD)`. Two paths:
  fast terrain-only (`ensure_terrain_chunk`) and full vanilla through the **Features stage with real
  trees** (`ensure_chunk`, via `generate_single_chunk` + a noop block registry = mobs off). In-memory
  chunk cache, no disk/network. Benchmarked (`src/bin/genbench.rs`): 2634 logs + 5804 leaves across
  36 chunks, ~31 ms/chunk full gen.
- **Registry bridge** (`src/registry.rs`) — Pumpkin `state_id` → `Block::from_state_id().name` →
  `minecraft:`+name → `schema/registry.json` int, as a dense precomputed table (`include_str!`,
  self-contained). `tests/registry.rs`: air→0, stone→1, oak_log→49, high vanilla coverage.
- **Voxel extraction** (`src/obs.rs`) — `fill_voxels` reproduces the Java gym's exact cell order
  (`index = ((dy+r)*edge+(dz+r))*edge+(dx+r)`, dy>dz>dx, X fastest) with registry-mapped ids.
  `tests/obs.rs`: grid centre == block at agent pos, tree trunk spans multiple cells.
  TODO: legit-mode `-1` occlusion for enclosed blocks (deployment parity).
- **Agent physics** (`src/physics.rs`) — player AABB (0.6×1.8), vanilla moveRelative→collide→
  gravity/friction with standard constants; axis-separated (Y,X,Z) collision sweep against real
  `pumpkin-data` block collision shapes. `tests/physics.rs`: drops and lands at the exact surface
  height, rests stably, walks forward. TODO refinements: step-up, fluid buoyancy/swim, fall damage,
  sprint-jump boost, tick-for-tick speed validation vs the client.
- **Full observation assembly** (`src/obs.rs::build_obs`) — self-state + near/far voxels + raycast
  `target_*` (voxel DDA from the eye, MC face order). Inventory/entities/occlusion still zero/TODO.
- **GymState + binary** (`src/gym.rs`, `src/main.rs`) — N agents in one world, `Gym` impl
  (RESET/STEP, no-teleport reset), grid spawn on the surface. CLI `--shm --sock --agents --seed
  --spacing --arena --curriculum`, fail-fast. `tests/gym.rs`: spawn → valid obs → agents walk.
  **Binary boots for real**: generates world, maps shm at exactly `64+n·27+n·40169` bytes, binds
  UDS, prints `MCAI_TRANSPORT_READY`. It is a drop-in the Python trainer can launch.
- **Mining + inventory** (`src/gym.rs::mine_step`, `src/physics.rs`) — `attack` + a block in reach
  accumulates vanilla break progress (speed/hardness/30; axe speed 6 on wood), breaks the block,
  drops the item into the agent's inventory (direct-add; item-entity-on-ground is a refinement).
  Agents start with an axe; inventory flows into the obs. `tests/mining.rs`: aim at a real log →
  breaks in ~10 ticks → log in inventory → block becomes air. **The wood reward signal now works.**
- **WILD curriculum** (`src/gym.rs`) — forest-seek spawn (stand by the nearest log, facing it),
  dynamic chunk generation as agents roam (never walk into void), persistent home, death→respawn at
  home (structure ready; no damage source yet), and stuck-relocation only when alive + no progress
  (no move ≥3 / no break) for 1500 ticks. RESET never teleports. `tests/curriculum.rs`: most
  spread-spawned agents start with logs in their observations.

### Benchmark (single thread, release, this box)

| metric | value |
|---|---|
| block reads | **74.8 M blocks/s** |
| 64-agent 17³ near-voxel fill | **4.5 ms** |
| terrain gen (no features yet) | 33 chunks/s (~30 ms/chunk), one-time/cached |

The read path — the Java gym's bottleneck — is ~8–15× cheaper per tick before any cross-core
parallelism. Core speed thesis validated.

## Benchmarks (validated)

Raw gym throughput (`scripts/bench_throughput.py`, local box, no policy):

| agents/world | TPS/world | agent-steps/s |
|---|---|---|
| 1  | 6248 | 6248 |
| 16 | 416  | 6659 |
| 64 | **97** | 6227 |

vs the **Java gym ~14 TPS/world** at 64 agents on a *faster* cloud box → **~7× faster per world**,
understated by the slower local machine. Agent-steps/s is ~constant (obs-fill-bound), so it scales
linearly with worlds across cores.

End-to-end: the real PyTorch trainer drives the Rust gym on CUDA (set `MCAI_RUST_GYM=<binary>`),
`[train] done: 6144 timesteps in 10.6s`, stable sps, no errors. Now that the gym is fast, the
per-step Python collect overhead (GPU forward + tensor copy + socket) is the new ceiling — exactly
what #30 targets. (wood/agent=0 because grid-spawned agents aren't near trees yet — that's #28.)

## GPU collect/update pipeline (#30 — done, in the trainer)

`trainer/.../train.py --pipeline`: runs the PPO update on a background thread (live model) while the
main thread collects the next rollout with a deep-copied snapshot policy — disjoint modules (no
races), bounded 1-rollout staleness (PPO clips it). Validated: serial 7.8s vs pipeline 6.5s on a
small local config, loss sane, clip_frac rises modestly as expected. Default (serial) path
unchanged. Bigger win needs CUDA-stream separation / larger batches — tune with profiling on the box.

## Scaling for the RTX 6000 (96 GB)

The old config (4–32 agents, minibatch 2048, 646k-param model) was sized for a GTX 1060 6 GB dev
card *and* the slow Java gym, leaving the server at ~5 GB VRAM and GPU-starved. The fast Rust gym
unlocks large batches. `trainer/scripts/train_box.sh` launches the scaled run (Rust gym + ~512
agents + minibatch 16k + `--pipeline`). Validated locally that scaling agents raises both VRAM and
throughput (8→32 agents: 769→2840 MiB, sps 680→987). Batch scaling does not invalidate a
checkpoint; growing the model would (a future lever for harder tasks — there's 90 GB of headroom).

**Caveat — spawn cost at scale:** forest-seek generates a 5×5 chunk neighbourhood per agent, so
booting ~512 agents generates thousands of chunks (minutes). One-time per run, but slow on
supervisor restarts. Mitigations: parallelise chunk gen (rayon), smaller spawn radius, or persist
the generated world.

## Next (remaining tasks)

1. **Obs refinements** — nearest-16 entities + legit-mode `-1` occlusion; physics refinements
   (step-up, fluids, fall damage incl. a death source so respawn-at-home actually triggers).
2. **Reproducible Pumpkin dep** — pinned git-dep/submodule instead of the gitignored path dep.
3. **Pipeline tuning** — separate CUDA streams for collect-forwards vs update so they don't
   serialise; profile on the cloud box at 64 agents × N envs.
4. **Scale validation** — a long multi-million-step run on the cloud box (forest-seek spawns) to
   confirm wood/agent climbs, then compare learning curves + wall-clock to the Java gym.

## Reproducibility TODO

Pumpkin is a path dep into `reference/pumpkin` (gitignored). Before this is the canonical build,
switch to a pinned git-dep (rev `7bff068`) or vendor/submodule.

## Build notes

- Needs **rustc ≥ 1.95** (Pumpkin). This box's distro `rustc` is 1.94.1 and `/bin` precedes
  `~/.cargo/bin` in PATH, so cargo's internal `rustc` resolves to the wrong one. Build with the
  stable toolchain bin first on PATH:
  `export PATH="$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH"`.
- Pumpkin is pulled via path deps into `reference/pumpkin` (gitignored). For a reproducible build,
  switch to a pinned git-dep (rev `7bff068`) or vendor/submodule before committing as the canonical
  build.
