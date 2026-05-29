# MCAI Training Stack — Design Spec

**Date:** 2026-05-29
**Status:** Approved (design phase)
**Authors:** rudolf.kratochvil@proton.me + Claude

## 1. Goal

Train a Minecraft agent **from scratch** with reinforcement learning, on GPU, in a
vanilla-accurate **headless** Minecraft running as fast as possible (rlgym /
rlgym-ppo–style). First task: **gather as much wood as possible**. The trained
policy must ultimately run as a **Fabric client-side mod** on any server, with no
special permissions.

### Hard constraints

- **Client-observable only.** The model may only consume information a vanilla
  client legitimately has. Server internals may be used for *rewards*,
  instrumentation, and debugging — never as policy inputs.
- **No vision.** Observations are structured/voxel data, not rendered pixels.
- **One canonical schema.** A single versioned observation/action schema is emitted
  byte-identically by (a) the Java gym during training and (b) the Fabric mod's
  mixins at inference. The schema *is* the transport wire format.
- **No corner-cutting.** Build the right architecture from the start; vanilla-accurate
  mechanics, resumable checkpoints, transport-agnostic core.

## 2. Repository layout

```
mcai/
  minecraft-decomp/                  # existing gym (Java, decompiled vanilla 1.21.11)
    patches/gym/
      benchmark.patch                # existing: free-running throughput benchmark
      agents.patch                   # NEW: real player-entity agents via vanilla code paths
      env_control.patch              # NEW: external reset/step + shared-memory transport
  schema/                            # NEW: single source of truth for obs/action layout
    mcai_schema.yaml                 # versioned field layout + block/item ID registry
    codegen/                         # generates Java + Python (+ Rust later)
  trainer/                           # NEW: Python package `mcai_train`
    pyproject.toml
    mcai_train/
      env/                           # VecEnv, shm transport, rlgym-style components
      ppo/                           # learner, policy, value, experience buffer
      models/                        # embedding trunk + heads
      checkpoint/                    # save / load / resume / export
      tasks/                         # task bundles (gather_wood)
      train.py                       # entrypoint
  fabric-mod/                        # LATER: inference mod; mixins emit the same schema
  reference/                         # cloned reference projects (gitignored)
  docs/superpowers/specs/            # this spec
```

`minecraft-decomp/` remains its own git repository. The umbrella `mcai/` repo
tracks `schema/`, `trainer/`, `fabric-mod/`, `docs/`; it gitignores `reference/`
and `minecraft-decomp/` (own repo) and large/generated artifacts.

## 3. Canonical schema (cornerstone)

A versioned spec in `schema/mcai_schema.yaml` is the single source of truth, codegen'd
into Java and Python (Rust later). It defines the exact binary byte layout *and* the
block/item ID registry dumped from `BuiltInRegistries` (stable order → "log = 42"
everywhere). A `schema_version` integer is written in the header and asserted on both
ends; a mismatch is a hard error.

### Observation (per agent, fixed size)

- **Self:** position(3, f32), velocity(3, f32), yaw(f32), pitch(f32), on_ground(u8),
  health(f32), food(f32), selected_hotbar_slot(u8).
- **Voxel grid:** block-type IDs (i32) in a cube of radius `R` blocks centered on the
  agent. **Default R = 8 → 17×17×17 = 4913 cells**, configurable. Subject to the
  `block_visibility` toggle:
  - `xray`: every block emitted as-is.
  - `legit`: blocks with no air-adjacent (exposed) face are replaced with a `HIDDEN`
    sentinel ID — what survives server-side anti-xray.
  The same gym emits either variant via a config flag, so the `xray` model and the
  `legit` model train from identical infrastructure.
- **Targeted block (raycast):** block_id(i32), face(u8), distance(f32), in_range(u8).
- **Entities:** up to `K` nearest mobs/players. **Default K = 16**, configurable. Each:
  type_id(i32), rel_pos(3, f32), velocity(3, f32), yaw(f32), health(f32), flags(u8).
  Entities are **always included regardless of line-of-sight** (no occlusion culling),
  matching real client behavior.
- **Inventory:** per-slot {item_id(i32), count(u8)} for main inventory (36) + armor (4)
  + offhand (1).

### Action (per agent, all client-legal)

- **Movement:** forward[-1,1] (f32), strafe[-1,1] (f32), jump(u8), sneak(u8), sprint(u8).
- **Camera:** yaw_delta(f32), pitch_delta(f32), clamped per tick to a realistic mouse
  range.
- **Interaction:** attack(u8), use(u8).
- **Hotbar:** selected_slot(0–8).
- **Inventory op:** a vanilla container-click {op_type, slot_a, slot_b} covering the
  azalea/vanilla `ClickOperation` vocabulary (Pickup / QuickMove / Swap / Throw).
  Mostly idle for v1 wood-gathering, but wired so crafting and tool-swapping come for
  free later.

For the policy, the action is encoded as a flat vector consumed by multi-discrete heads
(movement booleans, camera bins, hotbar, inventory-op) plus optional continuous camera.

## 4. Java gym integration — drive existing vanilla code

The decompiled tree already contains the complete vanilla implementation
(`Entity.move/collide`, `LivingEntity.travel/aiStep`, `ServerPlayerGameMode.destroyBlock`,
`Player`, `Inventory`, `Block` hardness/drops, real chunk gen). The current
`benchmark.patch` does **not** use it for agents — `AgentSlot.apply()` is a ~10-line
kinematic placeholder that fakes movement and ignores `attack`/`use`/`hotbarSlot`.

The work is **integration, not reimplementation**:

- Each agent becomes a **headless, networkless player entity** (no `Connection`/packets),
  ticked manually.
- Per tick: decode action → set player inputs (`zza/xxa/jumping/sprinting`,
  `setYRot/setXRot`) → run the existing `aiStep()/travel()/collide()` so vanilla gravity
  and collision apply. `attack` drives `ServerPlayerGameMode` block-breaking (real
  hardness + destroy progress) against the raycast target; vanilla `Block` drops →
  `Inventory` pickup.
- Agents spawn spread across the existing 512-block cells (no cross-interference) or in
  shared arenas per task.
- **Reward source** reads inventory wood count server-side (allowed — reward may use
  internals). Observations remain strictly client-observable.
- **Auto-reset:** per-agent `done` on timeout or task completion triggers a per-agent
  reset (teleport to a fresh spot, re-roll terrain features) so the vectorized env never
  stalls — mirrors rlgym-ppo auto-reset.

Two new patches: `agents.patch` (real player movement + mining) and `env_control.patch`
(external `reset(seed)→obs` / `step(actions)→(obs,reward,done)` + shared-memory transport).
The free-running benchmark loop becomes one run mode; training mode is externally driven.

## 5. Transport — shared memory + ping (option C), transport-agnostic core

The gym core exposes one synchronous function: `driveOneTick(actions) → obs`. Transport
is a separate layer so the core supports all three bridges from the same code.

**Default (C):** each Minecraft process mmaps a region (header + action buffer + obs
buffer) whose layout **is** the schema's binary form. A Unix-domain socket carries only
1-byte step/ready signals plus lifecycle messages (reset, close, env-shape handshake).
The bulk arrays are never serialized through the socket; both sides read the same RAM,
and Python wraps the obs region as a zero-copy numpy view.

- One Minecraft process = one **vectorized env** hosting N agents in one world.
- Python `VecEnv` wraps **M processes** → `M·N` agents gathered into one
  `(M·N, obs_dim)` GPU batch; actions scattered back into each process's action buffer.
- Process-per-env gives true multicore parallelism (no Python GIL bottleneck) and crash
  isolation (one world's chunk-gen failure doesn't kill the learner).

The binary codec is an isolated module reused by: the plain-socket fallback (B, for
debugging) and the GraalVM `@CEntryPoint` FFI (A, single in-process env) — both available
without rework.

## 6. RLGym-style modular components (Python)

Mirror the rlgym config API so tasks are swappable:

- `McaiEngine` — TransitionEngine; the shm bridge to a Minecraft process.
- `ObsBuilder` — decode shm bytes → obs tensors; `xray` / `legit` variants.
- `ActionParser` — policy action → packed action bytes.
- `GatherWoodReward` — Δ(wood count) + shaping (face a tree, look at a log, approach).
- `DoneCondition` — timeout / max-steps truncation; optional terminal on target count.
- `StateMutator` — reset config (seed, treed biome, bare-handed vs. axe).

A **task** bundles these. `tasks/gather_wood.py` is the first.

## 7. Model (from scratch, no LLM)

- **Shared learned embedding table** over block/item IDs, used by the voxel grid, the
  raycast target, and inventory (one table, shared weights).
- Voxel grid → small 3D-conv (or flatten + MLP). Entities → per-entity MLP + pooling
  (mean or attention). Inventory → MLP. Scalars → MLP. Concatenate → trunk
  (MLP or small transformer) → shared latent.
- **Heads:** policy = multi-discrete (movement booleans, discretized camera bins,
  hotbar, inventory-op) + optional continuous camera; value = scalar.
- A few million parameters; one GPU serves thousands of agents.

## 8. PPO trainer (adapt rlgym-ppo)

- Collect rollouts from `VecEnv` → experience buffer → GAE.
- PPO update on GPU: clipped surrogate, value loss, entropy bonus, gradient clipping,
  linear LR anneal, optional AMP.
- Welford running normalization on continuous/scalar inputs (IDs stay raw integers into
  the embedding table).
- Logging: console always; **wandb optional via explicit config flag** (no silent
  fallback, per project standards).

## 9. Checkpoints (resumable)

`checkpoints/<run-name>/<step>/` contains:

- `policy.pt`, `value.pt` (state dicts)
- `optimizer.pt`
- `obs_norm.json` (running normalization stats)
- `meta.json` (cumulative timesteps, env/agent counts, `schema_version`, patch/git hash,
  RNG states, full hyperparameters)

Retention: periodic every K steps + a `best` (by eval reward) + a `latest` symlink.
**Resume** restores optimizer state, timestep counter, RNG, and normalization stats →
exact continuation ("continue where it left off"). **Export** produces `policy.onnx`
(or TorchScript) tagged with the `schema_version` it was trained against, so the Fabric
mod loads the matching codec.

## 10. Testing

- **Java:** extend `GymPatchTest` — obs bytes match the schema on a fixed state; a
  scripted mining sequence yields +1 wood; seed determinism.
- **Schema:** Java-emit ↔ Python-decode round-trip equality on shared fixtures.
- **Python:** env smoke test (random policy, correct obs shapes, no crash); PPO learns a
  trivial reward (sanity); checkpoint save → load → resume equality.
- **Integration:** short `gather_wood` run; assert mean reward trends up.

## 11. Build order (thin vertical slice first)

1. Schema v0 + codegen (Java + Python).
2. `env_control` + `agents` patch: real vanilla movement, `reset`/`step`, shm transport;
   random-action smoke test driven from Python.
3. Mining: attack → `destroyBlock` → drop → inventory; reward reads wood count.
4. Model + PPO + checkpoints; train `gather_wood`; verify learning curve.
5. (Later) Fabric inference mod reusing the schema + ONNX export; `legit`/`xray` second
   model.

## 12. Defaults chosen (call out for review)

- Voxel radius **R = 8** (17³ = 4913 cells), configurable.
- Entity cap **K = 16**, configurable.
- v1 starts **bare-handed** (punch wood, MineRL TreeChop–style); axe variant is a later
  `StateMutator` option.
- Camera discretized into bins for the policy (continuous head optional).

## 13. References

Cloned to `reference/`: `rlgym`, `rlgym-ppo`, `rlgym-tools` (API + GPU PPO patterns,
primary model), `minerl`, `MineDojo`, `MineStudio` (env/obs/action precedents),
`azalea` (vanilla inventory click operations), `headlessmc`. User's `zurg-rs`
(`src/equipment.rs`) demonstrates azalea inventory/hotbar shuffling.
