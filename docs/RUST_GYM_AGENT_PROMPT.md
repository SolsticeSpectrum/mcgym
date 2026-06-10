# Build a Rust Minecraft RL Gym (drop-in replacement for the decompiled-vanilla gym)

You are building the **gym**: a headless, maximum-speed Minecraft environment in Rust that feeds a
PyTorch PPO trainer. You are **not** redesigning the learning algorithm, the model architecture, or
the deployment mod's *purpose*. But you have more latitude than a drop-in rewrite.

**Prime objective: maximize end-to-end training throughput, i.e. keep the GPU fed.** Today the
system is hard CPU-bound on the gym and the GPU sits ~90% idle (see §2). Your job is to flip that.
Raw gym TPS is the biggest lever, but **the data path matters too** — and you are explicitly
authorized to **redesign the observation schema and the gym↔trainer transport** if a better design
materially improves GPU utilization or throughput. The existing schema/transport (§4) is your
**baseline and reference**, not a cage.

The one hard invariant is the *system contract*, not the byte layout: the schema is the **single
source of truth shared across gym, trainer, and the deployment Fabric mod**. If you change it, it
must change *coherently* in all three (we will update the trainer codec and the mod codec to match
your new schema — propose it precisely), and it must still obey the legitimacy premise in §1 (no
information the real Minecraft client couldn't reconstruct). Don't churn the format gratuitously —
change it only for a real, measured throughput/GPU win, and document the change.

Read this whole document before writing code.

---

## 1. The project, so you understand *why*

MCAI trains a neural network to play Minecraft from scratch with RL, then runs the trained policy
as a **client-side Fabric mod** on any server. The bet that makes it work:

- **No vision.** The policy consumes *structured* observations (a grid of block ids, self-state,
  inventory, a raycast target, nearby entities) — never pixels. Every sample goes into behavior,
  not perception. This is legitimate because the deployed Minecraft client can reconstruct the
  exact same structured data, so the gym gives the policy nothing the mod can't.
- **One shared schema.** A single packed little-endian byte record (observation + action) is
  encoded identically by the gym, the trainer, and the mod. Swap the gym backend and nothing else
  changes — *that seam is exactly what you are plugging into.*
- **Vanilla fidelity is a hard requirement.** The Fabric mod running in the *real* client is the
  true benchmark; gym numbers don't count until the policy transfers. Any divergence between gym
  physics/mining/worldgen and real Minecraft poisons transfer. We learned this the hard way: a
  fast but unfaithful kinematic gym produced a policy that failed in-world.

Curriculum of goals: **gather wood** (current) → terrain navigation → PvE → PvP → parkour →
"beat the game." You only need wood-gathering mechanics correct *now*, but design observations and
actions to extend to the rest without a schema break.

## 2. Why we're replacing the current gym (your motivation, quantified)

The existing gym is a **decompiled, patched vanilla 1.21.11 server** driven headless. It is
faithful but **slow**: each world ticks at **~14 TPS (~71 ms/tick)** — *slower than real-time
vanilla's 20 TPS* — because every tick runs the full Java server plus heavy observation
extraction (~5,000 block lookups per agent per tick). It is hard CPU-bound; the GPU sits ~90%
idle. We get throughput only by cramming ~64 agents into one world and running a few worlds. The
"run the sim far faster than real-time, slow the policy down only in the mod" design **silently
failed** — the sim is the slow side.

**Your mandate: throw out the decompiled server entirely and build a Rust gym that runs the world
tick + observation extraction as fast as physically possible**, while staying vanilla-faithful.

## 3. Base it on Pumpkin

Use **Pumpkin** (`https://github.com/Pumpkin-MC/Pumpkin.git`) as the foundation — a Rust Minecraft
server that deliberately aims for 1:1 vanilla parity, often referencing decompiled vanilla code
and sacrificing performance to match behavior. That parity is precisely what we need; we will buy
the performance back by stripping everything a *training gym* doesn't need.

Approach (adapt as the code demands — you know Rust and Pumpkin's internals better than this doc):

- **Vendor/fork Pumpkin** and keep its world simulation: chunk storage, worldgen, block states,
  player movement & collision, fluid, block-breaking/mining timing, item drops & pickup,
  inventory. These must stay vanilla-accurate.
- **Rip out the real-client networking path.** A gym has no real TCP clients. Replace networked
  `Player`s with in-process driven agents you step directly. Keep just enough of the player model
  that movement/collision/mining/inventory behave vanilla.
- **Headless, uncapped, parallel.** No rendering, no 20-TPS sleep, no chat/keepalive/auth. Tick as
  fast as compute allows. Run **many agents per world** (they share one world tick) and **many
  worlds across cores** — that's where aggregate throughput comes from.
- **Combat is incomplete in Pumpkin — that's fine for now.** We train wood first; PvE/PvP come
  when those mechanics land. Don't block on them; just don't design them out.

"Race Pumpkin to maximum speed": profile the hot path (world tick + observation fill), cut
allocation per tick, exploit that the only consumers are N driven agents, and parallelize. Treat
TPS-per-world and total agent-steps/sec as the numbers you optimize.

---

## 4. The current contract — baseline & reference (redesignable per §0 prime objective)

This is the format the system speaks **today**. Treat it as your reference implementation and your
default: if you keep it, you get a drop-in gym for free, and the golden fixtures gate your
correctness. If you redesign it for throughput (§4.4), these files are still the precise spec of
*what information* the policy needs — you're changing the *layout/transport*, not deleting fields.

The authoritative source files in this repo (read them; do not paraphrase from memory):

- `schema/mcai_schema.yaml` — field list, dtypes, shapes, params. **Source of truth.**
- `schema/registry.json` — block-identifier→int and item-identifier→int maps. The gym must emit
  these exact integer ids (e.g. `minecraft:oak_log` → its registry int).
- `schema/fixtures/golden_obs.bin` / `golden_action.bin` (+ `.json`) — the cross-language byte
  contract. Your encoder/decoder **must** round-trip these byte-identically. This is your gate.
- `trainer/mcai_train/env/shm_transport.py` — the Python client (the wire protocol, authoritative).
- `trainer/mcai_train/env/gym_process.py` — how the gym subprocess is launched & detected ready.
- `minecraft-decomp/server/src/main/java/net/minecraft/mcai/` — the **reference gym behavior**:
  `McaiTransport.java` (transport), `McaiObsCodec.java` / `McaiActionCodec.java` (byte layout),
  `McaiGymRuntime.java` (curriculum, spawning, relocation, voxel/raycast fill), `McaiGymConfig.java`
  (world settings). Match its *behavior*; you are reimplementing it in Rust, faster.

### 4.1 Schema v1 — packed, little-endian, NO alignment padding

`schema_version = 1`. Params: `voxel_radius=8` (edge `2r+1=17`, `17³=4913` cells),
`voxel_far_stride=4` (→ radius 32), `max_entities=16`, `inventory_slots=41`,
`hidden_block_id=-1`.

**Critical footgun:** fields are written **sequentially with no alignment**. The `i64 tick` lands
at byte offset 4 (not 8-aligned). Do **not** use a naive `#[repr(C)]` struct or any auto-layout —
write each field little-endian in declared order with a sequential byte cursor (mirror how the
Java codec does `buf.putInt/putLong/...` and how NumPy builds the dtype with `align=False`).

**Observation record — `OBS_NBYTES = 40169`**, fields in this exact order:

| # | field | dtype | count | bytes |
|---|-------|-------|------:|------:|
| 1 | schema_version | i32 | 1 | 4 |
| 2 | tick | i64 | 1 | 8 |
| 3 | agent_id | i32 | 1 | 4 |
| 4 | pos | f32 | 3 | 12 |
| 5 | vel | f32 | 3 | 12 |
| 6 | yaw | f32 | 1 | 4 |
| 7 | pitch | f32 | 1 | 4 |
| 8 | on_ground | u8 | 1 | 1 |
| 9 | health | f32 | 1 | 4 |
| 10 | food | f32 | 1 | 4 |
| 11 | selected_slot | u8 | 1 | 1 |
| 12 | voxel_blocks | i32 | 4913 | 19652 |
| 13 | voxel_far | i32 | 4913 | 19652 |
| 14 | target_block | i32 | 1 | 4 |
| 15 | target_face | u8 | 1 | 1 |
| 16 | target_distance | f32 | 1 | 4 |
| 17 | target_in_range | u8 | 1 | 1 |
| 18 | entity_type_id | i32 | 16 | 64 |
| 19 | entity_rel_pos | f32 | 16×3 | 192 |
| 20 | entity_vel | f32 | 16×3 | 192 |
| 21 | entity_yaw | f32 | 16 | 64 |
| 22 | entity_health | f32 | 16 | 64 |
| 23 | entity_flags | u8 | 16 | 16 |
| 24 | inv_item_id | i32 | 41 | 164 |
| 25 | inv_count | u8 | 41 | 41 |

(Sum = 40169.) Multi-dim arrays are row-major (`entity_rel_pos[i]` = x,y,z contiguous).

**Action record — `ACTION_NBYTES = 27`**, in this exact order:

| # | field | dtype | meaning |
|---|-------|-------|---------|
| 1 | forward | f32 | move input, ~[-1,1] (+ = forward) |
| 2 | strafe | f32 | move input, ~[-1,1] (+ = right) |
| 3 | jump | u8 | 0/1 |
| 4 | sneak | u8 | 0/1 |
| 5 | sprint | u8 | 0/1 |
| 6 | yaw_delta | f32 | degrees added to yaw this tick |
| 7 | pitch_delta | f32 | degrees added to pitch this tick (clamp pitch to ±90) |
| 8 | attack | u8 | 0/1 — mine the targeted block / attack |
| 9 | use | u8 | 0/1 — use/place/interact |
| 10 | selected_slot | u8 | hotbar 0–8 |
| 11 | inv_op_type | u8 | inventory op (0 = none) |
| 12 | inv_slot_a | i16 | inventory op operand |
| 13 | inv_slot_b | i16 | inventory op operand |

`attack` semantics must match vanilla mining: holding attack on a log breaks it after the correct
hardness/tool time and drops the item, which the agent then walks over to pick up. Inventory ops
(`inv_op_type`/`slot_a`/`slot_b`) can be minimally stubbed for the wood task but keep the bytes.

### 4.2 Transport — shared memory + Unix-domain socket

One mmap'd file, all little-endian:

```
[0 .. 64)                              header
[64 .. 64 + N*27)                      ACTION region   (Python writes, gym reads)
[64 + N*27 .. + N*40169)               OBS region      (gym writes, Python reads)
```

Header (64 bytes): `magic i32 = 0x4D434149` ('MCAI'), `schema_version i32 = 1`, `n_agents i32`,
rest reserved/zero. The gym creates & maps the file at this exact size and validates/writes the
header; the Python client checks magic, schema_version, and n_agents on attach.

Stepping is a 1-byte command / 1-byte reply over the UDS, which paces the gym to the driver:

- Client sends one command byte: `RESET=1`, `STEP=2`, `CLOSE=3`.
- On `STEP`: read the ACTION region, apply one action per agent, advance the gym **one tick**, run
  per-agent curriculum/relocation logic, write the OBS region (one obs per agent), then reply one
  `OK=1` byte.
- On `RESET`: bring agents to their start state and write the initial OBS region, reply `OK`.
- On `CLOSE`: shut down cleanly.

The Python side splits send/recv so it fires `STEP` to all parallel gyms before collecting
replies (gyms tick concurrently) — your blocking read/reply per gym just needs to be correct.

### 4.3 Launch & readiness contract

The trainer launches the gym as a subprocess and waits for a readiness line on stdout. Expose a
clean CLI binary that accepts the run parameters and, **once shm is mapped and the socket is
bound and listening**, prints a line containing:

```
MCAI_TRANSPORT_READY
```

Parameters the trainer must pass (names can be CLI flags; the trainer's `gym_process.py` will be
updated to invoke your binary instead of `gradlew runGymTransport`, so propose a clean flag set):
`shm path`, `sock path`, `agents` (N), `seed`, `arena` (e.g. `wild`), `curriculum`. Fail fast with
a clear error if any required parameter is missing — **no silent fallbacks/defaults** (project
rule). First boot may be slow (worldgen); that's expected, the driver waits up to 180 s.

### 4.4 Redesigning the schema/transport for GPU utilization (encouraged, with guardrails)

The current path is: gym writes per-agent records (AoS) into shm → Python takes zero-copy NumPy
views → `obs_to_tensors` repacks/reshapes into tensors → policy forward on GPU. The transport is
already zero-copy on the NumPy side; the suspected throughput drags are (a) IPC + Python repacking
overhead per step, and (b) the gym simply not producing steps fast enough. You are free to attack
both. Options to weigh (pick what the profiler justifies — this is your call, not a mandate):

- **Layout for the consumer, not the producer.** Emit observations **batched, struct-of-arrays**
  (all agents' `voxel_blocks` contiguous, etc.) in the exact dtype/shape the policy wants, so the
  Python side does *zero* repacking — a single `torch.from_numpy`/`from_blob` per field. This can
  remove the `obs_to_tensors` copy entirely. (If you do this, hand us the matching trainer-side
  reader; we'll wire it in.)
- **Kill the IPC if it pays.** Consider running the gym **in-process** as a Python extension (PyO3
  / cffi) or writing obs directly into **pinned (page-locked) host buffers** the trainer can async
  `cudaMemcpy` from, so collected experience lands GPU-ready without a serialize→socket→deserialize
  hop. Keep the multi-process option viable too (it's how we scale across cores) — measure both.
- **Cheaper obs, GPU-side decode.** The voxel grids dominate (2×4913 ints/agent). Consider emitting
  a compact form (e.g. palette indices / packed bytes) and expanding to embeddings on the GPU,
  rather than shipping 40 KB/agent/step. Smaller payload → more agents per unit bandwidth.
- **Scale the batch.** Whatever maximizes GPU occupancy: more agents/world × more worlds, so each
  policy forward is a big batch. The transport must stay correct under high N.

**Guardrails on any redesign:**
1. It must be a *measured* win in GPU utilization or total agent-steps/sec — not a guess.
2. The schema stays a **single coherent source of truth**. Deliver the new spec in the same form
   as `schema/mcai_schema.yaml` (machine-readable), plus the matching encoder so we can update the
   trainer codec and the Fabric mod codec. A field that exists must be reproducible by the real
   Minecraft client (legitimacy premise, §1) — you may change *how* it's laid out, not smuggle in
   privileged info.
3. Keep the same *information content* the policy needs (§5) unless you can argue a field is
   redundant; don't silently drop signal.
4. Regenerate the golden fixtures for the new format and ship a round-trip test.

If the existing format is already near-optimal once the gym is fast, **keeping it is a fine
outcome** — the headline metric is GPU utilization / throughput, not novelty.

---

## 5. Observation semantics (behaviors, not just bytes)

Match the reference (`McaiObsCodec` + `McaiGymRuntime`):

- **self**: `pos` (x,y,z), `vel`, `yaw`, `pitch`, `on_ground`, `health`, `food`, `selected_slot`.
- **voxel_blocks**: agent-centered cube, radius 8 → 17×17×17 = 4913 cells, each cell = the
  registry int of the block at that world position (`minecraft:air`→0, etc.). Fixed cell ordering
  (match the reference's index order exactly — verify against golden / reference loop).
- **voxel_far**: same 4913-cell cube but **sampled every 4th block** (`voxel_far_stride=4`), giving
  a coarse radius-32 shell for distant terrain/navigation.
- **legit-mode occlusion**: blocks the real client could not observe (fully enclosed / not exposed
  to any visible face) are reported as `hidden_block_id = -1`, not their true id — the gym must not
  leak information the deployed mod wouldn't have. Reproduce the reference's visibility rule.
- **target_block / target_face / target_distance / target_in_range**: raycast from the eye along
  (yaw, pitch) up to block reach (~4.5). Report the hit block's registry id, which face, the
  distance, and whether it's within reach. This is how the policy knows trunk-vs-leaves and aim.
- **entities**: up to 16 nearest, each: `entity_type_id` (registry), `entity_rel_pos` (relative to
  agent), `entity_vel`, `entity_yaw`, `entity_health`, `entity_flags`. Empty slots zero-filled.
  (Mobs are off for the wood task — see §7 — but keep the fields live for PvE/PvP later.)
- **inventory**: 41 slots (36 main + 4 armor + 1 offhand), `inv_item_id` (registry) + `inv_count`.

## 6. Curriculum / spawning / relocation (match `McaiGymRuntime`)

- **Arena modes**: `FLAT` (flat platform), `WILD` (real worldgen, forest-seek spawn), `SPREAD`.
  The active task uses **WILD**.
- **WILD spawning**: agents are spread far apart so they don't share terrain — anchors on a coarse
  grid `WILD_SPACING=1024` blocks apart, spawn jittered within a `WILD_REGION=96` box around the
  anchor, biased to land near trees (`findWildSpawn`: scan radius `TREE_SCAN_RADIUS=24`, require
  `MIN_NEARBY_LOGS=4`, up to `WILD_SPAWN_ATTEMPTS=24` tries before fallback). Equip a starting axe.
- **Relocation philosophy (get this exactly right — it encodes a deliberate training choice):**
  - **Death is a wanted penalty.** On death, respawn at the agent's *persistent home*, so it
    re-attempts the hazard (e.g. drown repeatedly → learn to swim out). Never engineer death away.
  - **Episode reset does NOT teleport.** Agents persist across episodes for multi-episode journeys.
  - **Relocate to a fresh home ONLY when truly stuck** — i.e. an *alive* agent made no meaningful
    progress for `STUCK_GIVEUP=1500` steps, where "progress" = moved `MOVE_PROGRESS=3` blocks from
    its reference point, OR mined something, OR picked something up. **Dying a lot is NOT stuck.**
    Stuck means the home is hopeless (no progress possible from there); only then pick a new home.
- Keep these as named constants; the reward shaping lives in the trainer, not the gym.

## 7. World configuration (bound per-tick cost)

Match `McaiGymConfig`: `view-distance=2`, `simulation-distance=1`, **mobs/animals/NPCs off**
(`spawn-monsters/animals/npcs=false`). These keep ticking/memory bounded so per-tick cost stays in
the world+obs hot path, not in mob AI. (When PvE/PvP arrive, these get re-enabled selectively.)

## 8. Performance targets & acceptance criteria

**Correctness (gate — must pass before perf matters):**
1. **If you keep the schema (§4):** encode/decode the golden fixtures
   (`schema/fixtures/golden_obs.bin`, `golden_action.bin`) **byte-identical**, with a Rust test
   asserting it. **If you redesign it (§4.4):** ship the new machine-readable spec + encoder,
   regenerate the golden fixtures, and provide a round-trip test against the updated trainer codec.
2. The unmodified Python trainer drives your gym end-to-end: attach to shm, `RESET`, run many
   `STEP`s with real actions, read sane observations (block ids match registry, raycast finds a
   log when aimed at one, mining a log increments inventory). The trainer's `gym_process.py` will
   be repointed at your binary — coordinate that flag set, but the wire protocol stays as in §4.2.
3. Behavior is vanilla-faithful: movement/collision, mining times, item drops/pickup, worldgen,
   fluid, fall damage/drowning all match vanilla 1.21.x within reason. Spot-check against the
   reference gym's behavior.

**Performance (the point of the rewrite):**
- Dramatically beat the decomp gym's **~14 TPS/world**. Target **hundreds of TPS per headless
  world** with one agent, and high aggregate **agent-steps/sec** with many agents/world × many
  worlds across cores. Report TPS/world, agents/world scaling, and total steps/sec.
- Per-tick allocation in the obs/voxel path should be near-zero (reuse buffers). The voxel fill
  (2×4913 block lookups/agent) is the prime hot spot — make block-state lookup and the
  legit-mode visibility check cheap.
- No artificial tick cap; pace only by the transport handshake.

**Report** (so we can wire it into training): the exact CLI, the readiness line, TPS/world vs
agent count, total agent-steps/sec on an N-core box, and any behavior gaps vs vanilla (especially
anything that could hurt sim-to-real transfer).

## 9. Explicit non-goals / boundaries

- You **may** redesign the schema, wire protocol, and obs layout for throughput/GPU utilization
  (§4.4) — but only as a coherent, documented change delivered with a new spec + encoder so the
  trainer and the deployed mod can be updated in lockstep. Don't change it gratuitously, don't
  break the legitimacy premise (no privileged info), and don't drop information the policy needs.
  Changing it invalidates existing trained weights — that's an accepted cost *only* for a measured
  throughput win, not a stylistic one.
- **Do not** build the trainer, the PPO loop, the model, reward shaping, or the Fabric mod.
- **Do not** add information to observations that a real Minecraft client couldn't reconstruct
  (that's the whole legitimacy premise — respect legit-mode occlusion).
- **Do not** "fix" death by preventing it, and don't teleport on episode reset.
- Combat/PvP/PvE mechanics being incomplete in Pumpkin is acceptable; don't block on them, don't
  design them out.

Deliver a fast, vanilla-faithful Rust gym that the existing trainer can drive with a one-line
launch change and zero protocol change. The golden test passing + the trainer collecting sane wood
episodes at many× the current TPS is success.
