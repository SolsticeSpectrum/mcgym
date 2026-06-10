# MCAI — what we're building

> Teach a neural network to play Minecraft from scratch with reinforcement learning, and run
> the trained policy as a **client-side Fabric mod** that works on *any* server without
> permissions. (`STATUS.md` is the current state + how to run; this doc is the *why* and *how*.)

---

## The goal

A model that *plays* Minecraft — starting with "gather as much wood as possible," building
toward navigation, PvP/PvE, parkour, and eventually beating the game. The deployed form is a
Fabric mod: you type `.run <weights>` in chat (swallowed client-side, never sent to the server)
and the policy drives your player via valid client actions. The Fabric mod is the **true
benchmark** — gym numbers don't count until the policy works in the real client.

## The core bet (what makes this different from MineRL et al.)

**No vision. Structured observations instead.** MineRL-style agents learn from raw pixels and
burn an enormous fraction of their samples just learning to *see*. We hand the policy the
structured world data directly — a grid of block ids, self-state, inventory — so every sample
goes into *behavior*, not perception. This is the single biggest reason it learns nav + mining
so fast. It's legitimate precisely because the deployed client has that same structured data,
so there's no privileged information the mod can't reconstruct.

Three consequences fall out of that bet:
- **Sample-efficient + fast** — no pixel encoder, plus a headless gym that sprints past realtime.
- **One shared data structure** ("the schema") used identically in training and in the mod, so a
  policy trained in the gym consumes the exact same bytes the mod produces.
- **Vanilla-faithful** — we train on *real* Minecraft physics/worldgen/mining (via a decompiled
  server), so behavior transfers. Sim-to-real fidelity is a deployment requirement, not a nicety.

## Architecture

```
 schema/                  ← the canonical observation/action byte contract (one source of truth)
   mcai_schema.yaml
 minecraft-decomp/        ← the GYM: decompiled, patched vanilla 1.21.11 server
   patches/gym/...        ·   headless ServerPlayers, driven directly, real physics/mining/worldgen
                          ·   shm + unix-socket transport to Python
 trainer/  (mcai_train)   ← PPO learner (PyTorch, GPU), reward shaping, ONNX export, web monitor
 fabric-mod/              ← deployment: runs the ONNX policy in the real client via valid actions
 docker/                  ← GPU + SSH image to scale training on a cloud box
 weights/                 ← exported ONNX policies
 reference/               ← cloned prior art (rlgym, minerl, MineDojo, ..., portable-lce)
```

**The shared schema is the keystone.** `schema/mcai_schema.yaml` defines a packed, little-endian
record (observation + action). The Java gym, the Python trainer, and the Fabric mod each encode
those exact bytes; a golden fixture (`schema/fixtures/`) is verified byte-identical across
languages. That's why the same policy runs in all three places, and why we can swap the gym
backend without touching the trainer or mod.

## How one training step works

```
 Python: read obs (N agents) ─► policy forward (GPU) ─► actions
        └─ shm/socket ─────────────────────────────────────────┐
 Gym (Java): apply actions to N real ServerPlayers ─► doTick()  │
             (vanilla move/collide/mine) ─► pack obs ───────────┘
 Python: compute reward (Δwood + shaping) ─► PPO update
```

Observations are **structured**:
- a near voxel grid (17³, radius 8) of block ids + a foveated **far** shell (stride-4, radius 32)
  so the agent sees and navigates toward distant trees,
- self-state (velocity, look, health/food, on-ground),
- the block under the crosshair (`target_block`) so it can tell a trunk from leaves,
- the 41-slot inventory.

The policy is a small (~0.6 M-param) actor-critic: block/item embeddings + 3D CNNs over the voxel
grids + scalar/inventory branches → multi-discrete heads (move/strafe/jump/sprint/look/attack).

## Design principles we hold

- **Vanilla-accurate, no shortcuts** — real physics, real mining timing (so learned dig durations
  are packet-valid), real worldgen. The gym *is* Minecraft, not an approximation.
- **Death is a wanted penalty.** Drowning/falling/freezing are real negative signals the agent
  must learn around — never engineered away.
- **Train on the real distribution.** Agents spawn in real forests far apart, gather *real*
  worldgen trees (no fake placed trees), and keep a **persistent home**: death respawns them in
  place to re-attempt the hazard (learn to swim, route around powder snow), and they're only
  relocated when a home is genuinely hopeless. Episodes don't teleport them, so they can make
  multi-episode journeys.
- **Smooth, human-like control** — CAPS-style camera-*jerk* penalties (punish oscillation, not
  turning) so it scans and focuses instead of spinning.
- **The mod stays an honest executor** — no aim-assists or crutches. If the policy fails in-world,
  we fix the *gym/training*, not the mod.

## What it does so far

From scratch, with shaping + curriculum (no human demonstrations), the policy learns to navigate
real terrain and chop wood. Emergent, *uncoded* behaviors appear as byproducts of the broad
objective + real hazards: terrain navigation, jumping/parkour, swimming, and staying out of
water — all instrumental skills the death penalty and gather objective induce.

## The honest hard parts

- **Sim-to-real is the truth test.** A policy can ace the gym via a proxy ("attack whatever's in
  range near a tree") and flop in-world the moment real foliage gets between it and the trunk. The
  fix is always a *more faithful gym* (real trees, `target_block`), never a mod-side patch.
- **Throughput is the wall.** The gym runs real per-agent physics, so it's CPU-bound; on a 6-core
  box that caps agents/throughput, and free-roaming agents in real terrain are RAM-hungry. This is
  why training is moving to a GPU box (see `docker/`).
- **Beating the game is unsolved** on any approach (DreamerV3 got diamonds, not the dragon, on
  serious compute). The plan there is tiered: a fast sim (or this gym) for foundational motor
  skills, a complete-game environment for long-horizon progression, sample-efficient methods, and
  real hardware.

## Roadmap

1. **Scale on the GPU box** (`docker/`) — more agents via parallel gyms, the RTX 6000 removes the
   PPO-update bottleneck; optional GraalVM `nativeServer` build for extra throughput.
2. **New task modules** — PvP (self-play + combat reward) and parkour (course spawn + traversal
   reward); the environment is ready, the task code is next.
3. **Deeper progression** — toward the "beat Minecraft" tiers, likely with world-model methods and
   a complete-game env (the Java decomp, or the cloned `portable-lce` Legacy Console Edition).
