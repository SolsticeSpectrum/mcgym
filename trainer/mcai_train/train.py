"""PPO training entrypoint for the gather-wood Minecraft task."""
from __future__ import annotations

import argparse
import copy
import os
import json
import pathlib
import threading
import time

import numpy as np
import torch

from mcai_train import checkpoint
from mcai_train.env.wood_env import ParallelVecEnv, WoodEnv
from mcai_train.models.policy import ActorCritic, obs_to_tensors
from mcai_train.ppo.buffer import RolloutBuffer
from mcai_train.ppo.learner import PPOLearner
from mcai_train.schema import spec
from mcai_train.schema.registry import Registry

# Repo-relative so it works regardless of checkout location (train.py -> mcai_train -> trainer -> repo root).
REGISTRY_PATH = pathlib.Path(__file__).resolve().parents[2] / "schema" / "registry.json"


def _model_sizes(registry_path: pathlib.Path) -> tuple[int, int]:
    """Embedding table sizes = max id + 1 (+1 pad) for blocks and items."""
    doc = json.loads(registry_path.read_text())
    num_blocks = max(doc.get("blocks", {}).values()) + 2
    num_items = max(doc.get("items", {}).values()) + 2
    return num_blocks, num_items


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO trainer for MCAI gather-wood")
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--rollout-len", type=int, default=128)
    p.add_argument("--episode-len", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--run-name", default="gather_wood")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--checkpoint-every", type=int, default=20_000)
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument(
        "--model-scale",
        type=int,
        default=1,
        help="width multiplier for the policy/value net (1 = 646k-param default tuned for a "
        "GTX 1060; raise to ~4-8 on a big GPU for more capacity + GPU work per forward). Must "
        "match the checkpoint when resuming.",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--pipeline",
        action="store_true",
        help="overlap the PPO update (GPU) with the next rollout's collect (CPU gyms) via a "
        "background-thread update + a snapshot policy for collecting; bounded 1-rollout staleness",
    )
    p.add_argument(
        "--async-collect",
        action="store_true",
        help="2-cohort pipelined collection: split gyms into cohorts A/B and overlap A's gym "
        "ticks (CPU/cores) with B's policy forward (GPU) and vice versa, so neither waits. "
        "Combined with the background-thread update, keeps both the cores and the GPU busy. "
        "Requires --num-envs >= 2 (even split). Bounded 1-rollout policy staleness.",
    )
    p.add_argument("--curriculum", default="", help="gym curriculum, e.g. 'tree_ahead'")
    p.add_argument("--arena", default="", help="gym arena mode, e.g. 'flat'")
    p.add_argument("--num-envs", type=int, default=1,
                   help="parallel gym processes; total agents = num_envs * n_agents")
    p.add_argument("--monitor-port", type=int, default=0,
                   help="if >0, serve a top-down web view of agents on this port")
    return p.parse_args(argv)


def resolve_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    # TF32 for fp32 matmuls/convs (tensor cores at fp32 interface, ~1e-3 relative precision —
    # fine for policy nets). Speeds the eager fp32 collect forward and any non-autocast math.
    torch.set_float32_matmul_precision("high")
    ckpt_dir = args.checkpoint_dir or f"runs/{args.run_name}"

    registry = Registry.load(REGISTRY_PATH)
    num_blocks, num_items = _model_sizes(REGISTRY_PATH)
    print(f"[train] device={device} num_blocks={num_blocks} num_items={num_items}")

    model = ActorCritic(num_blocks, num_items, scale=args.model_scale).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params:,} (scale={args.model_scale})")

    learner = PPOLearner(
        model,
        lr=args.lr,
        epochs=args.epochs,
        minibatch=args.minibatch,
        device=device,
    )

    cumulative_timesteps = 0
    if args.resume:
        meta = checkpoint.load_latest(ckpt_dir, model, learner.optimizer)
        if meta is not None:
            cumulative_timesteps = int(meta["cumulative_timesteps"])
            print(f"[train] resumed at {cumulative_timesteps} timesteps")

    # --async-collect splits the gyms into two cohorts (A/B) that step offset so one's gym ticks
    # overlap the other's policy forward. Otherwise a single (vec-)env stepped in lockstep.
    env = None
    cohorts = None
    if args.async_collect:
        if args.num_envs < 2:
            raise SystemExit("--async-collect requires --num-envs >= 2")
        ka = args.num_envs // 2
        kb = args.num_envs - ka
        envA = ParallelVecEnv(ka, args.n_agents, args.seed, registry,
                              episode_len=args.episode_len, curriculum=args.curriculum,
                              arena=args.arena)
        envB = ParallelVecEnv(kb, args.n_agents, args.seed + 10_000, registry,
                              episode_len=args.episode_len, curriculum=args.curriculum,
                              arena=args.arena)
        cohorts = [envA, envB]
        n = envA.n_agents + envB.n_agents
    elif args.num_envs > 1:
        env = ParallelVecEnv(args.num_envs, args.n_agents, args.seed, registry,
                             episode_len=args.episode_len, curriculum=args.curriculum,
                             arena=args.arena)
        n = env.n_agents
    else:
        env = WoodEnv(args.n_agents, args.seed, registry, episode_len=args.episode_len,
                      curriculum=args.curriculum, arena=args.arena)
        n = env.n_agents
    buffer = RolloutBuffer(args.rollout_len, n)

    monitor = None
    if args.monitor_port:
        from .monitor import TrainMonitor
        monitor = TrainMonitor(args.monitor_port, registry, n, n_per=args.n_agents)
        print(f"[train] web monitor at {monitor.start()}")

    hyperparams = {
        "lr": args.lr,
        "rollout_len": args.rollout_len,
        "episode_len": args.episode_len,
        "n_agents": args.n_agents,
        "minibatch": args.minibatch,
        "epochs": args.epochs,
    }

    def make_meta() -> dict:
        return {
            "cumulative_timesteps": cumulative_timesteps,
            "schema_version": spec.SCHEMA_VERSION,
            "hyperparams": hyperparams,
            "num_blocks": num_blocks,
            "num_items": num_items,
        }

    # Async manages its own per-cohort obs/ep tracking inside run_async().
    obs_struct = env.reset() if env is not None else None
    ep_reward = np.zeros(n, dtype=np.float64)
    # Episodes (len 500) usually span multiple rollouts (len 128), so most
    # rollouts complete no episode. Carry the mean episode return across
    # rollouts as an EMA so ep_rew is reported every line and never NaN.
    EP_EMA_BETA = 0.9
    ep_rew_ema = None
    last_ckpt = cumulative_timesteps
    start = time.monotonic()

    def collect_rollout(cmodel, buf):
        """Collect one rollout into `buf` using `cmodel` for action selection. Returns the list of
        episode returns that completed during the rollout. Threads `obs_struct`/`ep_reward`."""
        nonlocal obs_struct, ep_reward
        buf.reset()
        completed = []
        for _ in range(args.rollout_len):
            with torch.no_grad():
                tensors = obs_to_tensors(obs_struct, device)
                action_idx, logprob, value = cmodel.get_action(tensors)
            action_np = action_idx.cpu().numpy()
            next_obs, reward, done = env.step(action_np)
            buf.add(obs_struct, action_np, logprob.cpu().numpy(), reward, value.cpu().numpy(), done)
            ep_reward += reward
            if done.all():
                completed.extend(ep_reward.tolist())
                ep_reward = np.zeros(n, dtype=np.float64)
            obs_struct = next_obs
            if monitor is not None:
                monitor.update(next_obs, action_np, reward, cumulative_timesteps)
        with torch.no_grad():
            last_value = cmodel.get_action(obs_to_tensors(obs_struct, device))[2]
        buf.compute_gae(last_value.cpu().numpy())
        return completed

    def log_iter(metrics, completed, collect_secs, update_secs, wood_mean, ep_fallback):
        nonlocal ep_rew_ema, cumulative_timesteps, last_ckpt
        cumulative_timesteps += args.rollout_len * n
        sps = (args.rollout_len * n) / max(collect_secs + update_secs, 1e-6)
        if completed:
            rollout_mean = float(np.mean(completed))
            ep_rew_ema = (
                rollout_mean
                if ep_rew_ema is None
                else EP_EMA_BETA * ep_rew_ema + (1.0 - EP_EMA_BETA) * rollout_mean
            )
        mean_ep_r = ep_rew_ema if ep_rew_ema is not None else ep_fallback
        print(
            f"[train] t={cumulative_timesteps} "
            f"ep_rew={mean_ep_r:.3f} wood/agent={wood_mean:.2f} "
            f"pi_loss={metrics['policy_loss']:.4f} v_loss={metrics['value_loss']:.4f} "
            f"ent={metrics['entropy']:.3f} clip={metrics['clip_frac']:.3f} "
            f"sps={sps:.0f} collect={collect_secs:.1f}s update={update_secs:.1f}s"
        )
        if cumulative_timesteps - last_ckpt >= args.checkpoint_every:
            checkpoint.save(ckpt_dir, model, learner.optimizer, make_meta())
            last_ckpt = cumulative_timesteps
            print(f"[train] checkpoint @ {cumulative_timesteps}")

    def wood_mean_of(obs_batch):
        return float(env.wood_held(obs_batch).mean())

    def run_async():
        # Two cohorts step offset: while A's gyms tick (CPU/cores), B's policy forward runs (GPU),
        # then vice versa — neither waits on the other. The PPO update runs on a background thread
        # (overlapping the next collect). Bounded 1-rollout policy staleness.
        envA, envB = cohorts
        bufs = [RolloutBuffer(args.rollout_len, n), RolloutBuffer(args.rollout_len, n)]
        inf = [copy.deepcopy(model).eval()]
        obsA, obsB = envA.reset(), envB.reset()
        epr = np.zeros(n, dtype=np.float64)
        result: dict = {}

        def run_update(b):
            result["m"] = learner.update(b)

        # Collect forwards run on a dedicated CUDA stream so they don't serialize behind the
        # background update's kernels (default stream). Different modules + disjoint data, so the
        # GPU can run both streams concurrently; the .cpu() inside the stream context blocks the
        # host until the forward lands, which also enforces correct ordering (no cross-stream race).
        collect_stream = torch.cuda.Stream() if str(device).startswith("cuda") else None

        # Compile the collect forward too (static 576-agent batch): the eager small-batch
        # forward is launch-overhead-bound. Compiling the bound method keeps load_state_dict
        # weight syncs working — the compiled code reads the module's current params each call.
        get_action = inf[0].get_action
        if collect_stream is not None and os.environ.get("MCAI_COMPILE"):
            get_action = torch.compile(inf[0].get_action)

        def fwd(m, obs):
            with torch.no_grad():
                if collect_stream is not None:
                    with torch.cuda.stream(collect_stream):
                        a, lp, v = get_action(obs_to_tensors(obs, device))
                        return a.cpu().numpy(), lp.cpu().numpy(), v.cpu().numpy()
                a, lp, v = get_action(obs_to_tensors(obs, device))
            return a.cpu().numpy(), lp.cpu().numpy(), v.cpu().numpy()

        prof_on = bool(os.environ.get("MCAI_PROFILE"))

        def collect(buf):
            nonlocal obsA, obsB, epr
            buf.reset()
            completed = []
            m = inf[0]
            pf = ps = pr = pp = 0.0
            for _ in range(args.rollout_len):
                t = time.perf_counter()
                aA, lpA, vA = fwd(m, obsA)
                envA.step_send(aA)              # A's gyms tick (CPU) ...
                aB, lpB, vB = fwd(m, obsB)      # ... while B's forward runs (GPU)
                envB.step_send(aB)
                pf += time.perf_counter() - t; t = time.perf_counter()
                oA, rA, dA = envA.step_recv()
                oB, rB, dB = envB.step_recv()
                pr += time.perf_counter() - t; t = time.perf_counter()
                # Concatenate each field once and reuse (was concatenating obs+act twice/step).
                obs_cat = np.concatenate([obsA, obsB])
                act_cat = np.concatenate([aA, aB])
                rew = np.concatenate([rA, rB])
                buf.add(obs_cat, act_cat, np.concatenate([lpA, lpB]), rew,
                        np.concatenate([vA, vB]), np.concatenate([dA, dB]))
                epr += rew
                if dA.all() and dB.all():
                    completed.extend(epr.tolist())
                    epr = np.zeros(n, dtype=np.float64)
                obsA, obsB = oA, oB
                if monitor is not None:
                    monitor.update(obs_cat, act_cat, rew, cumulative_timesteps)
                pp += time.perf_counter() - t
            _, _, lvA = fwd(m, obsA)
            _, _, lvB = fwd(m, obsB)
            buf.compute_gae(np.concatenate([lvA, lvB]))
            if prof_on:
                print(f"[profile] fwd+send={pf:.1f}s recv={pr:.1f}s post={pp:.1f}s", flush=True)
            return completed

        def wood_mean():
            return float(np.concatenate([envA.wood_held(obsA), envB.wood_held(obsB)]).mean())

        cur = 0
        completed_prev = collect(bufs[cur])  # prime (also compiles the collect forward, serially)
        first_update = True
        while cumulative_timesteps < args.total_timesteps:
            t0 = time.monotonic()
            th = threading.Thread(target=run_update, args=(bufs[cur],), daemon=True)
            th.start()
            if first_update:
                # Dynamo tracing is not thread-safe against executing another compiled function:
                # the first update compiles evaluate (fwd+bwd) — let it finish before collecting
                # concurrently. One serialized update, then full overlap.
                th.join()
                first_update = False
            nxt = 1 - cur
            completed = collect(bufs[nxt])      # collect (cores+GPU overlap) || update (GPU)
            th.join()
            # Sync the collect policy to the updated weights. Safe here: the collector is idle
            # (between rollouts). load_state_dict reuses the inference model (no per-rollout realloc).
            inf[0].load_state_dict(model.state_dict())
            log_iter(result["m"], completed_prev, time.monotonic() - t0, 0.0,
                     wood_mean(), float(epr.mean()))
            completed_prev = completed
            cur = nxt

    try:
        if cohorts is not None:
            run_async()
        elif args.pipeline:
            # Overlap the update (GPU) with the next collect (CPU gyms). A separate snapshot policy
            # collects while the live model trains on a background thread — disjoint modules, so no
            # races; the collecting policy is one rollout stale (PPO tolerates it).
            collect_model = copy.deepcopy(model)
            collect_model.eval()
            bufs = [buffer, RolloutBuffer(args.rollout_len, n)]
            cur = 0
            completed_prev = collect_rollout(model, bufs[cur])  # prime
            result: dict = {}

            def run_update(b):
                result["m"] = learner.update(b)

            while cumulative_timesteps < args.total_timesteps:
                t0 = time.monotonic()
                collect_model.load_state_dict(model.state_dict())  # pre-update snapshot
                th = threading.Thread(target=run_update, args=(bufs[cur],), daemon=True)
                th.start()
                nxt = 1 - cur
                completed = collect_rollout(collect_model, bufs[nxt])
                th.join()
                overlapped = time.monotonic() - t0
                log_iter(result["m"], completed_prev, overlapped, 0.0,
                         wood_mean_of(obs_struct), float(ep_reward.mean()))
                completed_prev = completed
                cur = nxt
        else:
            while cumulative_timesteps < args.total_timesteps:
                rollout_start = time.monotonic()
                completed = collect_rollout(model, buffer)
                collect_end = time.monotonic()
                metrics = learner.update(buffer)
                update_secs = time.monotonic() - collect_end
                log_iter(metrics, completed, collect_end - rollout_start, update_secs,
                         wood_mean_of(obs_struct), float(ep_reward.mean()))
    except KeyboardInterrupt:
        print("[train] interrupted; saving final checkpoint")
    finally:
        checkpoint.save(ckpt_dir, model, learner.optimizer, make_meta())
        if cohorts is not None:
            for c in cohorts:
                c.close()
        elif env is not None:
            env.close()
        total_elapsed = time.monotonic() - start
        print(f"[train] done: {cumulative_timesteps} timesteps in {total_elapsed:.1f}s")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
