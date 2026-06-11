"""ppo training entrypoint, pick a task with --task"""
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

from mcgym import checkpoint
from mcgym.env.env import Env, VecEnv
from mcgym.models.policy import ActorCritic, tensors
from mcgym.ppo.buffer import Buffer
from mcgym.ppo.learner import Learner
from mcgym.schema import spec
from mcgym.schema.registry import Registry
from mcgym.tasks import parkour, wood

TASKS = {"wood": wood.Wood, "parkour": parkour.Parkour}

REGISTRY = pathlib.Path(__file__).resolve().parents[2] / "schema" / "registry.json"
EMA = 0.9


def sizes(path: pathlib.Path) -> tuple[int, int]:
    # embedding table sizes, max id + 1 + pad
    doc = json.loads(path.read_text())
    return max(doc["blocks"].values()) + 2, max(doc["items"].values()) + 2


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mcgym PPO trainer")
    p.add_argument("--task", default="wood", choices=sorted(TASKS))
    p.add_argument("--agents", type=int, default=4, help="agents per gym")
    p.add_argument("--num-envs", type=int, default=1, help="parallel gym processes")
    p.add_argument("--total", type=int, default=1_000_000)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--run", default="wood")
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--ckpt-every", type=int, default=20_000)
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--scale", type=int, default=1, help="model width multiplier, must match checkpoint")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--pipeline", action="store_true",
                   help="overlap the update with the next collect on a background thread")
    p.add_argument("--cohorts", type=int, default=0,
                   help="K cohort async collect, gym ticks overlap other cohorts forwards, needs num-envs >= K")
    p.add_argument("--monitor", type=int, default=0, help="web monitor port")
    
    return p.parse_args(argv)


def device_of(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def train(args: argparse.Namespace) -> None:
    device = device_of(args.device)
    ckpt_dir = args.ckpt_dir or f"runs/{args.run}"

    # tf32 for fp32 matmuls, fine for policy nets
    torch.set_float32_matmul_precision("high")

    registry = Registry.load(REGISTRY)
    blocks, items = sizes(REGISTRY)
    print(f"[train] device={device} blocks={blocks} items={items}")

    model = ActorCritic(blocks, items, scale=args.scale).to(device)
    print(f"[train] model params {sum(p.numel() for p in model.parameters()):,} (scale {args.scale})")

    learner = Learner(model, lr=args.lr, epochs=args.epochs, minibatch=args.minibatch, device=device)

    steps = 0
    if args.resume:
        meta = checkpoint.load_latest(ckpt_dir, model, learner.optimizer)
        if meta is not None:
            steps = int(meta["steps"])
            print(f"[train] resumed at {steps} steps")

    def task(agents):
        return TASKS[args.task](agents, registry)

    env = None
    cohorts = None
    if args.cohorts > 1:
        k = args.cohorts
        if args.num_envs < k:
            raise SystemExit(f"--cohorts {k} needs --num-envs >= {k}")
        
        per = [args.num_envs // k + (1 if i < args.num_envs % k else 0) for i in range(k)]
        cohorts = [VecEnv(sz, args.agents, args.seed + 10_000 * i, task) for i, sz in enumerate(per)]
        n = sum(c.agents for c in cohorts)

        def metric(obs):
            out = []
            off = 0
            for c in cohorts:
                out.append(c.metric(obs[off:off + c.agents]))
                off += c.agents
                
            return np.concatenate(out)
    elif args.num_envs > 1:
        env = VecEnv(args.num_envs, args.agents, args.seed, task)
        n = env.agents
        
        metric = env.metric
    else:
        env = Env(args.agents, args.seed, task(args.agents))
        n = env.agents
        
        metric = env.metric
        
    buf = Buffer(args.rollout, n)

    monitor = None
    if args.monitor:
        from .monitor import Monitor
        monitor = Monitor(args.monitor, n, per=args.agents, metric=metric)
        print(f"[train] web monitor at {monitor.start()}")

    def meta() -> dict:
        return {
            "steps":          steps,
            "task":           args.task,
            "schema_version": spec.SCHEMA_VERSION,
            "hyperparams":    {"lr": args.lr, "rollout": args.rollout, "agents": args.agents,
                               "minibatch": args.minibatch, "epochs": args.epochs, "scale": args.scale},
            "blocks":         blocks,
            "items":          items,
        }

    obs = env.reset() if env is not None else None
    epr = np.zeros(n, dtype=np.float64)
    
    # episodes span rollouts so carry the mean episode return as an ema
    ema       = None
    last_ckpt = steps
    start     = time.monotonic()

    def rollout(cmodel, buf):
        # one rollout into buf, returns episode returns that completed
        nonlocal obs, epr
        buf.reset()
        
        completed = []
        for _ in range(args.rollout):
            with torch.no_grad():
                a, lp, v = cmodel.get_action(tensors(obs, device))
            a = a.cpu().numpy()

            nxt, rew, done = env.step(a)
            buf.add(obs, a, lp.cpu().numpy(), rew, v.cpu().numpy(), done)

            epr += rew
            if done.all():
                completed.extend(epr.tolist())
                epr = np.zeros(n, dtype=np.float64)

            obs = nxt
            if monitor is not None:
                monitor.update(nxt, a, rew, steps)

        with torch.no_grad():
            last = cmodel.get_action(tensors(obs, device))[2]
            
        buf.gae(last.cpu().numpy())
        return completed

    def log(metrics, completed, secs, score, fallback):
        nonlocal ema, steps, last_ckpt
        steps += args.rollout * n
        sps = (args.rollout * n) / max(secs, 1e-6)

        if completed:
            mean = float(np.mean(completed))
            ema = mean if ema is None else EMA * ema + (1.0 - EMA) * mean
            
        epm = ema if ema is not None else fallback

        print(f"[train] t={steps} ep_rew={epm:.3f} score={score:.2f} "
              f"pi={metrics['policy_loss']:.4f} v={metrics['value_loss']:.4f} "
              f"ent={metrics['entropy']:.3f} clip={metrics['clip_frac']:.3f} "
              f"sps={sps:.0f} collect={secs:.1f}s")
        
        if steps - last_ckpt >= args.ckpt_every:
            checkpoint.save(ckpt_dir, model, learner.optimizer, meta())
            last_ckpt = steps
            print(f"[train] checkpoint @ {steps}")

    def run_async():
        # K cohorts stepped offset, each cohorts gym tick overlaps the other
        # cohorts forwards on the gpu, update runs on a background thread,
        # bounded one rollout staleness
        nonlocal epr
        bufs   = [Buffer(args.rollout, n), Buffer(args.rollout, n)]
        inf    = copy.deepcopy(model).eval()
        cobs   = [c.reset() for c in cohorts]
        result: dict = {}

        def update(b):
            result["m"] = learner.update(b)

        # high priority stream so the small collect forwards dont queue behind
        # the big update kernels on the default stream
        stream = torch.cuda.Stream(priority=-1) if device.startswith("cuda") else None

        # compiling the bound method keeps load_state_dict weight syncs working
        get = inf.get_action
        if stream is not None and os.environ.get("MCGYM_COMPILE"):
            get = torch.compile(inf.get_action)
            
        # collect in the same precision as the updates recompute, a mismatch
        # shows up as ratio noise and inflates clip_frac
        bf16 = stream is not None and bool(os.environ.get("MCGYM_BF16"))

        def fwd(o):
            with torch.no_grad():
                if stream is not None:
                    with torch.cuda.stream(stream):
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                            a, lp, v = get(tensors(o, device))
                        return a.cpu().numpy(), lp.float().cpu().numpy(), v.float().cpu().numpy()
                a, lp, v = get(tensors(o, device))
            return a.cpu().numpy(), lp.cpu().numpy(), v.cpu().numpy()

        prof = bool(os.environ.get("MCGYM_PROFILE"))

        def collect(buf):
            nonlocal cobs, epr
            buf.reset()
            completed = []
            pf = pr = pp = 0.0
            for _ in range(args.rollout):
                t = time.perf_counter()
                acts, lps, vals = [], [], []
                for ci, c in enumerate(cohorts):
                    a, lp, v = fwd(cobs[ci])  # this cohorts forward
                    c.send(a)                 # its gyms tick while the rest forward
                    acts.append(a); lps.append(lp);
                    vals.append(v)
                    
                pf += time.perf_counter() - t;
                t = time.perf_counter()

                res = [c.recv() for c in cohorts]
                pr += time.perf_counter() - t; 
                t = time.perf_counter()

                ocat = np.concatenate(cobs)
                acat = np.concatenate(acts)
                rew  = np.concatenate([r[1] for r in res])
                buf.add(ocat, acat, np.concatenate(lps), rew,
                        np.concatenate(vals), np.concatenate([r[2] for r in res]))

                epr += rew
                if all(r[2].all() for r in res):
                    completed.extend(epr.tolist())
                    epr = np.zeros(n, dtype=np.float64)

                cobs = [r[0] for r in res]
                if monitor is not None:
                    monitor.update(ocat, acat, rew, steps)
                pp += time.perf_counter() - t

            buf.gae(np.concatenate([fwd(o)[2] for o in cobs]))
            if prof:
                print(f"[profile] fwd+send={pf:.1f}s recv={pr:.1f}s post={pp:.1f}s", flush=True)
            return completed

        def score():
            return float(np.concatenate([c.metric(o) for c, o in zip(cohorts, cobs)]).mean())

        cur = 0
        prev = collect(bufs[cur])  # prime, also compiles the collect forward serially
        first = True
        while steps < args.total:
            t0 = time.monotonic()
            th = threading.Thread(target=update, args=(bufs[cur],), daemon=True)
            th.start()
            if first:
                # dynamo tracing is not thread safe against running another
                # compiled function, let the first update compile alone
                th.join()
                first = False
                
            nxt = 1 - cur
            completed = collect(bufs[nxt])
            th.join()
            
            # sync collect weights, safe between rollouts
            inf.load_state_dict(model.state_dict())
            log(result["m"], prev, time.monotonic() - t0, score(), float(epr.mean()))
            prev = completed
            cur  = nxt

    try:
        if cohorts is not None:
            run_async()
        elif args.pipeline:
            # snapshot policy collects while the live model trains on a thread
            snap = copy.deepcopy(model).eval()
            bufs = [buf, Buffer(args.rollout, n)]
            cur  = 0
            prev = rollout(model, bufs[cur])
            result: dict = {}

            def update(b):
                result["m"] = learner.update(b)

            while steps < args.total:
                t0  = time.monotonic()
                snap.load_state_dict(model.state_dict())
                th  = threading.Thread(target=update, args=(bufs[cur],), daemon=True)
                th.start()
                nxt = 1 - cur
                completed = rollout(snap, bufs[nxt])
                th.join()
                
                log(result["m"], prev, time.monotonic() - t0, float(metric(obs).mean()), float(epr.mean()))
                prev = completed
                cur = nxt
        else:
            while steps < args.total:
                t0 = time.monotonic()
                completed = rollout(model, buf)
                
                metrics = learner.update(buf)
                log(metrics, completed, time.monotonic() - t0, float(metric(obs).mean()), float(epr.mean()))
    except KeyboardInterrupt:
        print("[train] interrupted, saving final checkpoint")
    finally:
        checkpoint.save(ckpt_dir, model, learner.optimizer, meta())
        for c in cohorts or []:
            c.close()
            
        if env is not None:
            env.close()
        print(f"[train] done, {steps} steps in {time.monotonic() - start:.1f}s")


if __name__ == "__main__":
    train(parse())
