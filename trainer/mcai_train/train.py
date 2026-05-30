"""PPO training entrypoint for the gather-wood Minecraft task."""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from mcai_train import checkpoint
from mcai_train.env.wood_env import WoodEnv
from mcai_train.models.policy import ActorCritic, obs_to_tensors
from mcai_train.ppo.buffer import RolloutBuffer
from mcai_train.ppo.learner import PPOLearner
from mcai_train.schema import spec
from mcai_train.schema.registry import Registry

REGISTRY_PATH = pathlib.Path("/home/user/github/mcai/schema/registry.json")


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
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def resolve_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    ckpt_dir = args.checkpoint_dir or f"runs/{args.run_name}"

    registry = Registry.load(REGISTRY_PATH)
    num_blocks, num_items = _model_sizes(REGISTRY_PATH)
    print(f"[train] device={device} num_blocks={num_blocks} num_items={num_items}")

    model = ActorCritic(num_blocks, num_items).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params:,}")

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

    env = WoodEnv(args.n_agents, args.seed, registry, episode_len=args.episode_len)
    buffer = RolloutBuffer(args.rollout_len, args.n_agents)

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

    obs_struct = env.reset()
    ep_reward = np.zeros(args.n_agents, dtype=np.float64)
    last_ckpt = cumulative_timesteps
    start = time.monotonic()

    try:
        while cumulative_timesteps < args.total_timesteps:
            buffer.reset()
            rollout_start = time.monotonic()
            completed_ep_rewards = []

            for _ in range(args.rollout_len):
                with torch.no_grad():
                    tensors = obs_to_tensors(obs_struct, device)
                    action_idx, logprob, value = model.get_action(tensors)
                action_np = action_idx.cpu().numpy()

                next_obs, reward, done = env.step(action_np)
                buffer.add(
                    obs_struct,
                    action_np,
                    logprob.cpu().numpy(),
                    reward,
                    value.cpu().numpy(),
                    done,
                )
                ep_reward += reward
                if done.all():
                    completed_ep_rewards.extend(ep_reward.tolist())
                    ep_reward = np.zeros(args.n_agents, dtype=np.float64)
                obs_struct = next_obs

            with torch.no_grad():
                last_value = model.get_action(obs_to_tensors(obs_struct, device))[2]
            buffer.compute_gae(last_value.cpu().numpy())

            metrics = learner.update(buffer)
            cumulative_timesteps += args.rollout_len * args.n_agents

            wood = env.wood_held(obs_struct)
            elapsed = time.monotonic() - rollout_start
            sps = (args.rollout_len * args.n_agents) / max(elapsed, 1e-6)
            mean_ep_r = (
                float(np.mean(completed_ep_rewards)) if completed_ep_rewards else float("nan")
            )
            print(
                f"[train] t={cumulative_timesteps} "
                f"ep_rew={mean_ep_r:.3f} wood/agent={wood.mean():.2f} "
                f"pi_loss={metrics['policy_loss']:.4f} v_loss={metrics['value_loss']:.4f} "
                f"ent={metrics['entropy']:.3f} clip={metrics['clip_frac']:.3f} "
                f"sps={sps:.0f}"
            )

            if cumulative_timesteps - last_ckpt >= args.checkpoint_every:
                checkpoint.save(ckpt_dir, model, learner.optimizer, make_meta())
                last_ckpt = cumulative_timesteps
                print(f"[train] checkpoint @ {cumulative_timesteps}")
    except KeyboardInterrupt:
        print("[train] interrupted; saving final checkpoint")
    finally:
        checkpoint.save(ckpt_dir, model, learner.optimizer, make_meta())
        env.close()
        total_elapsed = time.monotonic() - start
        print(f"[train] done: {cumulative_timesteps} timesteps in {total_elapsed:.1f}s")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
