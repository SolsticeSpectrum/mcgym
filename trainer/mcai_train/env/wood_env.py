"""Vectorised gather-wood environment over the real Minecraft gym.

The gym exposes N agents in one world; we treat them as N synchronised parallel
envs. Episodes truncate together at ``episode_len`` steps, at which point the
next reset issues a transport CMD_RESET (which respawns all agents) and clears
per-agent reward state.
"""
from __future__ import annotations

import pathlib
import tempfile
import uuid

import numpy as np

from mcai_train.models.action_space import actions_to_records
from mcai_train.schema.registry import Registry
from mcai_train.tasks.gather_wood import (
    W_DEATH,
    BatchWoodReward,
    log_block_ids,
    log_item_ids,
    wood_count,
)

from .gym_process import launch_gym
from .shm_transport import ShmTransport


class WoodEnv:
    def __init__(
        self,
        n_agents: int,
        seed: int,
        registry: Registry,
        episode_len: int = 500,
        gym_timeout_s: float = 180.0,
        curriculum: str = "",
        arena: str = "",
    ) -> None:
        self.n_agents = n_agents
        self.seed = seed
        self.registry = registry
        self.episode_len = episode_len
        self._log_ids = log_item_ids(registry)
        self._log_block_ids = log_block_ids(registry)
        self._log_id_arr = np.array(sorted(self._log_ids), dtype=np.int64)

        self._tmpdir = tempfile.mkdtemp(prefix="mcai_woodenv_")
        self._shm_path = f"/dev/shm/mcai_shm_{uuid.uuid4().hex}.bin"
        self._sock_path = str(pathlib.Path(self._tmpdir) / "gym.sock")

        self._proc = launch_gym(
            n_agents, seed, self._shm_path, self._sock_path,
            timeout_s=gym_timeout_s, curriculum=curriculum, arena=arena,
        )
        self.transport = ShmTransport(self._shm_path, self._sock_path, n_agents)

        self._reward = BatchWoodReward(n_agents, self._log_ids, self._log_block_ids)
        self._step_counter = 0
        # Per-agent: True on the step right after a death frame, so the next step
        # re-inits that agent's reward tracker against its fresh respawn obs.
        self._just_died = np.zeros(n_agents, dtype=bool)

    def _reset_reward_state(self, obs_struct: np.ndarray) -> None:
        self._reward.reset(obs_struct)
        self._step_counter = 0
        self._just_died[:] = False

    def reset(self) -> np.ndarray:
        obs_struct = self.transport.reset()
        self._reset_reward_state(obs_struct)
        return obs_struct

    def step(self, action_idx: np.ndarray):
        self.step_send(action_idx)
        return self.step_recv()

    def step_send(self, action_idx: np.ndarray) -> None:
        """Fire this env's gym step without waiting (for the parallel vec-env)."""
        self._pending_action = np.asarray(action_idx)
        self.transport.step_send(actions_to_records(self._pending_action))

    def step_recv(self):
        obs_struct = self.transport.step_recv()
        action_idx = self._pending_action

        # Attack is head index 6 (BINS order: forward,strafe,jump,sprint,yaw,pitch,attack);
        # value 1 = attack pressed this tick.
        attacked = action_idx[:, 6] == 1
        health = obs_struct["health"]

        # Vectorised reward over all agents (updates the batch tracker for all).
        reward = self._reward.compute(obs_struct, attacked)
        # Agents revived this step (the frame after a death): their cross-episode
        # delta is spurious, so zero it; the tracker is now re-based on the respawn.
        reward[self._just_died] = 0.0
        # Death frame: override with the penalty; the gym revives these next step.
        died = health <= 0.0
        reward[died] = -W_DEATH
        self._just_died = died.copy()

        self._step_counter += 1
        timeout = self._step_counter >= self.episode_len
        done = died | timeout  # per-agent terminal on death; synchronized on timeout

        if timeout:
            obs_struct = self.transport.reset()
            self._reset_reward_state(obs_struct)

        return obs_struct, reward, done

    def wood_held(self, obs_struct: np.ndarray) -> np.ndarray:
        """Per-agent total wood currently held (for logging)."""
        return np.array(
            [wood_count(obs_struct[i], self._log_ids) for i in range(self.n_agents)],
            dtype=np.int64,
        )

    def close(self) -> None:
        try:
            self.transport.close()
        finally:
            if self._proc is not None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=30)
                except Exception:
                    self._proc.kill()
            pathlib.Path(self._shm_path).unlink(missing_ok=True)
            pathlib.Path(self._sock_path).unlink(missing_ok=True)


class ParallelVecEnv:
    """M WoodEnv gyms stepped concurrently as one (M*N_per)-agent vec-env.

    step() fires every gym's tick first (step_send) then collects them (step_recv),
    so the M Java gym processes tick in parallel across CPU cores while Python
    waits once. Obs/reward/done are concatenated into one (M*N_per,) batch for a
    single GPU forward — the rlgym-ppo multi-process pattern, here over our shm
    transport. The single-process WoodEnv path is unchanged.

    Correct, but NO throughput win on this hardware. Profiled (py-spy): the logic is
    fine and it does NOT deadlock — 2 gyms x 8 = ~35 ms/step, 2 gyms x 96 = ~280
    ms/step (192 agents -> ~685 sps, WORSE per-agent than single-process's ~1083 sps
    at 100 agents). The gym tick for ~96 real-physics ServerPlayers is CPU-heavy, and
    a 6-core box can't run enough gyms concurrently to amortize it (the step profile
    is split across torch forward + socket recv waiting on the gyms, no Python
    hotspot). Net: this box is CPU-bound on per-agent vanilla physics; multiprocess
    helps only with more cores/machines or cheaper per-agent physics. Use --num-envs 1
    (default, supported). ParallelVecEnv is kept for multi-machine / future use.
    """

    def __init__(self, num_envs, n_agents, seed, registry, episode_len=256,
                 curriculum="", arena="", gym_timeout_s=180.0):
        self.num_envs = num_envs
        self.n_per = n_agents
        self.n_agents = num_envs * n_agents  # total, for the buffer/model
        self.envs = [
            WoodEnv(n_agents, seed + e, registry, episode_len=episode_len,
                    curriculum=curriculum, arena=arena, gym_timeout_s=gym_timeout_s)
            for e in range(num_envs)
        ]

    def reset(self) -> np.ndarray:
        return np.concatenate([e.reset() for e in self.envs])

    def step(self, action_idx: np.ndarray):
        action_idx = np.asarray(action_idx)
        chunks = np.split(action_idx, self.num_envs)  # each (n_per, 7)
        for e, a in zip(self.envs, chunks):
            e.step_send(a)
        obs, rew, done = [], [], []
        for e in self.envs:
            o, r, d = e.step_recv()
            obs.append(o); rew.append(r); done.append(d)
        return np.concatenate(obs), np.concatenate(rew), np.concatenate(done)

    def wood_held(self, obs_struct: np.ndarray) -> np.ndarray:
        return np.concatenate([
            self.envs[e].wood_held(obs_struct[e * self.n_per:(e + 1) * self.n_per])
            for e in range(self.num_envs)
        ])

    def close(self) -> None:
        for e in self.envs:
            e.close()
