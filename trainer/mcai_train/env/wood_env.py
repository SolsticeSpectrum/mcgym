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
    WoodShapedReward,
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

        self._rewards = [
            WoodShapedReward(self._log_ids, self._log_block_ids)
            for _ in range(n_agents)
        ]
        self._step_counter = 0
        # Per-agent: True on the step right after a death frame, so the next step
        # re-inits that agent's reward tracker against its fresh respawn obs.
        self._just_died = np.zeros(n_agents, dtype=bool)

    def _reset_reward_state(self, obs_struct: np.ndarray) -> None:
        for i in range(self.n_agents):
            self._rewards[i].reset(obs_struct[i])
        self._step_counter = 0
        self._just_died[:] = False

    def reset(self) -> np.ndarray:
        obs_struct = self.transport.reset()
        self._reset_reward_state(obs_struct)
        return obs_struct

    def step(self, action_idx: np.ndarray):
        action_idx = np.asarray(action_idx)
        records = actions_to_records(action_idx)
        obs_struct = self.transport.step(records)

        # Attack is head index 6 (BINS order: forward,strafe,jump,sprint,yaw,pitch,attack);
        # value 1 = attack pressed this tick.
        attacked = action_idx[:, 6] == 1
        health = obs_struct["health"]

        reward = np.zeros(self.n_agents, dtype=np.float32)
        died = np.zeros(self.n_agents, dtype=bool)
        for i in range(self.n_agents):
            if self._just_died[i]:
                # First frame of the fresh episode after the gym auto-revived it:
                # re-init the tracker against the respawn obs, no reward this step.
                self._rewards[i].reset(obs_struct[i])
                self._just_died[i] = False
            elif health[i] <= 0.0:
                # Death frame: big penalty, terminal. The gym revives this agent
                # before the next step (per-agent auto-reset).
                reward[i] = -W_DEATH
                self._just_died[i] = True
                died[i] = True
            else:
                reward[i] = self._rewards[i].compute(obs_struct[i], bool(attacked[i]))

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
