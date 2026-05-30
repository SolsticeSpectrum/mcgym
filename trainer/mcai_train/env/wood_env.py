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
from mcai_train.tasks.gather_wood import GatherWoodReward, log_item_ids, wood_count

from .gym_process import launch_gym
from .shm_transport import ShmTransport

# Small shaping bonus per step for looking at an in-range log block, to densify
# the otherwise sparse "wood gained" signal. Kept tiny so it cannot dominate the
# real reward of actually collecting wood (delta wood per log is >= 1.0).
TARGET_SHAPING = 0.01


class WoodEnv:
    def __init__(
        self,
        n_agents: int,
        seed: int,
        registry: Registry,
        episode_len: int = 500,
        gym_timeout_s: float = 180.0,
    ) -> None:
        self.n_agents = n_agents
        self.seed = seed
        self.registry = registry
        self.episode_len = episode_len
        self._log_ids = log_item_ids(registry)
        self._log_id_arr = np.array(sorted(self._log_ids), dtype=np.int64)

        self._tmpdir = tempfile.mkdtemp(prefix="mcai_woodenv_")
        self._shm_path = f"/dev/shm/mcai_shm_{uuid.uuid4().hex}.bin"
        self._sock_path = str(pathlib.Path(self._tmpdir) / "gym.sock")

        self._proc = launch_gym(
            n_agents, seed, self._shm_path, self._sock_path, timeout_s=gym_timeout_s
        )
        self.transport = ShmTransport(self._shm_path, self._sock_path, n_agents)

        self._rewards = [GatherWoodReward(self._log_ids) for _ in range(n_agents)]
        self._step_counter = 0

    def _reset_reward_state(self, obs_struct: np.ndarray) -> None:
        for i in range(self.n_agents):
            self._rewards[i].reset(obs_struct[i])
        self._step_counter = 0

    def reset(self) -> np.ndarray:
        obs_struct = self.transport.reset()
        self._reset_reward_state(obs_struct)
        return obs_struct

    def _target_is_log(self, obs_record) -> bool:
        return int(obs_record["target_block"]) in self._log_ids

    def step(self, action_idx: np.ndarray):
        records = actions_to_records(np.asarray(action_idx))
        obs_struct = self.transport.step(records)

        reward = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            delta = self._rewards[i].compute(obs_struct[i])
            shaping = 0.0
            if obs_struct[i]["target_in_range"] and self._target_is_log(obs_struct[i]):
                shaping = TARGET_SHAPING
            reward[i] = delta + shaping

        self._step_counter += 1
        done_flag = self._step_counter >= self.episode_len
        done = np.full(self.n_agents, done_flag, dtype=bool)

        if done_flag:
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
