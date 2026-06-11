"""generic env over the gym, task plugs in via mcgym.tasks.Task"""
from __future__ import annotations

import pathlib
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from mcgym.models.actions import actions_to_records
from mcgym.tasks.task import Task

from .launch import launch
from .transport import Transport


class Env:
    """one gym process, N agents in one world stepped in lockstep"""

    def __init__(self, agents: int, seed: int, task: Task, timeout: float = 180.0) -> None:
        self.agents = agents
        self.task   = task

        self._tmp  = tempfile.mkdtemp(prefix="mcgym_env_")
        self._shm  = f"/dev/shm/mcgym_shm_{uuid.uuid4().hex}.bin"
        self._sock = str(pathlib.Path(self._tmp) / "gym.sock")

        self._proc     = launch(agents, seed, self._shm, self._sock, timeout=timeout)
        self.transport = Transport(self._shm, self._sock, agents)

        self._step = 0
        # true on the step right after a death frame, gym revived the agent
        self._respawned = np.zeros(agents, dtype=bool)

    def reset(self) -> np.ndarray:
        obs = self.transport.reset()

        self._step         = 0
        self._respawned[:] = False
        self.task.reset(obs)

        return obs

    def step(self, act: np.ndarray):
        self.send(act)
        return self.recv()

    def send(self, act: np.ndarray) -> None:
        # fire without waiting so the vec env can tick all gyms in parallel
        self._pending = np.asarray(act)
        self.transport.send(actions_to_records(self._pending))

    def recv(self):
        obs = self.transport.recv()

        died   = obs["health"] <= 0.0
        reward = self.task.reward(obs, self._pending, died, self._respawned)
        self._respawned = died.copy()

        self._step += 1
        timeout = self._step >= self.task.eplen
        done    = self.task.done(died) | timeout

        if timeout:
            obs = self.transport.reset()
            self._step         = 0
            self._respawned[:] = False
            self.task.reset(obs)

        return obs, reward, done

    def metric(self, obs: np.ndarray) -> np.ndarray:
        return self.task.metric(obs)

    def close(self) -> None:
        try:
            self.transport.close()
        finally:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except Exception:
                self._proc.kill()

            pathlib.Path(self._shm).unlink(missing_ok=True)
            pathlib.Path(self._sock).unlink(missing_ok=True)


class VecEnv:
    """m gyms stepped as one M*per agent batch, gyms tick in parallel across cores"""

    def __init__(self, num_envs, agents, seed, task, timeout=180.0):
        self.num_envs = num_envs
        self.per      = agents
        self.agents   = num_envs * agents

        # pool reused for boot and recv, both release the gil so gyms overlap
        self._pool = ThreadPoolExecutor(max_workers=num_envs)
        self.envs = list(self._pool.map(
            lambda e: Env(agents, seed + e, task(agents), timeout=timeout),
            range(num_envs)))

    def reset(self) -> np.ndarray:
        return np.concatenate([e.reset() for e in self.envs])

    def send(self, act: np.ndarray) -> None:
        chunks = np.split(np.asarray(act), self.num_envs)
        for e, a in zip(self.envs, chunks):
            e.send(a)

    def recv(self):
        res = list(self._pool.map(lambda e: e.recv(), self.envs))
        return (
            np.concatenate([r[0] for r in res]),
            np.concatenate([r[1] for r in res]),
            np.concatenate([r[2] for r in res]),
        )

    def step(self, act: np.ndarray):
        self.send(act)
        return self.recv()

    def metric(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([
            self.envs[e].metric(obs[e * self.per:(e + 1) * self.per])
            for e in range(self.num_envs)
        ])

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        for e in self.envs:
            e.close()
