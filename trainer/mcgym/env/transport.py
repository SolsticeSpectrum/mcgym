"""shm + unix socket client for the gym, one byte command one byte reply"""
from __future__ import annotations

import socket

import numpy as np

from mcgym.schema import codec, spec

HEADER = 64
MAGIC  = 0x4D434149

RESET = 1
STEP  = 2
CLOSE = 3
OK    = 1


class Transport:
    def __init__(self, shm: str, sock: str, agents: int) -> None:
        self.agents = agents

        self._act_off = HEADER
        self._obs_off = HEADER + agents * spec.ACTION_NBYTES
        total         = self._obs_off + agents * spec.OBS_NBYTES

        self._mm = np.memmap(shm, dtype="u1", mode="r+", shape=(total,))
        magic, version, mapped = np.frombuffer(self._mm[:12].tobytes(), dtype="<i4")
        if magic != MAGIC:
            raise ValueError(f"shm magic mismatch, got {magic:#x} want {MAGIC:#x}")
        if version != spec.SCHEMA_VERSION:
            raise ValueError(f"shm schema {version} != local {spec.SCHEMA_VERSION}")
        if mapped != agents:
            raise ValueError(f"shm agents {mapped} != requested {agents}")

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(sock)

    def _acts(self) -> np.ndarray:
        end = self._act_off + self.agents * spec.ACTION_NBYTES
        return self._mm[self._act_off:end].view(spec.ACTION_DTYPE)

    def _obs(self) -> np.ndarray:
        end = self._obs_off + self.agents * spec.OBS_NBYTES
        return self._mm[self._obs_off:end].view(spec.OBS_DTYPE)

    def _command(self, cmd: int) -> None:
        self._sock.sendall(bytes([cmd]))
        reply = self._recv_exact(1)
        if reply[0] != OK:
            raise RuntimeError(f"gym replied {reply[0]} expected {OK}")

    def _recv_exact(self, n: int) -> bytes:
        chunks = []
        left   = n
        while left:
            chunk = self._sock.recv(left)
            if not chunk:
                raise ConnectionError("gym closed the socket mid reply")
            chunks.append(chunk)
            left -= len(chunk)

        return b"".join(chunks)

    def reset(self) -> np.ndarray:
        self._command(RESET)
        return self._obs().copy()

    def step(self, acts) -> np.ndarray:
        self.send(acts)
        return self.recv()

    def send(self, acts) -> None:
        # fire the step without waiting so a vec env can tick all gyms in parallel
        view = self._acts()
        if isinstance(acts, np.ndarray) and acts.dtype == spec.ACTION_DTYPE:
            view[:] = acts
        else:
            view[:] = np.frombuffer(codec.encode_action_batch(list(acts)), dtype=spec.ACTION_DTYPE)
        self._sock.sendall(bytes([STEP]))

    def recv(self) -> np.ndarray:
        reply = self._recv_exact(1)
        if reply[0] != OK:
            raise RuntimeError(f"gym replied {reply[0]} expected {OK}")
        return self._obs().copy()

    def close(self) -> None:
        try:
            self._command(CLOSE)
        except (OSError, RuntimeError, ConnectionError):
            pass
        finally:
            self._sock.close()
            self._mm = None  # drop the mmap so the file can be unlinked
