"""Shared-memory + Unix-domain-socket client for the Minecraft gym.

Mirrors net.minecraft.mcai.McaiTransport on the Java side. The single mapped file
holds a 64-byte header, then the ACTION region (Python writes), then the OBS
region (Java writes). Stepping is paced by a one-byte request/one-byte reply
handshake over the socket: Python sends a command, Java advances the gym and
writes the OBS region, then replies OK.
"""
from __future__ import annotations

import socket

import numpy as np

from mcai_train.schema import codec, spec

HEADER_NBYTES = 64
MAGIC = 0x4D434149  # 'MCAI'

CMD_RESET = 1
CMD_STEP = 2
CMD_CLOSE = 3
REPLY_OK = 1


class ShmTransport:
    def __init__(self, shm_path: str, sock_path: str, n_agents: int) -> None:
        self.shm_path = shm_path
        self.sock_path = sock_path
        self.n_agents = n_agents

        self._action_off = HEADER_NBYTES
        self._obs_off = HEADER_NBYTES + n_agents * spec.ACTION_NBYTES
        total = self._obs_off + n_agents * spec.OBS_NBYTES

        self._mm = np.memmap(shm_path, dtype="u1", mode="r+", shape=(total,))
        magic, schema_version, mapped_agents = np.frombuffer(
            self._mm[:12].tobytes(), dtype="<i4"
        )
        if magic != MAGIC:
            raise ValueError(f"shm magic mismatch: got {magic:#x}, want {MAGIC:#x}")
        if schema_version != spec.SCHEMA_VERSION:
            raise ValueError(
                f"shm schema_version {schema_version} != local {spec.SCHEMA_VERSION}"
            )
        if mapped_agents != n_agents:
            raise ValueError(
                f"shm n_agents {mapped_agents} != requested {n_agents}"
            )

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(sock_path)

    def _action_view(self) -> np.ndarray:
        """Writable (n,) ACTION_DTYPE view onto the shm ACTION region."""
        end = self._action_off + self.n_agents * spec.ACTION_NBYTES
        return self._mm[self._action_off:end].view(spec.ACTION_DTYPE)

    def _obs_view(self) -> np.ndarray:
        """(n,) OBS_DTYPE view onto the shm OBS region."""
        end = self._obs_off + self.n_agents * spec.OBS_NBYTES
        return self._mm[self._obs_off:end].view(spec.OBS_DTYPE)

    def _command(self, cmd: int) -> None:
        self._sock.sendall(bytes([cmd]))
        reply = self._recv_exact(1)
        if reply[0] != REPLY_OK:
            raise RuntimeError(f"gym replied {reply[0]}, expected OK={REPLY_OK}")

    def _recv_exact(self, n: int) -> bytes:
        chunks = []
        remaining = n
        while remaining:
            chunk = self._sock.recv(remaining)
            if not chunk:
                raise ConnectionError("gym closed the socket mid-reply")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def reset(self) -> np.ndarray:
        self._command(CMD_RESET)
        return self._obs_view().copy()

    def step(self, actions) -> np.ndarray:
        self.step_send(actions)
        return self.step_recv()

    def step_send(self, actions) -> None:
        """Write actions to shm and fire CMD_STEP WITHOUT waiting for the reply.

        Split from the reply so a vec-env can fire all its gyms' steps first and
        only then collect replies — the gyms then tick concurrently across cores.
        """
        view = self._action_view()
        if isinstance(actions, np.ndarray) and actions.dtype == spec.ACTION_DTYPE:
            view[:] = actions
        else:
            encoded = codec.encode_action_batch(list(actions))
            view[:] = np.frombuffer(encoded, dtype=spec.ACTION_DTYPE)
        self._sock.sendall(bytes([CMD_STEP]))

    def step_recv(self) -> np.ndarray:
        reply = self._recv_exact(1)
        if reply[0] != REPLY_OK:
            raise RuntimeError(f"gym replied {reply[0]}, expected OK={REPLY_OK}")
        return self._obs_view().copy()

    def close(self) -> None:
        try:
            self._command(CMD_CLOSE)
        except (OSError, RuntimeError, ConnectionError):
            pass
        finally:
            self._sock.close()
            # Drop the memmap handle so the backing file can be unlinked.
            self._mm = None
