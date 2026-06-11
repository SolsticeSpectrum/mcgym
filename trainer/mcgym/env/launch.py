"""launch the mcgym process and wait until its transport is live"""
from __future__ import annotations

import os
import subprocess
import threading
import time

READY = "MCGYM_TRANSPORT_READY"


def _drain(stream) -> None:
    # keep consuming stdout so the pipe never blocks the gym
    for line in iter(stream.readline, ""):
        print(line, end="")


def launch(agents: int, seed: int, shm: str, sock: str, timeout: float = 180.0) -> subprocess.Popen:
    """start mcgym and block until it prints the ready line"""
    gym = os.environ.get("MCGYM")
    if not gym:
        raise RuntimeError("MCGYM not set, point it at the mcgym release binary")

    cmd = [
        gym,
        "--shm", shm,
        "--sock", sock,
        "--agents", str(agents),
        "--seed", str(seed),
        "--spacing", os.environ.get("MCGYM_SPACING", "128"),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if line == "" and proc.poll() is not None:
            raise RuntimeError(f"gym exited (code {proc.returncode}) before ready")
        if line:
            print(line, end="")
            if READY in line:
                threading.Thread(target=_drain, args=(proc.stdout,), daemon=True).start()
                return proc

    proc.kill()
    raise TimeoutError(f"gym not ready within {timeout}s")
