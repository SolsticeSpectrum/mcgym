"""launch the sim process and wait until its transport is live"""
from __future__ import annotations

import os
import subprocess
import threading
import time

READY = "SIM_TRANSPORT_READY"


def _drain(stream) -> None:
    # keep consuming stdout so the pipe never blocks the sim
    for line in iter(stream.readline, ""):
        print(line, end="")


def launch(agents: int, seed: int, shm: str, sock: str, timeout: float = 180.0) -> subprocess.Popen:
    """start the sim and block until it prints the ready line"""
    binary = os.environ.get("SIM")
    if not binary:
        raise RuntimeError("SIM not set, point it at the sim release binary")

    cmd = [
        binary,
        "--shm", shm,
        "--sock", sock,
        "--agents", str(agents),
        "--seed", str(seed),
        "--spacing", os.environ.get("SIM_SPACING", "128"),
    ]
    world = os.environ.get("SIM_WORLD")
    if world:
        spawn = os.environ.get("SIM_SPAWN")
        if not spawn:
            raise RuntimeError("SIM_WORLD needs SIM_SPAWN as x,y,z,yaw")
        cmd += ["--world", world, "--spawn", spawn,
                "--mining", os.environ.get("SIM_MINING", "1")]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if line == "" and proc.poll() is not None:
            raise RuntimeError(f"sim exited (code {proc.returncode}) before ready")
        if line:
            print(line, end="")
            if READY in line:
                threading.Thread(target=_drain, args=(proc.stdout,), daemon=True).start()
                return proc

    proc.kill()
    raise TimeoutError(f"sim not ready within {timeout}s")
