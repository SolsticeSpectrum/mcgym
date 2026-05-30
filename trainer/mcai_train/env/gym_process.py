"""Launch the Java Minecraft gym in external-step transport mode."""
from __future__ import annotations

import pathlib
import subprocess
import threading
import time

GYM_REPO = pathlib.Path("/home/user/github/mcai/minecraft-decomp")
READY_PREFIX = "MCAI_TRANSPORT_READY"


def _drain(stream) -> None:
    """Keep consuming the gym's stdout so its pipe never blocks the server."""
    for line in iter(stream.readline, ""):
        print(line, end="")


def launch_gym(
    n_agents: int,
    seed: int,
    shm_path: str,
    sock_path: str,
    timeout_s: float = 180.0,
    curriculum: str = "",
) -> subprocess.Popen:
    """Start the gym via gradle runGymTransport and wait for readiness.

    Blocks until the gym prints MCAI_TRANSPORT_READY (shm mapped, socket bound)
    or the timeout elapses. World generation makes first boot slow, hence the
    generous default timeout. Returns the running process; the caller owns it.
    """
    gradlew = GYM_REPO / "gradlew"
    if not gradlew.exists():
        raise FileNotFoundError(f"gradlew not found at {gradlew}")

    cmd = [
        str(gradlew),
        "runGymTransport",
        f"-PshmPath={shm_path}",
        f"-PsockPath={sock_path}",
        f"-Pagents={n_agents}",
        f"-Pseed={seed}",
        f"-Pcurriculum={curriculum}",
        "--offline",
        "--console=plain",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(GYM_REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if line == "" and proc.poll() is not None:
            raise RuntimeError(
                f"gym exited (code {proc.returncode}) before becoming ready"
            )
        if line:
            print(line, end="")
            if READY_PREFIX in line:
                drainer = threading.Thread(
                    target=_drain, args=(proc.stdout,), daemon=True
                )
                drainer.start()
                return proc

    proc.kill()
    raise TimeoutError(f"gym did not become ready within {timeout_s}s")
