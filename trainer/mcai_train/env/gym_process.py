"""Launch the Java Minecraft gym in external-step transport mode."""
from __future__ import annotations

import os
import pathlib
import subprocess
import threading
import time

# Repo-relative (gym_process.py -> env -> mcai_train -> trainer -> repo root -> minecraft-decomp).
GYM_REPO = pathlib.Path(__file__).resolve().parents[3] / "minecraft-decomp"
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
    arena: str = "",
) -> subprocess.Popen:
    """Start the gym via gradle runGymTransport and wait for readiness.

    Blocks until the gym prints MCAI_TRANSPORT_READY (shm mapped, socket bound)
    or the timeout elapses. World generation makes first boot slow, hence the
    generous default timeout. Returns the running process; the caller owns it.
    """
    # Rust gym (mcgym binary) when MCGYM points at it; else the Java gradle gym.
    rust_bin = os.environ.get("MCGYM")
    if rust_bin:
        cmd = [
            rust_bin,
            "--shm", shm_path,
            "--sock", sock_path,
            "--agents", str(n_agents),
            "--seed", str(seed),
            "--spacing", os.environ.get("MCAI_GYM_SPACING", "64"),
        ]
        if arena:
            cmd += ["--arena", arena]
        if curriculum:
            cmd += ["--curriculum", curriculum]
        cwd = None
    else:
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
            f"-Parena={arena}",
            "--offline",
            "--console=plain",
        ]
        cwd = str(GYM_REPO)

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
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
