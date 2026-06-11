#!/usr/bin/env python3
"""raw gym throughput, drive RESET/STEP over real shm+uds with noop actions, report ticks/sec"""
import mmap
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time

MAGIC = 0x4D434759
CMD_RESET, CMD_STEP, CMD_CLOSE, REPLY_OK = b"\x01", b"\x02", b"\x03", 1


def main() -> None:
    # usage: bench_throughput.py <mcgym binary> [agents] [spacing] [steps]
    binary  = sys.argv[1] if len(sys.argv) > 1 else "target/release/mcgym"
    n       = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    spacing = sys.argv[3] if len(sys.argv) > 3 else "32"
    steps   = int(sys.argv[4]) if len(sys.argv) > 4 else 2000

    d = tempfile.mkdtemp()
    shm, sock = os.path.join(d, "shm.bin"), os.path.join(d, "gym.sock")
    proc = subprocess.Popen(
        [binary, "--shm", shm, "--sock", sock, "--agents", str(n), "--seed", "0", "--spacing", spacing],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    boot = time.monotonic()
    for line in proc.stdout:
        if "MCGYM_TRANSPORT_READY" in line:
            break
    else:
        raise SystemExit("gym never became ready")

    boot_dt = time.monotonic() - boot

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock)

    mm = mmap.mmap(os.open(shm, os.O_RDWR), 0)
    magic, ver, na = struct.unpack_from("<iii", mm, 0)
    assert magic == MAGIC and ver == 1 and na == n, (magic, ver, na)

    s.sendall(CMD_RESET)
    assert s.recv(1)[0] == REPLY_OK

    t0 = time.monotonic()
    for _ in range(steps):
        s.sendall(CMD_STEP)
        if s.recv(1)[0] != REPLY_OK:
            raise SystemExit("bad reply")

    dt = time.monotonic() - t0
    s.sendall(CMD_CLOSE)

    print(
        f"agents={n} boot={boot_dt:.1f}s steps={steps} dt={dt:.3f}s  "
        f"TPS/world={steps / dt:.0f}  agent-steps/s={n * steps / dt:.0f}"
    )

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


if __name__ == "__main__":
    main()
