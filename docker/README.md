# MCAI GPU box — one-command training deployment

Everything needed to turn a fresh GPU host into a running MCAI training box:
a Selkies web desktop, key-only SSH, and the wood task training automatically
with the live monitor — from three files and one `docker compose up -d`.

## Quickstart (host admin)

```bash
# on the host, in any folder:
#   docker-compose.yml  bootstrap.sh  .env   (copy .env.example -> .env, fill it in)
docker compose up -d
```

That's it. On every `up`:

1. **mcai-init** (busybox, exits immediately) writes the SSH `authorized_keys`
   (from `.env`) and two supervisord program configs into `${DATA_DIR}/mcai-init`.
2. **xgl** (the Selkies desktop image) starts; its supervisord picks up the two
   programs, both backed by the single `bootstrap.sh`:
   - **ssh** — dropbear on port `2222` (key-only; the host key persists across
     recreates so clients never see a host-key warning)
   - **train** — installs rustup + a Python venv with CUDA torch into
     `/drive2/tools` (first boot only; cached afterwards), `git clone`s the repo
     (`REPO_URL`), `cargo build`s the Rust gym, and launches the wood task in an
     auto-resume/auto-restart loop

After the first boot (toolchain download + gym build, ~10–20 min) the box is
live. Later boots skip straight to training in under a minute, resuming from
the latest checkpoint.

| What | Where |
|---|---|
| Web desktop | `http://<host>:8080` (user `ubuntu`, password from `.env`) |
| Training monitor | `http://<host>:9080` |
| SSH | `ssh -p 2222 ubuntu@<host>` (keys from `.env`) |
| Training log | `/drive2/train.log` (also `docker exec xgl tail -f /drive2/train.log`) |
| SSH program log | `/tmp/mcai-ssh.log` inside the container |

## Persistent layout (`${DATA_DIR}` = `/drive2` in-container)

The container is disposable; only `/drive2` survives recreates:

```
/drive2/
  tools/      rustup + cargo + python venv (torch) + pip cache
  mcai/       git clone of the repo (re-cloned/reset on boot — keep no state here)
  runs/       checkpoints per run name (this is the valuable part)
  xgl-ssh/    dropbear host key + cached debs
  mcai-init/  files written by the init service on each `up`
  train.log   training output
```

## Notes & quirks

- **Private repo**: embed a GitHub fine-grained PAT (read-only Contents) in
  `REPO_URL` — see `.env.example`.
- **`network_mode: host`**: the compose `ports:` section is decorative; selkies
  (8080), the monitor (9080) and dropbear (2222) bind directly on the host.
  Port 22 on the host IP is the host's own sshd, not the container.
- **The image has no real root**: `/usr/bin/sudo` is a fakeroot symlink. The
  bootstrap uses `fakeroot apt-get` for packages; the real setuid sudo is
  `sudo-root` (container `PASSWD`). OpenSSH sshd cannot run here (privsep
  chroot gets EPERM) — that's why dropbear.
- **Resources directly buy throughput**: at the default 2048 agents the
  trainer's two rollout buffers need ~21 GB and the 32 gym processes want a
  core each. If the host has headroom, raise `MEM_LIMIT`/`CPUS` in `.env` (and
  `NUM_ENVS` to match the cores). The tuned defaults hit ~14.7k steps/s on an
  RTX 6000 with `CPUS=12`/`MEM_LIMIT=64g`.
- **Changing training knobs**: edit `.env`, then `docker compose up -d`
  (recreates the init files) and restart training:
  `docker exec xgl supervisorctl restart mcai-train`.
