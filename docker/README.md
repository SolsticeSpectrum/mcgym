# docker

One compose file, fresh gpu host to running training. Copy this folder to the
host, fill `.env` from `.env.example`, `docker compose up -d`.

What happens on every up

1. `mcai-init` (busybox, exits right away) writes the ssh authorized_keys and
   two supervisord program configs into the data dir
2. `xgl` (the selkies desktop image) starts, supervisord picks up the programs,
   both backed by the single `bootstrap.sh`
   - ssh, dropbear on 2222, key only, host key persists so clients never see
     a host key warning
   - train, installs rustup and a python venv with cuda torch into
     `/drive2/tools` (first boot only), clones the repo, builds the gym and
     starts training in an auto resume loop

First boot takes 10 to 20 min for toolchains and the gym build, later boots go
straight to training and resume the latest checkpoint.

| what | where |
|---|---|
| web desktop | `http://host:8080`, user ubuntu, password from `.env` |
| training monitor | `http://host:9080` |
| ssh | `ssh -p 2222 ubuntu@host` |
| training log | `/drive2/train.log` |

## persistent layout

The container is disposable, only `/drive2` (the `DATA_DIR` mount) survives

```
tools/      rustup + cargo + venv + pip cache
mcai/       repo clone, re cloned on boot, keep no state here
runs/       checkpoints per task, the valuable part
xgl-ssh/    dropbear host key + cached debs
mcai-init/  files written by the init service
train.log   training output
```

## quirks

- private repo, embed a github pat in `REPO_URL`, see `.env.example`
- host networking, the compose ports section is decorative, port 22 on the
  host ip is the hosts own sshd not the container
- the image has no real root, sudo is fakeroot, the bootstrap uses
  `fakeroot apt-get`, real sudo is `sudo-root` with the container password,
  openssh sshd cannot run there which is why dropbear
- resources buy throughput directly, the rollout buffers need ~21 GB at 2048
  agents and the 32 gyms want a core each, raise `MEM_LIMIT` and `CPUS` if the
  host has headroom
- knob changes, edit `.env`, `docker compose up -d`, then
  `docker exec xgl supervisorctl restart mcai-train`
