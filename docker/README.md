# docker

fresh gpu host to running training. copy this folder, fill `.env` from
`.env.example`, `docker compose up -d`.

every up writes the ssh keys and two supervisord programs into the data dir,
both run `bootstrap.sh`. ssh is dropbear on 2222 with a persisted host key.
train installs rust and a cuda torch venv into `/drive2/tools` (first boot
only), clones the repo, builds the gym and the frontend and trains in an auto
resume loop. first boot takes 10 to 20 min, later boots resume right away.

- web desktop http://host:8080, user ubuntu, password from `.env`
- monitor http://host:9080
- `ssh -p 2222 ubuntu@host`
- log `/drive2/train.log`

## layout

only `/drive2` (the DATA_DIR mount) survives a recreate

```
tools/       rustup + cargo + venv + caches
mcgym/       repo clone, re cloned on boot, keep no state here
runs/        checkpoints per task
xgl-ssh/     dropbear host key + cached debs
mcgym-init/  files written by the init service
train.log    training output
```
