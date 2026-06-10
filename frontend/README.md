# MCAI monitor frontend

React + Vite + three.js viewer for training. Renders entirely client-side from the trainer's
vector API (the Python monitor does **no** image assembly):

- `/state` — columnar vectors for every agent (no voxels): `step, n_per, edge, x/y/z/yaw/wood/look`.
  Env of agent `g` is `g // n_per`.
- `/agent?env=E&i=I` — one agent's near voxel as sparse non-air cells `[[dx,dy,dz,blockId],…]`.
- `/palette` — block id → color.

## Run

Start training with the monitor (e.g. `--monitor-port 9080`), then:

```bash
cd frontend
npm install
MCAI_MONITOR=http://localhost:9080 npm run dev   # or the box's IP:9080
```

Vite proxies `/state`, `/agent`, `/palette` to `MCAI_MONITOR`. Open the printed URL.

## Views

- **Env grid** — one card per env (24), top-down minimap of its agents (green = holding wood),
  mean wood. Click to open.
- **Agent list** — the env's agents with live pose/wood. Click to open.
- **Voxel view** — three.js instanced cubes of that agent's near grid (on-demand, live, real block
  colors), agent at centre.
