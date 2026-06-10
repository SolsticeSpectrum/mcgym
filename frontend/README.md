# frontend

react + vite + three.js viewer for training, renders entirely client side from
the monitors vector api.

- `/state` columnar vectors for every agent, step, per, x/y/z/yaw/score/look,
  env of agent g is g // per
- `/agent?env=E&i=I` one agents near voxel as sparse non air cells plus pose
- `/palette` block id to color

## run

```bash
cd frontend
npm install
npm run dev    # vite proxies the api to the monitor
```

or `npm run build`, the trainer monitor serves `dist/` itself.

## views

- env grid, one minimap card per env, click to open
- agent cards, first person + top down per agent, click to open
- full 3d, orbitable voxel view with the player model and a draggable minimap
