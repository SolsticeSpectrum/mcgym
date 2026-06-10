"""Zero-gym-overhead web monitor: per-agent voxel 3D FOV + top-down minimap.

The gym already packs each agent's 17^3 voxel grid + pose into the observation and
the trainer already decodes it. This server hands that data (raw block ids + pose +
status) to the browser, throttled to the poll rate; the browser renders both views
on a 2D canvas (no GL). The 3D view is a per-pixel voxel raycaster (DDA) with
flat per-face cube shading — Minecraft-like, no textures. Colors are the real
Minecraft map colors (schema/block_colors.json). The gym does no extra work.
"""
from __future__ import annotations

import json
import pathlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .schema import spec
from .tasks.gather_wood import log_item_ids, wood_count

_COLORS_PATH = pathlib.Path(__file__).resolve().parents[2] / "schema" / "block_colors.json"


class TrainMonitor:
    def __init__(self, port: int, registry, n_agents: int, n_per: int = 0, poll_dt: float = 0.2):
        self.port = port
        self.n_agents = n_agents
        # Agents are laid out env-by-env: agent g belongs to env g // n_per. Lets the client
        # group the flat batch into envs without per-agent env tags.
        self._n_per = n_per if n_per > 0 else n_agents
        self._poll_dt = poll_dt
        self._log_arr = np.array(sorted(log_item_ids(registry)), dtype=np.int64)
        raw = json.loads(_COLORS_PATH.read_text())
        palette = {}
        for k, (r, g, b) in raw.items():
            palette[int(k)] = None if (r == 0 and g == 0 and b == 0) else f"#{r:02x}{g:02x}{b:02x}"
        self._palette_json = json.dumps({str(k): (v or "") for k, v in palette.items()})
        self._edge = spec.VOXEL_EDGE
        # Lightweight columnar snapshot for /state (vectors only — no voxels).
        self._snapshot = {"step": 0, "n_per": self._n_per, "edge": self._edge, "x": []}
        self._latest_obs = None  # ref to the most recent obs batch, for on-demand voxels
        self._last_snap = 0.0
        self._server = None

    def start(self) -> str:
        # Bind all interfaces so the monitor is reachable from outside the box (e.g. the
        # GPU box's public IP). It is an unauthenticated read-only view; expose only on a
        # trusted/reserved port.
        self._server = ThreadingHTTPServer(("0.0.0.0", self.port), self._make_handler())
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return f"http://0.0.0.0:{self.port}/"

    def update(self, obs_struct: np.ndarray, action_idx: np.ndarray, reward: np.ndarray, step: int) -> None:
        # Always keep the latest obs ref (cheap) so /agent can serve any agent's voxel on demand.
        self._latest_obs = obs_struct
        now = time.monotonic()
        if now - self._last_snap < self._poll_dt:
            return
        self._last_snap = now
        # Columnar vectors for every agent — tiny (~10 KB for 1152), no voxels, built vectorised.
        pos = obs_struct["pos"]
        wood = (np.isin(obs_struct["inv_item_id"], self._log_arr) * obs_struct["inv_count"]).sum(axis=1)
        self._snapshot = {
            "step": int(step),
            "n_per": self._n_per,
            "edge": self._edge,
            "x": np.round(pos[:, 0], 1).tolist(),
            "y": np.round(pos[:, 1], 1).tolist(),
            "z": np.round(pos[:, 2], 1).tolist(),
            "yaw": np.round(obs_struct["yaw"], 1).tolist(),
            "wood": wood.astype(int).tolist(),
            "look": obs_struct["target_in_range"].astype(int).tolist(),
        }

    def _agent_voxel(self, env: int, i: int) -> dict:
        """Sparse non-air cells of one agent's near voxel grid, on demand (client renders them)."""
        obs = self._latest_obs
        gi = env * self._n_per + i
        if obs is None or gi < 0 or gi >= len(obs):
            return {"env": env, "i": i, "cells": []}
        o = obs[gi]
        edge = self._edge
        r = (edge - 1) // 2
        vox = o["voxel_blocks"]
        nz = np.nonzero(vox)[0]
        ids = vox[nz]
        dy = nz // (edge * edge) - r
        rem = nz % (edge * edge)
        dz = rem // edge - r
        dx = rem % edge - r
        cells = np.stack([dx, dy, dz, ids], axis=1).astype(int).tolist()
        return {
            "env": env, "i": i, "edge": edge,
            "yaw": round(float(o["yaw"]), 1), "pitch": round(float(o["pitch"]), 1),
            "wood": int((np.isin(o["inv_item_id"], self._log_arr) * o["inv_count"]).sum()),
            "look": int(o["target_in_range"]), "target": int(o["target_block"]),
            "cells": cells,
        }

    def _make_handler(self):
        monitor = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, body: bytes, ctype: str):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")  # for the Vite dev server
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                from urllib.parse import parse_qs, urlparse
                parsed = urlparse(self.path)
                if parsed.path == "/state":
                    self._send(json.dumps(monitor._snapshot).encode(), "application/json")
                elif parsed.path == "/agent":
                    q = parse_qs(parsed.query)
                    env = int(q.get("env", ["0"])[0])
                    i = int(q.get("i", ["0"])[0])
                    self._send(json.dumps(monitor._agent_voxel(env, i)).encode(), "application/json")
                elif parsed.path == "/palette":
                    self._send(monitor._palette_json.encode(), "application/json")
                else:
                    self._send(_LANDING.encode(), "text/html")

        return Handler


_LANDING = """<!doctype html><meta charset=utf-8><title>MCAI monitor</title>
<body style="font:14px system-ui;background:#0b0e14;color:#cdd6f4;padding:2rem">
<h2>MCAI training monitor — JSON API</h2>
<p>Vector data for the React/Vite/three.js frontend (see <code>frontend/</code>):</p>
<ul>
<li><code>GET /state</code> — columnar vectors for every agent (no voxels): step, n_per, x/y/z/yaw/wood/look arrays. env of agent g = g // n_per.</li>
<li><code>GET /agent?env=E&i=I</code> — one agent's near voxel as sparse non-air cells [[dx,dy,dz,blockId],…] + pose/wood/target.</li>
<li><code>GET /palette</code> — block id → color.</li>
</ul>
<p>Run the UI: <code>cd frontend && npm i && npm run dev</code> (set VITE_MONITOR to this origin).</p>
</body>"""
