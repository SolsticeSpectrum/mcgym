"""Web monitor, vectors for every agent plus on demand voxel views, zero gym overhead"""
from __future__ import annotations

import json
import pathlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .schema import spec

COLORS = pathlib.Path(__file__).resolve().parents[2] / "schema" / "block_colors.json"
DIST   = pathlib.Path(__file__).resolve().parents[2] / "frontend" / "dist"


class Monitor:
    def __init__(self, port: int, agents: int, per: int, metric, poll: float = 0.2):
        self.port    = port
        self.agents  = agents
        self.per     = per  # agents per env, agent g belongs to env g // per
        self._metric = metric
        self._poll   = poll

        raw     = json.loads(COLORS.read_text())
        palette = {}
        for k, (r, g, b) in raw.items():
            palette[int(k)] = "" if (r == 0 and g == 0 and b == 0) else f"#{r:02x}{g:02x}{b:02x}"

        self._palette = json.dumps({str(k): v for k, v in palette.items()})

        self._edge   = spec.VOXEL_EDGE
        self._snap   = {"step": 0, "per": per, "edge": self._edge, "x": []}
        self._obs    = None
        self._act    = None
        self._last   = 0.0
        self._server = None

    def start(self) -> str:
        # bound on all interfaces, read only view, expose on a trusted port
        self._server = ThreadingHTTPServer(("0.0.0.0", self.port), self._handler())
        self.port    = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

        return f"http://0.0.0.0:{self.port}/"

    def update(self, obs: np.ndarray, act: np.ndarray, reward: np.ndarray, step: int) -> None:
        # keep refs so /agent can serve any agent on demand, snapshot is throttled
        self._obs = obs
        self._act = act

        now = time.monotonic()
        if now - self._last < self._poll:
            return

        self._last = now
        pos        = obs["pos"]
        self._snap = {
            "step":  int(step),
            "per":   self.per,
            "edge":  self._edge,
            "x":     np.round(pos[:, 0], 1).tolist(),
            "y":     np.round(pos[:, 1], 1).tolist(),
            "z":     np.round(pos[:, 2], 1).tolist(),
            "yaw":   np.round(obs["yaw"], 1).tolist(),
            "score": np.round(self._metric(obs).astype(float), 2).tolist(),
            "look":  obs["target_in_range"].astype(int).tolist(),
        }

    def _agent(self, env: int, i: int) -> dict:
        # sparse non air cells of one agents voxel grid, client renders them
        obs = self._obs
        g   = env * self.per + i
        if obs is None or g < 0 or g >= len(obs):
            return {"env": env, "i": i, "cells": []}

        o     = obs[g]
        edge  = self._edge
        r     = (edge - 1) // 2
        vox   = o["voxel_blocks"]
        nz    = np.nonzero(vox)[0]
        dy    = nz // (edge * edge) - r
        rem   = nz % (edge * edge)
        dz    = rem // edge - r
        dx    = rem % edge - r
        cells = np.stack([dx, dy, dz, vox[nz]], axis=1).astype(int).tolist()

        # target block pos from the look ray and hit distance, eye is 1.62 up
        yaw    = np.deg2rad(float(o["yaw"]))
        pitch  = np.deg2rad(float(o["pitch"]))
        cp     = np.cos(pitch)
        look   = np.array([-np.sin(yaw) * cp, -np.sin(pitch), np.cos(yaw) * cp])
        hit    = np.array([0.0, 1.62, 0.0]) + look * (float(o["target_distance"]) + 0.5)
        aimed  = int(o["target_in_range"])
        target = [int(np.floor(c)) for c in hit] if aimed else None

        attacking = 0
        if self._act is not None and g < len(self._act):
            attacking = int(self._act[g][6] != 0)

        return {
            "env": env, "i": i, "edge": edge,
            "yaw": round(float(o["yaw"]), 1), "pitch": round(float(o["pitch"]), 1),
            "score": float(self._metric(obs[g:g + 1])[0]),
            "look": aimed, "target": int(o["target_block"]), "targetPos": target,
            "attacking": attacking,
            "head": int(vox[((1 + r) * edge + r) * edge + r]),  # block at eye for the medium tint
            "cells": cells,
        }

    def _handler(self):
        mon = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, body: bytes, ctype: str):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                from urllib.parse import parse_qs, urlparse
                url = urlparse(self.path)
                if url.path == "/state":
                    self._send(json.dumps(mon._snap).encode(), "application/json")
                elif url.path == "/agent":
                    q = parse_qs(url.query)
                    self._send(json.dumps(mon._agent(int(q["env"][0]), int(q["i"][0]))).encode(),
                               "application/json")
                elif url.path == "/palette":
                    self._send(mon._palette.encode(), "application/json")
                else:
                    self._static(url.path)

            def _static(self, path: str):
                # serve the built frontend, unknown paths get index.html
                rel = path.lstrip("/") or "index.html"
                f   = (DIST / rel).resolve()
                if not (str(f).startswith(str(DIST)) and f.is_file()):
                    f = DIST / "index.html"
                ctype = {
                    ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
                    ".svg": "image/svg+xml", ".json": "application/json",
                }.get(f.suffix, "application/octet-stream")
                self._send(f.read_bytes(), ctype)

        return Handler
