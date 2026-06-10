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
    def __init__(self, port: int, registry, n_agents: int, poll_dt: float = 0.2,
                 preview_cap: int = 24):
        self.port = port
        self.n_agents = n_agents
        # Each previewed agent ships a full 4913-int voxel grid; serialising all of them (e.g.
        # 1000+) every snapshot is huge and stalls the browser. Cap to a representative sample.
        self._preview_cap = preview_cap
        self._poll_dt = poll_dt
        self._log_ids = log_item_ids(registry)
        raw = json.loads(_COLORS_PATH.read_text())
        palette = {}
        for k, (r, g, b) in raw.items():
            palette[int(k)] = None if (r == 0 and g == 0 and b == 0) else f"#{r:02x}{g:02x}{b:02x}"
        self._palette_json = json.dumps({str(k): (v or "") for k, v in palette.items()})
        self._snapshot = {"step": 0, "edge": spec.VOXEL_EDGE, "agents": []}
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
        now = time.monotonic()
        if now - self._last_snap < self._poll_dt:
            return
        self._last_snap = now
        agents = []
        n_show = min(self.n_agents, len(obs_struct), self._preview_cap)
        for i in range(n_show):
            o = obs_struct[i]
            agents.append({
                "id": int(o["agent_id"]),
                "x": round(float(o["pos"][0]), 1),
                "y": round(float(o["pos"][1]), 1),
                "z": round(float(o["pos"][2]), 1),
                "yaw": round(float(o["yaw"]), 1),
                "pitch": round(float(o["pitch"]), 1),
                "voxel": o["voxel_blocks"].astype(int).tolist(),
                "wood": int(wood_count(o, self._log_ids)),
                "reward": round(float(reward[i]), 3),
                "act": self._act_str(action_idx[i]),
                "look_log": int(o["target_in_range"]) == 1,
            })
        self._snapshot = {
            "step": int(step),
            "edge": spec.VOXEL_EDGE,
            "agents": agents,
            "total": int(self.n_agents),
        }

    @staticmethod
    def _act_str(a: np.ndarray) -> str:
        fwd = {0: "BACK", 1: "·", 2: "FWD"}[int(a[0])]
        strafe = {0: "L", 1: "·", 2: "R"}[int(a[1])]
        parts = [fwd, strafe]
        if int(a[2]) == 1:
            parts.append("jump")
        if int(a[3]) == 1:
            parts.append("sprint")
        if int(a[6]) == 1:
            parts.append("ATTACK")
        return " ".join(parts)

    def _make_handler(self):
        monitor = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, body: bytes, ctype: str):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/state"):
                    self._send(json.dumps(monitor._snapshot).encode(), "application/json")
                elif self.path.startswith("/palette"):
                    self._send(monitor._palette_json.encode(), "application/json")
                else:
                    self._send(_HTML.encode(), "text/html")

        return Handler


_HTML = r"""<!doctype html><html><head><meta charset=utf-8><title>MCAI agents</title>
<style>
 html,body{background:#0b0e14;color:#cdd6e0;font:12px/1.3 monospace;margin:0}
 #hdr{padding:6px 8px;color:#9fd}
 #grid{display:flex;flex-wrap:wrap;gap:8px;padding:8px;align-content:flex-start}
 .tile{width:320px;height:184px;box-sizing:border-box;background:#121724;border:1px solid #243;
       border-radius:5px;padding:6px;overflow:hidden;flex:0 0 auto}
 .views{display:flex;gap:6px;height:120px}
 .views canvas{background:#0b0e14;display:block;image-rendering:pixelated}
 .status{height:44px;margin-top:5px;overflow:hidden;white-space:nowrap;color:#aeb6c2}
 .wood{color:#caa56a}.atk{color:#ff6b6b}.look{color:#7fd}
</style></head><body>
<div id=hdr>connecting…</div><div id=grid></div>
<script>
// FOV internal render res (cheap); displayed scaled-up, pixelated. Minimap 120px.
const FOVW=132,FOVH=90, FOVDW=176,FOVDH=120, MM=120, FOV=70*Math.PI/180;
let PAL={};
function col(id){const c=PAL[id];return (c===''||c===undefined)?null:c;}
function rgbOf(hex){return [parseInt(hex.slice(1,3),16),parseInt(hex.slice(3,5),16),parseInt(hex.slice(5,7),16)];}

// DDA voxel ray: returns {r,g,b,dist,shade} of the first solid block, or null (sky).
function cast(v,E,ox,oy,oz,dx,dy,dz){
 const R=(E-1)/2;
 let ix=Math.floor(ox),iy=Math.floor(oy),iz=Math.floor(oz);
 const sgx=dx>0?1:-1,sgy=dy>0?1:-1,sgz=dz>0?1:-1;
 const tDX=Math.abs(1/dx),tDY=Math.abs(1/dy),tDZ=Math.abs(1/dz);
 let tMX=(dx>0?(ix+1-ox):(ox-ix))*tDX;
 let tMY=(dy>0?(iy+1-oy):(oy-iy))*tDY;
 let tMZ=(dz>0?(iz+1-oz):(oz-iz))*tDZ;
 let t=0,shade=0.8;
 for(let s=0;s<48;s++){
  if(tMX<tMY&&tMX<tMZ){ix+=sgx;t=tMX;tMX+=tDX;shade=0.70;}          // X face
  else if(tMY<tMZ){iy+=sgy;t=tMY;tMY+=tDY;shade=(sgy<0?1.0:0.42);}  // down->top(bright), up->bottom(dark)
  else{iz+=sgz;t=tMZ;tMZ+=tDZ;shade=0.86;}                          // Z face
  if(ix<-R||ix>R||iy<-R||iy>R||iz<-R||iz>R)return null;
  const c=col(v[((iy+R)*E+(iz+R))*E+(ix+R)]);
  if(c){const rgb=rgbOf(c);return {r:rgb[0],g:rgb[1],b:rgb[2],dist:t,shade:shade};}
 }
 return null;
}
function drawFOV(ctx,v,E,yaw,pitch){
 const W=FOVW,H=FOVH, img=ctx.createImageData(W,H), d=img.data;
 const yr=yaw*Math.PI/180, pr=pitch*Math.PI/180;
 const fx=-Math.sin(yr)*Math.cos(pr), fy=-Math.sin(pr), fz=Math.cos(yr)*Math.cos(pr); // forward (MC look)
 const rx=Math.cos(yr), ry=0, rz=Math.sin(yr);                                        // right (horizontal)
 const ux=fy*rz-fz*ry, uy=fz*rx-fx*rz, uz=fx*ry-fy*rx;                                 // up = forward x right
 const tanf=Math.tan(FOV/2), asp=W/H, ox=0.5,oy=1.62,oz=0.5;
 for(let sy=0;sy<H;sy++){
  const vv=(1-2*sy/H)*tanf;
  for(let sx=0;sx<W;sx++){
   const u=(2*sx/W-1)*asp*tanf;
   let dx=fx+u*rx+vv*ux, dy=fy+u*ry+vv*uy, dz=fz+u*rz+vv*uz;
   const inv=1/Math.hypot(dx,dy,dz); dx*=inv;dy*=inv;dz*=inv;
   const hit=cast(v,E,ox,oy,oz,dx,dy,dz), o=(sy*W+sx)*4;
   if(hit){const f=hit.shade*Math.max(0.32,1-hit.dist/14);
     d[o]=hit.r*f;d[o+1]=hit.g*f;d[o+2]=hit.b*f;d[o+3]=255;}
   else{ // simple sky: lighter near horizon
     const k=sy/H; d[o]=18+30*k;d[o+1]=26+34*k;d[o+2]=44+40*k;d[o+3]=255;}
  }
 }
 ctx.putImageData(img,0,0);
}
function drawMini(ctx,v,E,yaw){
 const cell=MM/E;
 for(let z=0;z<E;z++)for(let x=0;x<E;x++){
  let c=null;
  for(let y=(E-1)/2;y>=-(E-1)/2;y--){const hc=col(v[((y+(E-1)/2)*E+z)*E+x]);if(hc){c=hc;break;}}
  ctx.fillStyle=c||'#0b0e14';ctx.fillRect(x*cell,z*cell,Math.ceil(cell),Math.ceil(cell));
 }
 const cx=MM/2,cy=MM/2,r=yaw*Math.PI/180,dx=-Math.sin(r),dz=Math.cos(r);
 ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+dx*14,cy+dz*14);ctx.stroke();
 ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(cx,cy,2.5,0,7);ctx.fill();
}
function tileFor(id){
 let t=document.getElementById('t'+id); if(t)return t;
 t=document.createElement('div');t.className='tile';t.id='t'+id;
 t.innerHTML='<div class=views><canvas width='+FOVW+' height='+FOVH+
   ' style="width:'+FOVDW+'px;height:'+FOVDH+'px"></canvas>'+
   '<canvas width='+MM+' height='+MM+'></canvas></div><div class=status id=s'+id+'></div>';
 document.getElementById('grid').appendChild(t);return t;
}
async function tick(){
 try{
  const s=await (await fetch('/state')).json();
  document.getElementById('hdr').textContent='MCAI training — step '+s.step+' — '+s.agents.length+' agents (left: voxel 3D · right: minimap · real map colors)';
  for(const a of s.agents){
   const t=tileFor(a.id), cv=t.querySelectorAll('canvas');
   drawFOV(cv[0].getContext('2d'),a.voxel,s.edge,a.yaw,a.pitch);
   drawMini(cv[1].getContext('2d'),a.voxel,s.edge,a.yaw);
   const atk=a.act.includes('ATTACK')?' <span class=atk>ATTACK</span>':'';
   document.getElementById('s'+a.id).innerHTML=
     'agent '+a.id+'  <span class=wood>wood '+a.wood+'</span>  r '+a.reward+(a.look_log?'  <span class=look>look-log</span>':'')+'<br>'+
     a.act.replace('ATTACK','')+atk+'<br>'+
     'xz '+a.x+','+a.z+'  y'+a.y+'  yaw'+a.yaw+' pit'+a.pitch;
  }
 }catch(e){document.getElementById('hdr').textContent='waiting for trainer… ('+e+')';}
 setTimeout(tick,400);
}
fetch('/palette').then(r=>r.json()).then(p=>{PAL=p;tick();});
</script></body></html>"""
