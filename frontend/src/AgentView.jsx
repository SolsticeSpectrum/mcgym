import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { createVoxelScene, WATER, LAVA } from './voxelScene.js'
import { drawAgentMini } from './miniRender.js'
import { smoothAgent } from './sticky.js'

const mediumTint = (head) =>
  head === WATER ? 'rgba(40,90,210,0.42)' : head === LAVA ? 'rgba(235,110,20,0.5)' : null

const POVW = 168
const VH = 110

// Full agent view: fullscreen mouse-orbit 3D (own scene, Steve visible, optional transparency)
// plus a small minimap card (POV + top-down) in the top-right, rendered by the shared
// mini-renderer (always opaque, Steve hidden in POV, arrow in top-down).
export default function AgentView({ env, i, palette }) {
  const wrap = useRef(null)
  const povRef = useRef(null)
  const topRef = useRef(null)
  const api = useRef(null)
  const [info, setInfo] = useState(null)
  const [transparent, setTransparent] = useState(false)
  // Draggable + resizable minimap card. w = total canvas width, h = height; top-down is h x h
  // (square), POV is (w - h) x h (wider).
  const [pos, setPos] = useState(() => ({ x: Math.max(10, window.innerWidth - 352), y: 54 }))
  const [size, setSize] = useState({ w: 340, h: 120 })
  const drag = useRef(null)
  useEffect(() => {
    const move = (e) => {
      const d = drag.current
      if (!d) return
      if (d.type === 'move') setPos({ x: d.px + (e.clientX - d.sx), y: d.py + (e.clientY - d.sy) })
      else setSize({ w: Math.max(220, d.pw + (e.clientX - d.sx)), h: Math.max(90, d.ph + (e.clientY - d.sy)) })
    }
    const up = () => { drag.current = null }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    return () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up) }
  }, [])
  const startDrag = (e) => { drag.current = { type: 'move', sx: e.clientX, sy: e.clientY, px: pos.x, py: pos.y }; e.preventDefault() }
  const startResize = (e) => { drag.current = { type: 'resize', sx: e.clientX, sy: e.clientY, pw: size.w, ph: size.h }; e.preventDefault(); e.stopPropagation() }
  const povW = Math.max(90, size.w - size.h)

  useEffect(() => {
    const mount = wrap.current
    const { scene, update, setTransparent: setT } = createVoxelScene()
    const cam = new THREE.PerspectiveCamera(50, 2, 0.1, 500)
    cam.position.set(22, 17, 22)
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(window.devicePixelRatio)
    mount.appendChild(renderer.domElement)
    const controls = new OrbitControls(cam, renderer.domElement)
    controls.target.set(0, 1, 0)
    controls.enableDamping = true
    api.current = { update, setT }

    const resize = () => {
      const w = mount.clientWidth, h = mount.clientHeight
      renderer.setSize(w, h)
      cam.aspect = w / h
      cam.updateProjectionMatrix()
    }
    resize()
    window.addEventListener('resize', resize)
    let raf
    const render = () => { controls.update(); renderer.render(scene, cam); raf = requestAnimationFrame(render) }
    render()
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
      controls.dispose()
      renderer.dispose()
      if (renderer.domElement.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement)
      api.current = null
    }
  }, [])

  useEffect(() => { if (api.current) api.current.setT(transparent) }, [transparent])

  useEffect(() => {
    let alive = true
    const tick = () =>
      fetch(`/agent?env=${env}&i=${i}`)
        .then((r) => r.json())
        .then((raw) => {
          if (!alive) return
          const d = smoothAgent(`${env}:${i}`, raw)
          setInfo(d)
          if (api.current) api.current.update(d, palette)
          drawAgentMini(d, palette, povRef.current, topRef.current)
        })
        .catch(() => {})
    tick()
    const id = setInterval(tick, 350)
    return () => { alive = false; clearInterval(id) }
  }, [env, i, palette])

  return (
    <div style={{ position: 'relative', height: 'calc(100vh - 44px)' }}>
      <div ref={wrap} style={{ position: 'absolute', inset: 0 }} />
      <div style={{ position: 'absolute', top: 10, left: 10, padding: '6px 10px', background: '#0d1018cc', borderRadius: 6 }}>
        env {env} · agent {i}
        {info && <span style={{ color: '#9aa4bf' }}> — wood {info.wood} · yaw {info.yaw} · pitch {info.pitch}{info.attacking ? ' · breaking' : info.look ? ' · aimed' : ''}</span>}
        <label style={{ marginLeft: 14, color: '#9aa4bf' }}>
          <input type="checkbox" checked={transparent} onChange={(e) => setTransparent(e.target.checked)} /> transparent
        </label>
      </div>
      <div style={{ position: 'absolute', left: pos.x, top: pos.y, background: '#0d1018ee', border: '1px solid #2a3040', borderRadius: 8, overflow: 'hidden', userSelect: 'none', boxShadow: '0 4px 16px #0008' }}>
        <div onPointerDown={startDrag} style={{ cursor: 'move', padding: '3px 8px', fontSize: 11, color: '#9aa4bf', background: '#161b27', display: 'flex', justifyContent: 'space-between' }}>
          <span>POV</span><span style={{ color: '#5b6378' }}>drag · resize ↘</span><span>top-down</span>
        </div>
        <div style={{ display: 'flex' }}>
          <div style={{ position: 'relative' }}>
            <canvas ref={povRef} width={povW} height={size.h} style={{ display: 'block', borderRight: '1px solid #1e2230' }} />
            {info && mediumTint(info.head) && (
              <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', background: mediumTint(info.head) }} />
            )}
          </div>
          <canvas ref={topRef} width={size.h} height={size.h} style={{ display: 'block' }} />
        </div>
        <div onPointerDown={startResize} title="resize"
          style={{ position: 'absolute', right: 0, bottom: 0, width: 16, height: 16, cursor: 'nwse-resize', background: 'linear-gradient(135deg, transparent 45%, #5b6378 45%, #5b6378 70%, transparent 70%)' }} />
      </div>
      <div style={{ position: 'absolute', bottom: 8, left: 10, color: '#6b7390', fontSize: 12 }}>drag to orbit · scroll to zoom</div>
    </div>
  )
}
