import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { createVoxelScene, lookDir } from './voxelScene.js'

const EYE = 1.62

// Full agent view: a mouse-orbit 3D scene plus a POV camera (from the agent's eyes) and a
// top-down camera, all of one shared scene rendered into three viewports of a single canvas.
export default function AgentView({ env, i, palette }) {
  const wrap = useRef(null)
  const api = useRef(null)
  const [info, setInfo] = useState(null)
  const [transparent, setTransparent] = useState(false)

  useEffect(() => {
    const mount = wrap.current
    const W = () => mount.clientWidth
    const H = () => mount.clientHeight
    const { scene, update, setTransparent: setT } = createVoxelScene()

    const orbitCam = new THREE.PerspectiveCamera(50, 2, 0.1, 500)
    orbitCam.position.set(20, 16, 20)
    const povCam = new THREE.PerspectiveCamera(75, 1, 0.05, 500)
    const topCam = new THREE.OrthographicCamera(-10, 10, 10, -10, 0.1, 500)
    topCam.position.set(0, 30, 0)
    topCam.up.set(0, 0, -1)
    topCam.lookAt(0, 0, 0)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setScissorTest(true)
    mount.appendChild(renderer.domElement)
    const controls = new OrbitControls(orbitCam, renderer.domElement)
    controls.target.set(0, 1, 0)
    controls.enableDamping = true

    api.current = { update, setT, scene, povCam }

    const resize = () => {
      renderer.setSize(W(), H())
    }
    resize()
    window.addEventListener('resize', resize)

    let raf
    const render = () => {
      controls.update()
      const w = W(), h = H()
      const mw = Math.floor(w * 0.66)
      const sw = w - mw
      const sh = Math.floor(h / 2)
      // main orbit (left)
      orbitCam.aspect = mw / h; orbitCam.updateProjectionMatrix()
      renderer.setViewport(0, 0, mw, h); renderer.setScissor(0, 0, mw, h)
      renderer.render(scene, orbitCam)
      // POV (top-right)
      povCam.aspect = sw / sh; povCam.updateProjectionMatrix()
      renderer.setViewport(mw, sh, sw, sh); renderer.setScissor(mw, sh, sw, sh)
      renderer.render(scene, povCam)
      // top-down (bottom-right)
      renderer.setViewport(mw, 0, sw, sh); renderer.setScissor(mw, 0, sw, sh)
      renderer.render(scene, topCam)
      raf = requestAnimationFrame(render)
    }
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

  useEffect(() => {
    if (api.current) api.current.setT(transparent)
  }, [transparent])

  useEffect(() => {
    let alive = true
    const tick = () =>
      fetch(`/agent?env=${env}&i=${i}`)
        .then((r) => r.json())
        .then((d) => {
          if (!alive || !api.current) return
          setInfo(d)
          api.current.update(d, palette)
          // place the POV camera at the eyes, looking along the agent's gaze
          const dir = lookDir(d.yaw || 0, d.pitch || 0)
          const cam = api.current.povCam
          cam.position.set(0, EYE, 0)
          cam.lookAt(dir.x, EYE + dir.y, dir.z)
        })
        .catch(() => {})
    tick()
    const id = setInterval(tick, 400)
    return () => { alive = false; clearInterval(id) }
  }, [env, i, palette])

  return (
    <div style={{ position: 'relative', height: 'calc(100vh - 44px)' }}>
      <div ref={wrap} style={{ position: 'absolute', inset: 0 }} />
      <div style={{ position: 'absolute', top: 8, left: 8, padding: '6px 10px', background: '#0d1018cc', borderRadius: 6 }}>
        env {env} · agent {i}
        {info && <span style={{ color: '#9aa4bf' }}> — wood {info.wood} · yaw {info.yaw} · pitch {info.pitch}{info.attacking ? ' · breaking' : info.look ? ' · aimed at block' : ''}</span>}
        <label style={{ marginLeft: 14, color: '#9aa4bf' }}>
          <input type="checkbox" checked={transparent} onChange={(e) => setTransparent(e.target.checked)} /> transparent blocks
        </label>
      </div>
      <div style={{ position: 'absolute', top: 8, right: 8, color: '#9aa4bf', background: '#0d1018cc', padding: '2px 8px', borderRadius: 6 }}>POV</div>
      <div style={{ position: 'absolute', bottom: 8, right: 8, color: '#9aa4bf', background: '#0d1018cc', padding: '2px 8px', borderRadius: 6 }}>top-down</div>
      <div style={{ position: 'absolute', bottom: 8, left: 8, color: '#6b7390', fontSize: 12 }}>drag to orbit · scroll to zoom</div>
    </div>
  )
}
