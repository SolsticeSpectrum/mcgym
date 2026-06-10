import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { createVoxelScene } from './voxelScene.js'
import { drawAgentMini } from './miniRender.js'

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
        .then((d) => {
          if (!alive) return
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
      <div style={{ position: 'absolute', top: 10, right: 10, background: '#0d1018cc', border: '1px solid #1e2230', borderRadius: 8, overflow: 'hidden' }}>
        <div style={{ display: 'flex' }}>
          <canvas ref={povRef} width={POVW} height={VH} style={{ display: 'block', borderRight: '1px solid #1e2230' }} />
          <canvas ref={topRef} width={VH} height={VH} style={{ display: 'block' }} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 8px', fontSize: 11, color: '#6b7390' }}>
          <span>POV</span><span>top-down</span>
        </div>
      </div>
      <div style={{ position: 'absolute', bottom: 8, left: 10, color: '#6b7390', fontSize: 12 }}>drag to orbit · scroll to zoom</div>
    </div>
  )
}
