import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'

// Renders one agent's near voxel grid (sparse non-air cells fetched on demand) as instanced
// cubes, colored by the block palette. All rendering is client-side; the server only sends
// vectors. Re-fetches while open so the view is live.
export default function VoxelView({ env, i, palette, edge = 17 }) {
  const mountRef = useRef(null)
  const meshRef = useRef(null)
  const [info, setInfo] = useState(null)

  useEffect(() => {
    const mount = mountRef.current
    const w = mount.clientWidth || 600
    const h = mount.clientHeight || 400
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0b0e14)
    const cam = new THREE.PerspectiveCamera(55, w / h, 0.1, 1000)
    cam.position.set(24, 20, 24)
    cam.lookAt(0, 0, 0)
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setSize(w, h)
    mount.appendChild(renderer.domElement)
    scene.add(new THREE.AmbientLight(0xffffff, 0.65))
    const dir = new THREE.DirectionalLight(0xffffff, 0.8)
    dir.position.set(1, 2, 1)
    scene.add(dir)
    const group = new THREE.Group()
    scene.add(group)
    const mesh = new THREE.InstancedMesh(
      new THREE.BoxGeometry(0.96, 0.96, 0.96),
      new THREE.MeshLambertMaterial(),
      edge * edge * edge,
    )
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    mesh.count = 0
    group.add(mesh)
    meshRef.current = mesh
    let raf
    const animate = () => {
      group.rotation.y += 0.003
      renderer.render(scene, cam)
      raf = requestAnimationFrame(animate)
    }
    animate()
    const onResize = () => {
      const w2 = mount.clientWidth, h2 = mount.clientHeight
      cam.aspect = w2 / h2; cam.updateProjectionMatrix(); renderer.setSize(w2, h2)
    }
    window.addEventListener('resize', onResize)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
      renderer.dispose()
      if (renderer.domElement.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement)
    }
  }, [edge])

  useEffect(() => {
    let alive = true
    const dummy = new THREE.Object3D()
    const col = new THREE.Color()
    const colorFor = (id) => {
      const c = palette[String(id)]
      return c && c.length ? c : '#5a5a5a'
    }
    const tick = () =>
      fetch(`/agent?env=${env}&i=${i}`)
        .then((r) => r.json())
        .then((d) => {
          if (!alive) return
          setInfo(d)
          const mesh = meshRef.current
          if (!mesh) return
          const cells = d.cells || []
          mesh.count = cells.length
          for (let k = 0; k < cells.length; k++) {
            const [dx, dy, dz, id] = cells[k]
            dummy.position.set(dx, dy, dz)
            dummy.updateMatrix()
            mesh.setMatrixAt(k, dummy.matrix)
            col.set(colorFor(id))
            mesh.setColorAt(k, col)
          }
          mesh.instanceMatrix.needsUpdate = true
          if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
        })
        .catch(() => {})
    tick()
    const id = setInterval(tick, 700)
    return () => { alive = false; clearInterval(id) }
  }, [env, i, palette])

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 44px)' }}>
      <div ref={mountRef} style={{ flex: 1, minWidth: 0 }} />
      <div style={{ width: 220, padding: 14, borderLeft: '1px solid #1e2230' }}>
        <h3 style={{ marginTop: 0 }}>env {env} · agent {i}</h3>
        {info && (
          <div style={{ display: 'grid', gap: 6 }}>
            <div>🪵 wood: <b>{info.wood}</b></div>
            <div>yaw {info.yaw}° · pitch {info.pitch}°</div>
            <div>target block: {info.target}{info.look ? ' · in range 🎯' : ''}</div>
            <div style={{ color: '#6c7086' }}>{info.cells ? info.cells.length : 0} solid cells (17³ grid)</div>
          </div>
        )}
        <p style={{ color: '#6c7086', marginTop: 16 }}>Near voxel grid, agent at centre, auto-rotating. Real Minecraft block colors.</p>
      </div>
    </div>
  )
}
