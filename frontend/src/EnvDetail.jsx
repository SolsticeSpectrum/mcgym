import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { createVoxelScene, lookDir } from './voxelScene.js'

const EYE = 1.62
const CW = 150
const CH = 96

// One env's agents, each shown as a POV + top-down mini-view. A single shared WebGL renderer
// renders one reused scene per agent and copies the pixels into each card's 2D canvases — so we
// get 2*N live views without N WebGL contexts. Click a card to open the full 3D view.
export default function EnvDetail({ env, nPer, palette, onPick }) {
  const cards = useRef({}) // i -> { pov, top, setMeta }
  const [meta, setMeta] = useState({}) // i -> {wood, look, attacking}

  useEffect(() => {
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(CW, CH)
    const { scene, update } = createVoxelScene()
    const povCam = new THREE.PerspectiveCamera(75, CW / CH, 0.05, 500)
    const topCam = new THREE.OrthographicCamera(-9, 9, 9, -9, 0.1, 500)
    topCam.position.set(0, 30, 0)
    topCam.up.set(0, 0, -1)
    topCam.lookAt(0, 0, 0)

    let alive = true
    const refresh = async () => {
      while (alive) {
        const datas = await Promise.all(
          Array.from({ length: nPer }, (_, i) =>
            fetch(`/agent?env=${env}&i=${i}`).then((r) => r.json()).catch(() => null)),
        )
        if (!alive) return
        const nextMeta = {}
        for (let i = 0; i < nPer; i++) {
          const d = datas[i]
          const c = cards.current[i]
          if (!d) continue
          nextMeta[i] = { wood: d.wood, look: d.look, attacking: d.attacking }
          if (!c || !c.pov) continue
          update(d, palette)
          const dir = lookDir(d.yaw || 0, d.pitch || 0)
          povCam.position.set(0, EYE, 0)
          povCam.lookAt(dir.x, EYE + dir.y, dir.z)
          renderer.render(scene, povCam)
          c.pov.getContext('2d').drawImage(renderer.domElement, 0, 0, CW, CH)
          renderer.render(scene, topCam)
          c.top.getContext('2d').drawImage(renderer.domElement, 0, 0, CW, CH)
        }
        setMeta(nextMeta)
        await new Promise((res) => setTimeout(res, 250))
      }
    }
    refresh()
    return () => { alive = false; renderer.dispose() }
  }, [env, nPer, palette])

  return (
    <div style={{ padding: 14, display: 'grid', gridTemplateColumns: `repeat(auto-fill, ${CW}px)`, gap: 12 }}>
      {Array.from({ length: nPer }, (_, i) => {
        const m = meta[i] || {}
        return (
          <div key={i} onClick={() => onPick(i)} title={`agent ${i}`}
            style={{ background: '#11151f', border: '1px solid #1e2230', borderRadius: 8, cursor: 'pointer', overflow: 'hidden' }}>
            <canvas ref={(el) => { cards.current[i] = { ...cards.current[i], pov: el } }} width={CW} height={CH} style={{ display: 'block', borderBottom: '1px solid #1e2230' }} />
            <canvas ref={(el) => { cards.current[i] = { ...cards.current[i], top: el } }} width={CW} height={CH} style={{ display: 'block' }} />
            <div style={{ padding: '4px 6px', display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
              <span>#{i}</span>
              <span style={{ color: m.attacking ? '#f38ba8' : m.look ? '#f9e2af' : '#9aa4bf' }}>
                {m.attacking ? 'breaking' : m.look ? 'aimed' : `wood ${m.wood ?? 0}`}
              </span>
            </div>
          </div>
        )
      })}
    </div>
  )
}
