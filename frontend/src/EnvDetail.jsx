import React, { useEffect, useRef, useState } from 'react'
import { drawAgentMini } from './miniRender.js'
import { WATER, LAVA } from './voxelScene.js'

const mediumTint = (head) =>
  head === WATER ? 'rgba(40,90,210,0.42)' : head === LAVA ? 'rgba(235,110,20,0.5)' : null

const POVW = 150
const VH = 100 // shared height; top-down is VH x VH (square), POV is POVW x VH (wider)

// One env's agents, each a POV (wide) + top-down (square) mini-view side by side, rendered by the
// shared mini-renderer. Click a card to open the full 3D view.
export default function EnvDetail({ env, nPer, palette, onPick }) {
  const cards = useRef({}) // i -> { pov, top }
  const [meta, setMeta] = useState({})

  useEffect(() => {
    let alive = true
    const loop = async () => {
      while (alive) {
        const datas = await Promise.all(
          Array.from({ length: nPer }, (_, i) =>
            fetch(`/agent?env=${env}&i=${i}`).then((r) => r.json()).catch(() => null)),
        )
        if (!alive) return
        const nextMeta = {}
        for (let i = 0; i < nPer; i++) {
          const d = datas[i]
          if (!d) continue
          nextMeta[i] = { wood: d.wood, look: d.look, attacking: d.attacking, head: d.head }
          const c = cards.current[i]
          if (c && c.pov) drawAgentMini(d, palette, c.pov, c.top)
        }
        setMeta(nextMeta)
        await new Promise((r) => setTimeout(r, 250))
      }
    }
    loop()
    return () => { alive = false }
  }, [env, nPer, palette])

  return (
    <div style={{ padding: 14, display: 'grid', gridTemplateColumns: `repeat(auto-fill, ${POVW + VH}px)`, gap: 12 }}>
      {Array.from({ length: nPer }, (_, i) => {
        const m = meta[i] || {}
        return (
          <div key={i} onClick={() => onPick(i)} title={`agent ${i}`}
            style={{ background: '#11151f', border: '1px solid #1e2230', borderRadius: 8, cursor: 'pointer', overflow: 'hidden' }}>
            <div style={{ display: 'flex' }}>
              <div style={{ position: 'relative' }}>
                <canvas ref={(el) => { cards.current[i] = { ...cards.current[i], pov: el } }} width={POVW} height={VH} style={{ display: 'block', borderRight: '1px solid #1e2230' }} />
                {mediumTint(m.head) && <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', background: mediumTint(m.head) }} />}
              </div>
              <canvas ref={(el) => { cards.current[i] = { ...cards.current[i], top: el } }} width={VH} height={VH} style={{ display: 'block' }} />
            </div>
            <div style={{ padding: '4px 8px', display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
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
