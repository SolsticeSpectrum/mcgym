import React, { useEffect, useMemo, useRef, useState } from 'react'
import VoxelView from './VoxelView.jsx'

const POLL_MS = 500

// One env's top-down minimap: agents as dots at (x,z), colored by whether they hold wood.
function EnvMinimap({ agents, size = 120 }) {
  const ref = useRef(null)
  useEffect(() => {
    const cv = ref.current
    if (!cv) return
    const ctx = cv.getContext('2d')
    ctx.clearRect(0, 0, size, size)
    ctx.fillStyle = '#11151f'
    ctx.fillRect(0, 0, size, size)
    if (!agents.length) return
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity
    for (const a of agents) {
      minX = Math.min(minX, a.x); maxX = Math.max(maxX, a.x)
      minZ = Math.min(minZ, a.z); maxZ = Math.max(maxZ, a.z)
    }
    const span = Math.max(maxX - minX, maxZ - minZ, 8)
    for (const a of agents) {
      const px = 6 + ((a.x - minX) / span) * (size - 12)
      const pz = 6 + ((a.z - minZ) / span) * (size - 12)
      ctx.fillStyle = a.wood > 0 ? '#a6e3a1' : a.look ? '#f9e2af' : '#6c7086'
      ctx.beginPath(); ctx.arc(px, pz, 2.5, 0, 7); ctx.fill()
    }
  }, [agents, size])
  return <canvas ref={ref} width={size} height={size} style={{ borderRadius: 6, display: 'block' }} />
}

export default function App() {
  const [state, setState] = useState(null)
  const [palette, setPalette] = useState({})
  const [selEnv, setSelEnv] = useState(null)
  const [selAgent, setSelAgent] = useState(null) // {env, i}

  useEffect(() => {
    fetch('/palette').then((r) => r.json()).then(setPalette).catch(() => {})
  }, [])
  useEffect(() => {
    let alive = true
    const tick = () =>
      fetch('/state').then((r) => r.json()).then((s) => alive && setState(s)).catch(() => {})
    tick()
    const id = setInterval(tick, POLL_MS)
    return () => { alive = false; clearInterval(id) }
  }, [])

  // Group the flat columnar batch into envs.
  const envs = useMemo(() => {
    if (!state || !state.x) return []
    const total = state.x.length
    const nPer = state.n_per || total
    const num = Math.max(1, Math.ceil(total / nPer))
    const out = []
    for (let e = 0; e < num; e++) {
      const agents = []
      for (let i = 0; i < nPer; i++) {
        const g = e * nPer + i
        if (g >= total) break
        agents.push({ i, x: state.x[g], y: state.y[g], z: state.z[g], yaw: state.yaw[g], wood: state.wood[g], look: state.look[g] })
      }
      const wood = agents.reduce((s, a) => s + a.wood, 0)
      out.push({ id: e, agents, meanWood: agents.length ? wood / agents.length : 0 })
    }
    return out
  }, [state])

  if (!state || !state.x) {
    return <div style={{ padding: 24 }}>Connecting to the monitor… (train with <code>--monitor-port 9080</code>, then <code>npm run dev</code>)</div>
  }

  const total = state.x.length
  const Header = (
    <div style={{ padding: '10px 16px', borderBottom: '1px solid #1e2230', display: 'flex', gap: 16, alignItems: 'baseline' }}>
      <b>MCAI</b>
      <span>step {state.step.toLocaleString()}</span>
      <span>{envs.length} envs · {total} agents</span>
      {selEnv != null && <a href="#" onClick={(e) => { e.preventDefault(); setSelEnv(null); setSelAgent(null) }}>← all envs</a>}
      {selAgent && <a href="#" onClick={(e) => { e.preventDefault(); setSelAgent(null) }}>← env {selAgent.env}</a>}
    </div>
  )

  if (selAgent) {
    return <div>{Header}<VoxelView env={selAgent.env} i={selAgent.i} palette={palette} edge={state.edge} /></div>
  }

  if (selEnv != null) {
    const env = envs[selEnv]
    return (
      <div>{Header}
        <div style={{ padding: 16, display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(150px,1fr))', gap: 10 }}>
          {env.agents.map((a) => (
            <button key={a.i} onClick={() => setSelAgent({ env: selEnv, i: a.i })}
              style={{ textAlign: 'left', background: '#11151f', border: '1px solid #1e2230', borderRadius: 8, color: '#cdd6f4', padding: 10, cursor: 'pointer' }}>
              <div style={{ fontWeight: 600 }}>agent {a.i}</div>
              <div>wood {a.wood} {a.look ? '· 🎯' : ''}</div>
              <div style={{ color: '#6c7086' }}>{a.x.toFixed(0)}, {a.y.toFixed(0)}, {a.z.toFixed(0)}</div>
            </button>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div>{Header}
      <div style={{ padding: 16, display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(150px,1fr))', gap: 12 }}>
        {envs.map((env) => (
          <div key={env.id} onClick={() => setSelEnv(env.id)}
            style={{ background: '#11151f', border: '1px solid #1e2230', borderRadius: 8, padding: 8, cursor: 'pointer' }}>
            <EnvMinimap agents={env.agents} />
            <div style={{ marginTop: 6, display: 'flex', justifyContent: 'space-between' }}>
              <span>env {env.id}</span>
              <span style={{ color: '#a6e3a1' }}>🪵 {env.meanWood.toFixed(1)}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
