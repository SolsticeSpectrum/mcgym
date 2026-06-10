import React, { useEffect, useMemo, useRef, useState } from 'react'
import EnvDetail from './EnvDetail.jsx'
import AgentView from './AgentView.jsx'

const POLL_MS = 600

// Top-down scatter of an env's agents. Fixed world->pixel scale around the env centroid so dots
// don't jitter as agents move. Yellow = aimed at a log, green = scoring, grey = neither.
function EnvMap({ agents, size = 150 }) {
  const ref = useRef(null)
  useEffect(() => {
    const ctx = ref.current.getContext('2d')
    ctx.fillStyle = '#0d1018'; ctx.fillRect(0, 0, size, size)
    if (!agents.length) return
    const cx = agents.reduce((s, a) => s + a.x, 0) / agents.length
    const cz = agents.reduce((s, a) => s + a.z, 0) / agents.length
    let span = 8
    for (const a of agents) span = Math.max(span, Math.abs(a.x - cx) * 2, Math.abs(a.z - cz) * 2)
    span *= 1.1
    for (const a of agents) {
      const px = size / 2 + ((a.x - cx) / span) * size
      const pz = size / 2 + ((a.z - cz) / span) * size
      ctx.fillStyle = a.score > 0 ? '#a6e3a1' : a.look ? '#f9e2af' : '#5b6378'
      ctx.beginPath(); ctx.arc(px, pz, 2.5, 0, 7); ctx.fill()
    }
  }, [agents, size])
  return <canvas ref={ref} width={size} height={size} style={{ display: 'block', borderRadius: 6 }} />
}

export default function App() {
  const [state, setState] = useState(null)
  const [palette, setPalette] = useState({})
  const [selEnv, setSelEnv] = useState(null)
  const [selAgent, setSelAgent] = useState(null)

  useEffect(() => { fetch('/palette').then((r) => r.json()).then(setPalette).catch(() => {}) }, [])
  useEffect(() => {
    let alive = true
    const tick = () => fetch('/state').then((r) => r.json()).then((s) => alive && setState(s)).catch(() => {})
    tick()
    const id = setInterval(tick, POLL_MS)
    return () => { alive = false; clearInterval(id) }
  }, [])

  const envs = useMemo(() => {
    if (!state || !state.x) return []
    const total = state.x.length
    const nPer = state.per || total
    const num = Math.max(1, Math.ceil(total / nPer))
    return Array.from({ length: num }, (_, e) => {
      const agents = []
      for (let i = 0; i < nPer; i++) {
        const g = e * nPer + i
        if (g >= total) break
        agents.push({ i, x: state.x[g], z: state.z[g], score: state.score[g], look: state.look[g] })
      }
      const meanScore = agents.length ? agents.reduce((s, a) => s + a.score, 0) / agents.length : 0
      return { id: e, agents, meanScore }
    })
  }, [state])

  if (!state || !state.x) {
    return <div style={{ padding: 24 }}>Connecting to the monitor…</div>
  }
  const nPer = state.per || state.x.length

  const Header = (
    <div style={{ height: 44, padding: '0 16px', borderBottom: '1px solid #1e2230', display: 'flex', gap: 18, alignItems: 'center' }}>
      <b>MCGym</b>
      <span style={{ color: '#9aa4bf' }}>step {state.step.toLocaleString()}</span>
      <span style={{ color: '#9aa4bf' }}>{envs.length} envs · {state.x.length} agents</span>
      {(selEnv != null || selAgent) && (
        <a href="#" onClick={(e) => { e.preventDefault(); selAgent ? setSelAgent(null) : setSelEnv(null) }}
          style={{ color: '#89b4fa' }}>← back</a>
      )}
    </div>
  )

  if (selAgent != null) return <div>{Header}<AgentView env={selEnv} i={selAgent} palette={palette} /></div>
  if (selEnv != null) return <div>{Header}<EnvDetail env={selEnv} nPer={nPer} palette={palette} onPick={setSelAgent} /></div>

  return (
    <div>{Header}
      <div style={{ padding: 16, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 12 }}>
        {envs.map((env) => (
          <div key={env.id} onClick={() => setSelEnv(env.id)}
            style={{ background: '#11151f', border: '1px solid #1e2230', borderRadius: 8, padding: 8, cursor: 'pointer' }}>
            <EnvMap agents={env.agents} />
            <div style={{ marginTop: 6, display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
              <span>env {env.id}</span>
              <span style={{ color: '#9aa4bf' }}>score {env.meanScore.toFixed(1)}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
