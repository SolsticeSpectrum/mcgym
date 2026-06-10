import { useEffect, useState } from 'react'
import { Envs } from './pages/envs.tsx'
import { Env } from './pages/env.tsx'
import { Agent } from './pages/agent.tsx'
import type { Palette, State } from './types.ts'

const POLL = 600

export function App() {
    const [state, setState]     = useState<State | null>(null)
    const [palette, setPalette] = useState<Palette>({})
    const [env, setEnv]         = useState<number | null>(null)
    const [agent, setAgent]     = useState<number | null>(null)

    useEffect(() => {
        fetch('/palette').then((r) => r.json()).then(setPalette).catch(() => {})
    }, [])

    useEffect(() => {
        let alive = true
        const tick = () => fetch('/state').then((r) => r.json()).then((s) => alive && setState(s)).catch(() => {})
        tick()
        const id = setInterval(tick, POLL)
        return () => { alive = false; clearInterval(id) }
    }, [])

    if (!state || !state.x.length) return <div style={{ padding: 24 }}>connecting to the monitor</div>

    const per = state.per || state.x.length
    const back = () => (agent != null ? setAgent(null) : setEnv(null))

    return (
        <div>
            <div className="head">
                <b>MCGym</b>
                <span className="dim">step {state.step.toLocaleString()}</span>
                <span className="dim">{Math.ceil(state.x.length / per)} envs · {state.x.length} agents</span>
                {env != null && <a href="#" onClick={(e) => { e.preventDefault(); back() }}>back</a>}
            </div>

            {env == null ? <Envs state={state} onpick={setEnv} />
                : agent == null ? <Env env={env} per={per} palette={palette} onpick={setAgent} />
                : <Agent env={env} i={agent} palette={palette} />}
        </div>
    )
}
