import { Map, type Dot } from '../components/map.tsx'
import type { State } from '../types.ts'

interface Env {
    id:    number
    dots:  Dot[]
    score: number
}

function split(state: State): Env[] {
    const total = state.x.length
    const per = state.per || total
    const num = Math.max(1, Math.ceil(total / per))

    return Array.from({ length: num }, (_, e) => {
        const dots: Dot[] = []
        for (let i = 0; i < per; i++) {
            const g = e * per + i
            if (g >= total) break
            dots.push({ x: state.x[g], z: state.z[g], score: state.score[g], look: state.look[g] })
        }
        const score = dots.length ? dots.reduce((s, d) => s + d.score, 0) / dots.length : 0
        return { id: e, dots, score }
    })
}

// home page, one minimap card per env, click opens the env
export function Envs({ state, onpick }: { state: State; onpick: (env: number) => void }) {
    return (
        <div className="grid">
            {split(state).map((env) => (
                <div key={env.id} className="card" onClick={() => onpick(env.id)}>
                    <Map dots={env.dots} />
                    <div className="meta">
                        <span>env {env.id}</span>
                        <span className="dim">score {env.score.toFixed(1)}</span>
                    </div>
                </div>
            ))}
        </div>
    )
}
