import { useEffect, useState } from 'react'
import { Views } from '../components/views.tsx'
import { smooth } from '../render/sticky.ts'
import type { Agent, Palette } from '../types.ts'

const POVW = 150
const H    = 100
const POLL = 250

// one env, every agent as a pov + top down card, click opens the agent
export function Env({ env, per, palette, onpick }: {
    env:     number
    per:     number
    palette: Palette
    onpick:  (i: number) => void
}) {
    const [agents, setAgents] = useState<(Agent | null)[]>([])

    useEffect(() => {
        let alive = true
        const loop = async () => {
            while (alive) {
                const batch = await Promise.all(
                    Array.from({ length: per }, (_, i) =>
                        fetch(`/agent?env=${env}&i=${i}`).then((r) => r.json() as Promise<Agent>).catch(() => null)))
                if (!alive) return

                setAgents(batch.map((d, i) => d && smooth(`${env}:${i}`, d)))
                await new Promise((r) => setTimeout(r, POLL))
            }
        }
        loop()
        return () => { alive = false }
    }, [env, per])

    return (
        <div className="agents">
            {Array.from({ length: per }, (_, i) => {
                const d = agents[i] ?? null
                return (
                    <div key={i} className="card" onClick={() => onpick(i)}>
                        <Views data={d} palette={palette} povw={POVW} h={H} />
                        <div className="meta">
                            <span>#{i}</span>
                            <span style={{ color: d?.attacking ? '#f38ba8' : d?.look ? '#f9e2af' : '#9aa4bf' }}>
                                {d?.attacking ? 'breaking' : d?.look ? 'aimed' : `score ${d?.score ?? 0}`}
                            </span>
                        </div>
                    </div>
                )
            })}
        </div>
    )
}
