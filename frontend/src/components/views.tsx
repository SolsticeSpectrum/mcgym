import { useEffect, useRef } from 'react'
import { mini } from '../render/mini.ts'
import { tint } from '../render/scene.ts'
import type { Agent, Palette } from '../types.ts'

// pov + top down canvas pair drawn by the shared mini renderer, pov gets the
// fluid tint overlay when the agents eye is under water or lava
export function Views({ data, palette, povw, h }: { data: Agent | null; palette: Palette; povw: number; h: number }) {
    const pov = useRef<HTMLCanvasElement>(null)
    const top = useRef<HTMLCanvasElement>(null)

    useEffect(() => {
        if (data) mini(data, palette, pov.current, top.current)
    }, [data, palette, povw, h])

    const over = data ? tint(data.head) : null
    return (
        <div className="views">
            <div className="pov">
                <canvas ref={pov} width={povw} height={h} />
                {over && <div className="tint" style={{ background: over }} />}
            </div>
            <canvas ref={top} width={h} height={h} />
        </div>
    )
}
