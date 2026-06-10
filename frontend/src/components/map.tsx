import { useEffect, useRef } from 'react'

export interface Dot {
    x:     number
    z:     number
    score: number
    look:  number
}

// top down scatter of one envs agents, fixed scale around the centroid so dots
// dont jitter, yellow aimed, green scoring, grey neither
export function Map({ dots, size = 150 }: { dots: Dot[]; size?: number }) {
    const ref = useRef<HTMLCanvasElement>(null)

    useEffect(() => {
        const ctx = ref.current!.getContext('2d')!
        ctx.fillStyle = '#0d1018'
        ctx.fillRect(0, 0, size, size)
        if (!dots.length) return

        const cx = dots.reduce((s, d) => s + d.x, 0) / dots.length
        const cz = dots.reduce((s, d) => s + d.z, 0) / dots.length
        let span = 8
        for (const d of dots) span = Math.max(span, Math.abs(d.x - cx) * 2, Math.abs(d.z - cz) * 2)
        span *= 1.1

        for (const d of dots) {
            const px = size / 2 + ((d.x - cx) / span) * size
            const pz = size / 2 + ((d.z - cz) / span) * size
            ctx.fillStyle = d.score > 0 ? '#a6e3a1' : d.look ? '#f9e2af' : '#5b6378'
            ctx.beginPath()
            ctx.arc(px, pz, 2.5, 0, 7)
            ctx.fill()
        }
    }, [dots, size])

    return <canvas ref={ref} width={size} height={size} style={{ borderRadius: 6 }} />
}
