import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { Views } from '../components/views.tsx'
import { scene } from '../render/scene.ts'
import { smooth } from '../render/sticky.ts'
import type { Agent as Data, Palette } from '../types.ts'

const POLL = 350

// full agent page, orbitable 3d voxel view with steve plus a draggable and
// resizable minimap card in the corner
export function Agent({ env, i, palette }: { env: number; i: number; palette: Palette }) {
    const wrap = useRef<HTMLDivElement>(null)
    const api  = useRef<ReturnType<typeof scene> | null>(null)
    const [data, setData] = useState<Data | null>(null)
    const [clear, setClear] = useState(false)

    // minimap card drag and resize, w is both canvases, top down is h x h
    const [pos, setPos]   = useState(() => ({ x: Math.max(10, window.innerWidth - 352), y: 54 }))
    const [size, setSize] = useState({ w: 340, h: 120 })
    const drag = useRef<{ kind: 'move' | 'size'; sx: number; sy: number; px: number; py: number } | null>(null)

    useEffect(() => {
        const move = (e: PointerEvent) => {
            const d = drag.current
            if (!d) return
            if (d.kind === 'move') setPos({ x: d.px + (e.clientX - d.sx), y: d.py + (e.clientY - d.sy) })
            else setSize({ w: Math.max(220, d.px + (e.clientX - d.sx)), h: Math.max(90, d.py + (e.clientY - d.sy)) })
        }
        const up = () => { drag.current = null }

        window.addEventListener('pointermove', move)
        window.addEventListener('pointerup', up)
        return () => {
            window.removeEventListener('pointermove', move)
            window.removeEventListener('pointerup', up)
        }
    }, [])

    // own scene + renderer for the orbit view, steve stays visible here
    useEffect(() => {
        const mount = wrap.current!
        const s = scene()
        api.current = s

        const cam = new THREE.PerspectiveCamera(50, 2, 0.1, 500)
        cam.position.set(22, 17, 22)

        const renderer = new THREE.WebGLRenderer({ antialias: true })
        renderer.setPixelRatio(window.devicePixelRatio)
        mount.appendChild(renderer.domElement)

        const controls = new OrbitControls(cam, renderer.domElement)
        controls.target.set(0, 1, 0)
        controls.enableDamping = true

        const resize = () => {
            renderer.setSize(mount.clientWidth, mount.clientHeight)
            cam.aspect = mount.clientWidth / mount.clientHeight
            cam.updateProjectionMatrix()
        }
        resize()
        window.addEventListener('resize', resize)

        let raf = 0
        const render = () => {
            controls.update()
            renderer.render(s.root, cam)
            raf = requestAnimationFrame(render)
        }
        render()

        return () => {
            cancelAnimationFrame(raf)
            window.removeEventListener('resize', resize)
            controls.dispose()
            renderer.dispose()
            renderer.domElement.remove()
            api.current = null
        }
    }, [])

    useEffect(() => { api.current?.transparent(clear) }, [clear])

    useEffect(() => {
        let alive = true
        const tick = () =>
            fetch(`/agent?env=${env}&i=${i}`)
                .then((r) => r.json() as Promise<Data>)
                .then((raw) => {
                    if (!alive) return
                    const d = smooth(`${env}:${i}`, raw)
                    setData(d)
                    api.current?.update(d, palette)
                })
                .catch(() => {})
        tick()
        const id = setInterval(tick, POLL)
        return () => { alive = false; clearInterval(id) }
    }, [env, i, palette])

    const grab = (e: React.PointerEvent, kind: 'move' | 'size') => {
        drag.current = { kind, sx: e.clientX, sy: e.clientY,
                         px: kind === 'move' ? pos.x : size.w,
                         py: kind === 'move' ? pos.y : size.h }
        e.preventDefault()
        e.stopPropagation()
    }

    return (
        <div className="full">
            <div ref={wrap} style={{ position: 'absolute', inset: 0 }} />

            <div className="info">
                env {env} · agent {i}
                {data && <span className="dim"> — score {data.score} · yaw {data.yaw} · pitch {data.pitch}
                    {data.attacking ? ' · breaking' : data.look ? ' · aimed' : ''}</span>}
                <label className="dim" style={{ marginLeft: 14 }}>
                    <input type="checkbox" checked={clear} onChange={(e) => setClear(e.target.checked)} /> transparent
                </label>
            </div>

            <div className="mini" style={{ left: pos.x, top: pos.y }}>
                <div className="bar" onPointerDown={(e) => grab(e, 'move')}>
                    <span>pov</span><span style={{ color: '#5b6378' }}>drag · resize</span><span>top down</span>
                </div>
                <Views data={data} palette={palette} povw={Math.max(90, size.w - size.h)} h={size.h} />
                <div className="grip" onPointerDown={(e) => grab(e, 'size')} />
            </div>

            <div className="hint">drag to orbit · scroll to zoom</div>
        </div>
    )
}
