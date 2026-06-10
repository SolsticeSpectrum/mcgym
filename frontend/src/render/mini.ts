import * as THREE from 'three'
import { scene, look, type Scene } from './scene.ts'
import type { Agent, Palette } from '../types.ts'

const EYE = 1.62

// one shared renderer and scene for every mini view, no per view gl contexts
let R: { renderer: THREE.WebGLRenderer; s: Scene; pov: THREE.PerspectiveCamera; top: THREE.OrthographicCamera } | null = null

function get() {
    if (R) return R

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    const s = scene()

    const pov = new THREE.PerspectiveCamera(75, 1.5, 0.05, 500)
    // ortho frustum matches the 17 cell grid so the terrain fills the square
    const top = new THREE.OrthographicCamera(-8.5, 8.5, 8.5, -8.5, 0.1, 500)
    top.position.set(0, 30, 0)
    top.up.set(0, 0, -1)
    top.lookAt(0, 0, 0)

    R = { renderer, s, pov, top }
    return R
}

// first person into povc, top down with the arrow into topc, always opaque
export function mini(data: Agent, palette: Palette, povc: HTMLCanvasElement | null, topc: HTMLCanvasElement | null) {
    const { renderer, s, pov, top } = get()
    s.update(data, palette)
    s.steve.visible = false

    if (povc) {
        const w = povc.width, h = povc.height
        s.arrow.visible = false

        const dir = look(data.yaw || 0, data.pitch || 0)
        pov.aspect = w / h
        pov.position.set(0, EYE, 0)
        pov.lookAt(dir.x, EYE + dir.y, dir.z)
        pov.updateProjectionMatrix()

        renderer.setSize(w, h, false)
        renderer.render(s.root, pov)
        povc.getContext('2d')!.drawImage(renderer.domElement, 0, 0, w, h)
    }

    if (topc) {
        const size = topc.width
        s.arrow.visible = true

        renderer.setSize(size, size, false)
        renderer.render(s.root, top)
        topc.getContext('2d')!.drawImage(renderer.domElement, 0, 0, size, size)
    }
}
