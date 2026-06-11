import * as THREE from 'three'
import type { Agent, Palette } from '../types.ts'

// mcgym registry block ids
export const WATER = 35
export const LAVA  = 36

// face corners as lo/hi picks so any aabb renders, 0 = lo, 1 = hi
const FACES = [
    { n: [1, 0, 0],  c: [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]] },
    { n: [-1, 0, 0], c: [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]] },
    { n: [0, 1, 0],  c: [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]] },
    { n: [0, -1, 0], c: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]] },
    { n: [0, 0, 1],  c: [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]] },
    { n: [0, 0, -1], c: [[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]] },
]
const TRI = [0, 1, 2, 0, 2, 3]

// collision aabb in 16ths, no collision renders as the full cube (water, plants)
function aabb(cell: number[]): [number[], number[], boolean] {
    const [x0, y0, z0, x1, y1, z1] = cell.length >= 10 ? cell.slice(4, 10) : []
    if (!(x1 > x0)) return [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], true]

    const lo   = [x0 / 16 - 0.5, y0 / 16 - 0.5, z0 / 16 - 0.5]
    const hi   = [x1 / 16 - 0.5, y1 / 16 - 0.5, z1 / 16 - 0.5]
    const full = x0 === 0 && y0 === 0 && z0 === 0 && x1 === 16 && y1 === 16 && z1 === 16
    return [lo, hi, full]
}

const key = (x: number, y: number, z: number) => (x + 8) + (y + 8) * 17 + (z + 8) * 289

export interface Scene {
    root:        THREE.Scene
    steve:       THREE.Group
    arrow:       THREE.Mesh
    update:      (data: Agent, palette: Palette) => void
    transparent: (on: boolean) => void
}

// reusable scene of one agents near voxel grid, callers toggle steve and the
// arrow per pass, blocks use front side so back faces never draw
export function scene(): Scene {
    const root = new THREE.Scene()
    root.background = new THREE.Color(0x0b0e14)
    root.add(new THREE.AmbientLight(0xffffff, 0.8))

    const sun = new THREE.DirectionalLight(0xffffff, 0.6)
    sun.position.set(0.6, 1, 0.4)
    root.add(sun)

    // exposed face mesh, only faces with an empty neighbour are emitted
    const mat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.FrontSide })
    const geo = new THREE.BufferGeometry()
    root.add(new THREE.Mesh(geo, mat))

    function build(cells: number[][], palette: Palette) {
        // only full cubes occlude, partial shapes keep all their faces
        const occ = new Set<number>()
        for (const c of cells) {
            if (aabb(c)[2]) occ.add(key(c[0], c[1], c[2]))
        }

        const pos: number[] = []
        const nor: number[] = []
        const col: number[] = []
        const tmp = new THREE.Color()
        for (const c of cells) {
            const [dx, dy, dz, id] = c
            const [lo, hi, full]   = aabb(c)
            tmp.set(palette[String(id)] || '#5a5a5a')
            for (const f of FACES) {
                const [nx, ny, nz] = f.n
                const bx = dx + nx, by = dy + ny, bz = dz + nz
                const inside = bx >= -8 && bx <= 8 && by >= -8 && by <= 8 && bz >= -8 && bz <= 8
                if (full && inside && occ.has(key(bx, by, bz))) continue

                for (const vi of TRI) {
                    const v = f.c[vi]
                    pos.push(dx + (v[0] ? hi[0] : lo[0]),
                             dy + (v[1] ? hi[1] : lo[1]),
                             dz + (v[2] ? hi[2] : lo[2]))
                    nor.push(nx, ny, nz)
                    col.push(tmp.r, tmp.g, tmp.b)
                }
            }
        }

        geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
        geo.setAttribute('normal',   new THREE.Float32BufferAttribute(nor, 3))
        geo.setAttribute('color',    new THREE.Float32BufferAttribute(col, 3))
        geo.computeBoundingSphere()
    }

    // steve, legs + body + head that yaws and pitches, nose marks facing
    const steve = new THREE.Group()
    const skin  = new THREE.MeshLambertMaterial({ color: 0xc28a5a })
    const shirt = new THREE.MeshLambertMaterial({ color: 0x32b0ef })
    const pants = new THREE.MeshLambertMaterial({ color: 0x3550a0 })

    const legs = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.7, 0.28), pants)
    legs.position.y = 0.45
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.6, 0.28), shirt)
    body.position.y = 1.1

    const pivot = new THREE.Group()
    pivot.position.y = 1.5
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.45, 0.45, 0.45), skin)
    const nose = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.14, 0.09),
                                new THREE.MeshLambertMaterial({ color: 0xf0c69a }))
    nose.position.set(0, 0, 0.27)
    pivot.add(head, nose)
    steve.add(legs, body, pivot)
    root.add(steve)

    // facing arrow for the top down view, flat triangle above the terrain
    const tri = new THREE.Shape()
    tri.moveTo(0, 1.6); tri.lineTo(-0.9, -1); tri.lineTo(0, -0.4); tri.lineTo(0.9, -1); tri.lineTo(0, 1.6)
    const arrow = new THREE.Mesh(
        new THREE.ShapeGeometry(tri),
        new THREE.MeshBasicMaterial({ color: 0xff5555, side: THREE.DoubleSide, depthTest: false }))
    arrow.rotation.x = -Math.PI / 2
    arrow.position.y = 11
    arrow.visible = false
    root.add(arrow)

    const outline = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(1.04, 1.04, 1.04)),
        new THREE.LineBasicMaterial({ color: 0x000000 }))
    outline.visible = false
    root.add(outline)

    const breaking = new THREE.Mesh(
        new THREE.BoxGeometry(1.06, 1.06, 1.06),
        new THREE.MeshBasicMaterial({ color: 0x111111, transparent: true, opacity: 0.45, depthWrite: false }))
    breaking.visible = false
    root.add(breaking)

    function update(data: Agent, palette: Palette) {
        build(data.cells || [], palette)

        const yaw = THREE.MathUtils.degToRad(data.yaw || 0)
        steve.rotation.y = -yaw
        pivot.rotation.x = THREE.MathUtils.degToRad(data.pitch || 0)
        arrow.rotation.z = yaw

        if (data.targetPos) {
            const [tx, ty, tz] = data.targetPos
            outline.visible = true
            outline.position.set(tx, ty, tz)
            breaking.position.set(tx, ty, tz)
            breaking.visible = !!data.attacking
        } else {
            outline.visible = false
            breaking.visible = false
        }
    }

    function transparent(on: boolean) {
        mat.transparent = on
        mat.opacity = on ? 0.3 : 1.0
        mat.depthWrite = !on
        mat.needsUpdate = true
    }

    return { root, steve, arrow, update, transparent }
}

export function look(yaw: number, pitch: number): THREE.Vector3 {
    const y  = THREE.MathUtils.degToRad(yaw)
    const p  = THREE.MathUtils.degToRad(pitch)
    const cp = Math.cos(p)
    return new THREE.Vector3(-Math.sin(y) * cp, -Math.sin(p), Math.cos(y) * cp)
}

// pov overlay color when the agents eye sits in a fluid
export function tint(head: number | undefined): string | null {
    if (head === WATER) return 'rgba(40,90,210,0.42)'
    if (head === LAVA)  return 'rgba(235,110,20,0.5)'
    return null
}
