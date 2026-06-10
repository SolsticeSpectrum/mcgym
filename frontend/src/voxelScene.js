import * as THREE from 'three'

const EDGE = 17
const MAX = EDGE * EDGE * EDGE
// MCAI registry block ids (schema/registry.json).
export const WATER = 35
export const LAVA = 36

// A reusable three.js scene of one agent's near voxel grid. Exposes handles so callers can
// toggle Steve (hidden in POV/top-down) and the facing arrow (shown only top-down), and set
// per-pass transparency. Blocks use FrontSide so back faces are never drawn.
export function createVoxelScene() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x0b0e14)
  scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const sun = new THREE.DirectionalLight(0xffffff, 0.6)
  sun.position.set(0.6, 1, 0.4)
  scene.add(sun)

  // Exposed-face mesh: only faces whose neighbour cell is empty are emitted, so interior faces
  // (shared between adjacent solid blocks) are never drawn — correct even in transparent mode.
  const blockMat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.FrontSide })
  const blockGeo = new THREE.BufferGeometry()
  const blocks = new THREE.Mesh(blockGeo, blockMat)
  scene.add(blocks)
  // 6 faces as CCW-outward quads (triangulated 0,1,2,0,2,3) with their outward normal.
  const FACES = [
    { n: [1, 0, 0], c: [[0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]] },
    { n: [-1, 0, 0], c: [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]] },
    { n: [0, 1, 0], c: [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]] },
    { n: [0, -1, 0], c: [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]] },
    { n: [0, 0, 1], c: [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]] },
    { n: [0, 0, -1], c: [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]] },
  ]
  const TRI = [0, 1, 2, 0, 2, 3]
  const key = (x, y, z) => (x + 8) + (y + 8) * 17 + (z + 8) * 289
  const colorFor = (id, palette) => {
    const c = palette[String(id)]
    return c && c.length ? c : '#5a5a5a'
  }

  function buildBlocks(cells, palette) {
    const occ = new Map()
    for (const c of cells) occ.set(key(c[0], c[1], c[2]), c[3])
    const pos = []
    const nor = []
    const colr = []
    const tmp = new THREE.Color()
    for (const [dx, dy, dz, id] of cells) {
      tmp.set(colorFor(id, palette))
      for (const f of FACES) {
        const [nx, ny, nz] = f.n
        const bx = dx + nx, by = dy + ny, bz = dz + nz
        const inside = bx >= -8 && bx <= 8 && by >= -8 && by <= 8 && bz >= -8 && bz <= 8
        if (inside) {
          const nid = occ.get(key(bx, by, bz))
          // Cull only behind an opaque neighbour, or between two of the same fluid. Water/lava
          // don't occlude (see through them), but their faces exposed to air always render.
          if (nid !== undefined && ((nid !== WATER && nid !== LAVA) || nid === id)) continue
        }
        for (const vi of TRI) {
          const v = f.c[vi]
          pos.push(dx + v[0], dy + v[1], dz + v[2])
          nor.push(nx, ny, nz)
          colr.push(tmp.r, tmp.g, tmp.b)
        }
      }
    }
    blockGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
    blockGeo.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3))
    blockGeo.setAttribute('color', new THREE.Float32BufferAttribute(colr, 3))
    blockGeo.computeBoundingSphere()
  }

  // Steve: legs + body + a head that yaws/pitches, nose for facing. Feet on top of the centre cell.
  const steve = new THREE.Group()
  const skin = new THREE.MeshLambertMaterial({ color: 0xc28a5a })
  const shirt = new THREE.MeshLambertMaterial({ color: 0x32b0ef })
  const legs = new THREE.MeshLambertMaterial({ color: 0x3550a0 })
  const legMesh = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.7, 0.28), legs); legMesh.position.y = 0.45
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.6, 0.28), shirt); body.position.y = 1.1
  const headPivot = new THREE.Group(); headPivot.position.y = 1.5
  const head = new THREE.Mesh(new THREE.BoxGeometry(0.45, 0.45, 0.45), skin)
  const nose = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.14, 0.09), new THREE.MeshLambertMaterial({ color: 0xf0c69a }))
  nose.position.set(0, 0, 0.27)
  headPivot.add(head, nose)
  steve.add(legMesh, body, headPivot)
  scene.add(steve)

  // Facing arrow for the top-down view: a flat triangle above the terrain pointing along the
  // agent's heading. Hidden in 3D/POV.
  const arrowShape = new THREE.Shape()
  arrowShape.moveTo(0, 1.6); arrowShape.lineTo(-0.9, -1); arrowShape.lineTo(0, -0.4); arrowShape.lineTo(0.9, -1); arrowShape.lineTo(0, 1.6)
  const arrow = new THREE.Mesh(
    new THREE.ShapeGeometry(arrowShape),
    new THREE.MeshBasicMaterial({ color: 0xff5555, side: THREE.DoubleSide, depthTest: false }),
  )
  arrow.rotation.x = -Math.PI / 2 // lay flat in XZ
  arrow.position.y = 11
  arrow.visible = false
  scene.add(arrow)

  const outline = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(1.04, 1.04, 1.04)),
    new THREE.LineBasicMaterial({ color: 0x000000 }),
  )
  outline.visible = false
  scene.add(outline)

  const breakBox = new THREE.Mesh(
    new THREE.BoxGeometry(1.06, 1.06, 1.06),
    new THREE.MeshBasicMaterial({ color: 0x111111, transparent: true, opacity: 0.45, depthWrite: false }),
  )
  breakBox.visible = false
  scene.add(breakBox)

  function update(data, palette) {
    buildBlocks(data.cells || [], palette)

    const yaw = THREE.MathUtils.degToRad(data.yaw || 0)
    steve.rotation.y = -yaw
    headPivot.rotation.x = THREE.MathUtils.degToRad(data.pitch || 0)
    arrow.rotation.z = yaw // about the now-flat plane's normal -> heading in XZ

    if (data.targetPos) {
      const [tx, ty, tz] = data.targetPos
      outline.visible = true
      outline.position.set(tx, ty, tz)
      breakBox.position.set(tx, ty, tz)
      breakBox.visible = !!data.attacking
    } else {
      outline.visible = false
      breakBox.visible = false
    }
  }

  function setTransparent(on) {
    blockMat.transparent = on
    blockMat.opacity = on ? 0.3 : 1.0
    blockMat.depthWrite = !on
    blockMat.needsUpdate = true
  }

  return { scene, update, setTransparent, steve, arrow, outline, breakBox }
}

export function lookDir(yawDeg, pitchDeg) {
  const yaw = THREE.MathUtils.degToRad(yawDeg)
  const pitch = THREE.MathUtils.degToRad(pitchDeg)
  const cp = Math.cos(pitch)
  return new THREE.Vector3(-Math.sin(yaw) * cp, -Math.sin(pitch), Math.cos(yaw) * cp)
}
