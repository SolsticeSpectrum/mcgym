import * as THREE from 'three'

const EDGE = 17
const MAX = EDGE * EDGE * EDGE

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

  const blockMat = new THREE.MeshLambertMaterial({ side: THREE.FrontSide })
  const blocks = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), blockMat, MAX)
  blocks.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
  blocks.count = 0
  scene.add(blocks)

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

  const dummy = new THREE.Object3D()
  const col = new THREE.Color()
  const colorFor = (id, palette) => {
    const c = palette[String(id)]
    return c && c.length ? c : '#5a5a5a'
  }

  function update(data, palette) {
    const cells = data.cells || []
    blocks.count = cells.length
    for (let k = 0; k < cells.length; k++) {
      const [dx, dy, dz, id] = cells[k]
      dummy.position.set(dx, dy, dz)
      dummy.updateMatrix()
      blocks.setMatrixAt(k, dummy.matrix)
      col.set(colorFor(id, palette))
      blocks.setColorAt(k, col)
    }
    blocks.instanceMatrix.needsUpdate = true
    if (blocks.instanceColor) blocks.instanceColor.needsUpdate = true

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
