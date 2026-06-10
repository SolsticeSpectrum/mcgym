import * as THREE from 'three'

const EDGE = 17
const MAX = EDGE * EDGE * EDGE

// A reusable three.js scene of one agent's near voxel grid: instanced blocks, a Steve model
// showing facing, a black target-block outline, and a breaking overlay. Used by both the full
// agent view and the small per-agent cards (rendered with one shared renderer).
export function createVoxelScene() {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x0b0e14)
  scene.add(new THREE.AmbientLight(0xffffff, 0.75))
  const sun = new THREE.DirectionalLight(0xffffff, 0.65)
  sun.position.set(0.6, 1, 0.4)
  scene.add(sun)

  const blockMat = new THREE.MeshLambertMaterial()
  const blocks = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), blockMat, MAX)
  blocks.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
  blocks.count = 0
  scene.add(blocks)

  // Steve: body + a head that yaws/pitches, with a nose so facing is unambiguous. Feet sit on
  // top of the centre cell (the block the agent stands in is at y=0, so feet ~ y=0.5).
  const steve = new THREE.Group()
  const skin = new THREE.MeshLambertMaterial({ color: 0xb07a4f })
  const shirt = new THREE.MeshLambertMaterial({ color: 0x3aa0e0 })
  const legs = new THREE.MeshLambertMaterial({ color: 0x3b5da8 })
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.55, 0.26), shirt)
  body.position.y = 0.95
  const legMesh = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.6, 0.26), legs)
  legMesh.position.y = 0.4
  const headPivot = new THREE.Group()
  headPivot.position.y = 1.32
  const head = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.42, 0.42), skin)
  const nose = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.12, 0.08), new THREE.MeshLambertMaterial({ color: 0xe7b98c }))
  nose.position.set(0, 0, 0.25)
  headPivot.add(head, nose)
  steve.add(legMesh, body, headPivot)
  scene.add(steve)

  const outline = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(1.04, 1.04, 1.04)),
    new THREE.LineBasicMaterial({ color: 0x000000, linewidth: 2 }),
  )
  outline.visible = false
  scene.add(outline)

  const breakBox = new THREE.Mesh(
    new THREE.BoxGeometry(1.06, 1.06, 1.06),
    new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.4, depthWrite: false }),
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

    steve.rotation.y = -THREE.MathUtils.degToRad(data.yaw || 0)
    headPivot.rotation.x = THREE.MathUtils.degToRad(data.pitch || 0)

    if (data.targetPos) {
      const [tx, ty, tz] = data.targetPos
      outline.visible = true
      outline.position.set(tx, ty, tz)
      breakBox.position.set(tx, ty, tz)
      breakBox.visible = !!data.attacking
      if (data.attacking) {
        // Approximate breaking progress by pulsing the overlay (real per-tick progress isn't in
        // the obs yet); shrinks the dark box so it "cracks in".
        const t = (Date.now() % 1000) / 1000
        breakBox.scale.setScalar(0.5 + 0.5 * t)
        breakBox.material.opacity = 0.25 + 0.45 * t
      }
    } else {
      outline.visible = false
      breakBox.visible = false
    }
  }

  function setTransparent(on) {
    blockMat.transparent = on
    blockMat.opacity = on ? 0.32 : 1.0
    blockMat.depthWrite = !on
    blockMat.needsUpdate = true
  }

  return { scene, update, setTransparent }
}

// Minecraft look vector from yaw/pitch (degrees); yaw 0 = +Z.
export function lookDir(yawDeg, pitchDeg) {
  const yaw = THREE.MathUtils.degToRad(yawDeg)
  const pitch = THREE.MathUtils.degToRad(pitchDeg)
  const cp = Math.cos(pitch)
  return new THREE.Vector3(-Math.sin(yaw) * cp, -Math.sin(pitch), Math.cos(yaw) * cp)
}
