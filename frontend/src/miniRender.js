import * as THREE from 'three'
import { createVoxelScene, lookDir } from './voxelScene.js'

const EYE = 1.62
let R = null // one shared renderer/scene for all mini views (no per-view WebGL contexts)

function get() {
  if (R) return R
  const renderer = new THREE.WebGLRenderer({ antialias: true })
  const handles = createVoxelScene()
  const povCam = new THREE.PerspectiveCamera(75, 1.5, 0.05, 500)
  // Match the 17-cell grid extent (cells -8..8, cubes +/-0.5 -> +/-8.5) so the terrain fills the
  // square with no padding.
  const topCam = new THREE.OrthographicCamera(-8.5, 8.5, 8.5, -8.5, 0.1, 500)
  topCam.position.set(0, 30, 0)
  topCam.up.set(0, 0, -1)
  topCam.lookAt(0, 0, 0)
  R = { renderer, povCam, topCam, ...handles }
  return R
}

// POV (Steve hidden, first-person) into povCanvas; top-down (Steve hidden, facing arrow shown)
// into topCanvas. Always opaque — the transparency toggle only affects the full 3D view.
export function drawAgentMini(data, palette, povCanvas, topCanvas) {
  const { renderer, scene, update, steve, arrow, povCam, topCam } = get()
  update(data, palette)
  steve.visible = false
  if (povCanvas) {
    const w = povCanvas.width, h = povCanvas.height
    arrow.visible = false
    const dir = lookDir(data.yaw || 0, data.pitch || 0)
    povCam.aspect = w / h
    povCam.position.set(0, EYE, 0)
    povCam.lookAt(dir.x, EYE + dir.y, dir.z)
    povCam.updateProjectionMatrix()
    renderer.setSize(w, h, false)
    renderer.render(scene, povCam)
    povCanvas.getContext('2d').drawImage(renderer.domElement, 0, 0, w, h)
  }
  if (topCanvas) {
    const s = topCanvas.width
    arrow.visible = true
    renderer.setSize(s, s, false)
    renderer.render(scene, topCam)
    topCanvas.getContext('2d').drawImage(renderer.domElement, 0, 0, s, s)
  }
}
