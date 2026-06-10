// The monitor reports the instantaneous tick, but the agent's attack presses and aim wobble
// in and out between polls (attack ~65% of ticks, aimed ~38%, both ~25%), so the target
// outline and breaking overlay blink. Remember each agent's last aimed block and last attack
// briefly and merge them, so the overlay persists while the agent is actively working a block.
const TTL_MS = 1000
const mem = new Map() // "env:i" -> { pos, posT, atkT }

export function smoothAgent(key, d) {
  const now = performance.now()
  const m = mem.get(key) || {}
  if (d.targetPos) { m.pos = d.targetPos; m.posT = now }
  if (d.attacking) m.atkT = now
  mem.set(key, m)
  const targetPos = d.targetPos || (now - (m.posT ?? -Infinity) < TTL_MS ? m.pos : null)
  const attacking = targetPos && (d.attacking || now - (m.atkT ?? -Infinity) < TTL_MS) ? 1 : 0
  return { ...d, targetPos, attacking }
}
