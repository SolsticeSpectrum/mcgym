import type { Agent } from '../types.ts'

// the monitor reports the instantaneous tick but attack and aim wobble between
// polls, remember each agents last aimed block and last attack briefly so the
// outline and breaking overlay dont blink
const TTL = 1000
const mem = new Map<string, { pos?: number[]; t?: number; atk?: number }>()

export function smooth(id: string, d: Agent): Agent {
    const now = performance.now()
    const m = mem.get(id) || {}
    if (d.targetPos) { m.pos = d.targetPos; m.t = now }
    if (d.attacking) m.atk = now
    mem.set(id, m)

    const pos = d.targetPos || (now - (m.t ?? -Infinity) < TTL ? m.pos! : null)
    const atk = pos && (d.attacking || now - (m.atk ?? -Infinity) < TTL) ? 1 : 0
    return { ...d, targetPos: pos, attacking: atk }
}
