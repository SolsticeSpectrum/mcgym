// /state, columnar vectors for every agent, env of agent g is g / per
export interface State {
    step:  number
    per:   number
    edge:  number
    x:     number[]
    y:     number[]
    z:     number[]
    yaw:   number[]
    score: number[]
    look:  number[]
}

// /agent, one agents voxel view plus pose
export interface Agent {
    env:       number
    i:         number
    yaw:       number
    pitch:     number
    score:     number
    look:      number
    target:    number
    targetPos: number[] | null
    attacking: number
    head:      number
    cells:     number[][]
}

export type Palette = Record<string, string>
