//! observation extraction, world + agent state -> schema Obs

use pumpkin_data::BlockState;

use crate::physics::Agent;
use crate::registry::Registry;
use crate::schema::{Obs, VOXEL_CELLS, VOXEL_EDGE, VOXEL_FAR_STRIDE, VOXEL_RADIUS};
use crate::world::World;

const EYE_HEIGHT:      f64 = 1.62;
pub const BLOCK_REACH: f64 = 4.5;

// stride 1 = near grid, 4 = far grid
pub fn fill_voxels(
    world:  &World,
    reg:    &Registry,
    cx:     i32,
    cy:     i32,
    cz:     i32,
    stride: i32,
    out:    &mut [i32],
) {
    debug_assert_eq!(out.len(), VOXEL_CELLS);
    // todo legit mode occlusion, enclosed cells should read HIDDEN_BLOCK_ID like the java gym
    world.fill_voxels(reg, cx, cy, cz, stride, out);
}

#[inline]
pub fn center_index() -> usize {
    let r    = VOXEL_RADIUS;
    let edge = VOXEL_EDGE;
    (r * edge + r) * edge + r
}

// vanilla convention, yaw 0 => +z
fn look_dir(yaw: f32, pitch: f32) -> [f64; 3] {
    let (yaw, pitch) = (f64::from(yaw).to_radians(), f64::from(pitch).to_radians());
    let f = pitch.cos();
    [-yaw.sin() * f, -pitch.sin(), yaw.cos() * f]
}

fn is_solid(world: &World, x: i32, y: i32, z: i32) -> bool {
    world
        .block_state_raw(x, y, z)
        .is_some_and(|sid| BlockState::from_id(sid).get_block_collision_shapes().count() > 0)
}

// voxel dda, face uses mc direction order down,up,north,south,west,east = 0..5
fn raycast(
    world: &World,
    eye:   [f64; 3],
    dir:   [f64; 3],
    reach: f64
) -> Option<([i32; 3], u8, f64)> {
    let mut b = [
        eye[0].floor() as i32,
        eye[1].floor() as i32,
        eye[2].floor() as i32,
    ];
    let step  = [
        dir[0].signum() as i32,
        dir[1].signum() as i32,
        dir[2].signum() as i32,
    ];

    let face_for = |axis: usize, s: i32| -> u8 {
        match (axis, s) {
            (0, 1) => 4,
            (0, _) => 5,
            (1, 1) => 0,
            (1, _) => 1,
            (2, 1) => 2,
            _      => 3,
        }
    };

    let mut t_max   = [0.0f64; 3];
    let mut t_delta = [0.0f64; 3];
    for a in 0..3 {
        if dir[a] == 0.0 {
            t_max[a]   = f64::INFINITY;
            t_delta[a] = f64::INFINITY;
        } else {
            let next = if dir[a] > 0.0 { f64::from(b[a]) + 1.0 - eye[a] }
                       else { eye[a] - f64::from(b[a]) };
            t_delta[a] = (1.0 / dir[a]).abs();
            t_max[a]   = next * t_delta[a];
        }
    }

    let mut t = 0.0;
    while t <= reach {
        // advance along the axis with the smallest t_max
        let axis = if t_max[0] < t_max[1] && t_max[0] < t_max[2] { 0 }
                   else if t_max[1] < t_max[2] { 1 }
                   else { 2 };
        b[axis]    += step[axis];
        t           = t_max[axis];
        t_max[axis] += t_delta[axis];

        if t > reach {
            break;
        }
        if is_solid(world, b[0], b[1], b[2]) {
            return Some((b, face_for(axis, step[axis]), t));
        }
    }

    None
}

pub fn raycast_target(
    world: &World,
    pos:   [f64; 3],
    yaw:   f32,
    pitch: f32,
    reach: f64,
) -> Option<([i32; 3], u8, f64)> {
    let eye = [pos[0], pos[1] + EYE_HEIGHT, pos[2]];
    raycast(world, eye, look_dir(yaw, pitch), reach)
}

// entity slots stay zero until mobs exist
pub fn build_obs(
    agent:    &Agent,
    world:    &World,
    reg:      &Registry,
    tick:     i64,
    agent_id: i32
) -> Obs {
    let mut o = Obs {
        tick,
        agent_id,
        pos:       [agent.pos[0] as f32, agent.pos[1] as f32, agent.pos[2] as f32],
        vel:       [agent.vel[0] as f32, agent.vel[1] as f32, agent.vel[2] as f32],
        yaw:       agent.yaw,
        pitch:     agent.pitch,
        on_ground: u8::from(agent.on_ground),
        health:    agent.health,
        food:      agent.food,
        ..Default::default()
    };

    let [cx, cy, cz] = agent.block_pos();
    fill_voxels(world, reg, cx, cy, cz, 1,                &mut o.voxel_blocks);
    fill_voxels(world, reg, cx, cy, cz, VOXEL_FAR_STRIDE, &mut o.voxel_far);

    if let Some(([hx, hy, hz], face, dist)) =
        raycast_target(world, agent.pos, agent.yaw, agent.pitch, BLOCK_REACH)
    {
        let sid = world.block_state_raw(hx, hy, hz).expect("raycast hit an unloaded block");
        o.target_block    = reg.block(sid);
        o.target_face     = face;
        o.target_distance = dist as f32;
        o.target_in_range = u8::from(dist <= BLOCK_REACH);
    }

    o.selected_slot = agent.selected_slot;
    o.inv_item_id   = agent.inv_item_id;
    o.inv_count     = agent.inv_count;
    o
}
