//! observation extraction, azalea ecs + shared client world -> schema Obs

use azalea_client::local_player::{Hunger, WorldHolder};
use azalea_core::position::BlockPos;
use azalea_entity::metadata::Health;
use azalea_entity::inventory::Inventory;
use azalea_entity::{LookDirection, Physics, Position};
use azalea_inventory::ItemStack;
use azalea_world::World;

use crate::client::Swarm;
use crate::registry::Registry;
use crate::schema::{BOUNDS_DIMS, INVENTORY_SLOTS, Obs, VOXEL_EDGE, VOXEL_FAR_STRIDE, VOXEL_RADIUS};

const EYE_HEIGHT:      f64 = 1.62;
pub const BLOCK_REACH: f64 = 4.5;

// near and far grids share the same y z x cell order as the java gym
fn fill_voxels(world: &World, reg: &Registry, c: BlockPos, stride: i32, out: &mut [i32]) {
    let r = VOXEL_RADIUS as i32;
    let mut at = 0;
    for dy in -r..=r {
        for dz in -r..=r {
            for dx in -r..=r {
                let pos = BlockPos::new(c.x + dx * stride, c.y + dy * stride, c.z + dz * stride);
                out[at] = world
                    .chunks
                    .get_block_state(pos)
                    .map_or(0, |s| reg.block(s));
                at += 1;
            }
        }
    }
}

fn fill_bounds(world: &World, reg: &Registry, c: BlockPos, out: &mut [u8]) {
    let r = VOXEL_RADIUS as i32;
    let mut at = 0;
    for dy in -r..=r {
        for dz in -r..=r {
            for dx in -r..=r {
                let pos = BlockPos::new(c.x + dx, c.y + dy, c.z + dz);
                let cell = world
                    .chunks
                    .get_block_state(pos)
                    .map_or([0; BOUNDS_DIMS], |s| *reg.bounds(s));
                out[at..at + BOUNDS_DIMS].copy_from_slice(&cell);
                at += BOUNDS_DIMS;
            }
        }
    }
}

// vanilla convention, yaw 0 => +z
fn look_dir(yaw: f32, pitch: f32) -> [f64; 3] {
    let (yaw, pitch) = (f64::from(yaw).to_radians(), f64::from(pitch).to_radians());
    let f = pitch.cos();
    [-yaw.sin() * f, -pitch.sin(), yaw.cos() * f]
}

fn solid(world: &World, reg: &Registry, pos: BlockPos) -> bool {
    world.chunks.get_block_state(pos).is_some_and(|s| reg.solid(s))
}

// voxel dda, face uses mc direction order down,up,north,south,west,east = 0..5
fn raycast(world: &World, reg: &Registry, eye: [f64; 3], dir: [f64; 3], reach: f64)
    -> Option<(BlockPos, u8, f64)>
{
    let mut b = [eye[0].floor() as i32, eye[1].floor() as i32, eye[2].floor() as i32];
    let step  = [dir[0].signum() as i32, dir[1].signum() as i32, dir[2].signum() as i32];

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
        let axis = if t_max[0] < t_max[1] && t_max[0] < t_max[2] { 0 }
                   else if t_max[1] < t_max[2] { 1 }
                   else { 2 };
        b[axis]    += step[axis];
        t           = t_max[axis];
        t_max[axis] += t_delta[axis];

        if t > reach {
            break;
        }
        let pos = BlockPos::new(b[0], b[1], b[2]);
        if solid(world, reg, pos) {
            return Some((pos, face_for(axis, step[axis]), t));
        }
    }

    None
}

// player menu 46 slots -> schema 41, hotbar 0..8, main 9..35, armor 36..39, offhand 40
fn fill_inventory(menu: &azalea_inventory::Menu, reg: &Registry, ids: &mut [i32; 41], counts: &mut [u8; 41]) {
    let slots = menu.slots();
    for (dst, src) in (0..INVENTORY_SLOTS).zip(
        (36..45).chain(9..36).chain((5..9).rev()).chain(std::iter::once(45)),
    ) {
        if let ItemStack::Present(stack) = &slots[src] {
            ids[dst]    = reg.item(stack.kind);
            counts[dst] = stack.count.clamp(0, 255) as u8;
        } else {
            ids[dst]    = 0;
            counts[dst] = 0;
        }
    }
}

// entity slots stay zero until mobs land in the obs contract
pub fn build_obs(swarm: &Swarm, reg: &Registry, i: usize, tick: i64) -> Obs {
    let pos:    Position      = swarm.get(i);
    let phys:   Physics       = swarm.get(i);
    let look:   LookDirection = swarm.get(i);
    let health: Health        = swarm.get(i);
    let hunger: Hunger        = swarm.get(i);
    let inv:    Inventory     = swarm.get(i);
    let holder: WorldHolder   = swarm.get(i);

    let mut o = Obs {
        tick,
        agent_id:      i as i32,
        pos:           [pos.x as f32, pos.y as f32, pos.z as f32],
        vel:           [phys.velocity.x as f32, phys.velocity.y as f32, phys.velocity.z as f32],
        yaw:           look.y_rot(),
        pitch:         look.x_rot(),
        on_ground:     u8::from(phys.on_ground()),
        health:        *health,
        food:          hunger.food as f32,
        selected_slot: inv.selected_hotbar_slot,
        ..Default::default()
    };

    let world  = holder.shared.read();
    let center = BlockPos::new(pos.x.floor() as i32, pos.y.floor() as i32, pos.z.floor() as i32);
    fill_voxels(&world, reg, center, 1,                &mut o.voxel_blocks);
    fill_voxels(&world, reg, center, VOXEL_FAR_STRIDE, &mut o.voxel_far);
    fill_bounds(&world, reg, center, &mut o.voxel_bounds);

    let eye = [pos.x, pos.y + EYE_HEIGHT, pos.z];
    if let Some((hit, face, dist)) = raycast(&world, reg, eye, look_dir(o.yaw, o.pitch), BLOCK_REACH) {
        let state = world.chunks.get_block_state(hit).expect("raycast hit an unloaded block");
        o.target_block    = reg.block(state);
        o.target_face     = face;
        o.target_distance = dist as f32;
        o.target_in_range = 1;
    }
    drop(world);

    fill_inventory(&inv.inventory_menu, reg, &mut o.inv_item_id, &mut o.inv_count);
    o
}

pub fn center_index() -> usize {
    let r = VOXEL_RADIUS;
    (r * VOXEL_EDGE + r) * VOXEL_EDGE + r
}
