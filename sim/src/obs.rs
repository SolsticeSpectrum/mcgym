//! observation extraction, azalea ecs + shared client world -> schema Obs

use std::sync::Arc;

use azalea_client::local_player::{Hunger, WorldHolder};
use azalea_core::position::{BlockPos, ChunkBlockPos, ChunkPos};
use azalea_entity::inventory::Inventory;
use azalea_entity::metadata::Health;
use azalea_entity::{LookDirection, Physics, Position};
use azalea_inventory::ItemStack;
use azalea_world::{Chunk, World};
use bevy_ecs::entity::Entity;
use parking_lot::RwLock;

use crate::registry::Registry;
use crate::schema::{BOUNDS_DIMS, INVENTORY_SLOTS, Obs, VOXEL_EDGE, VOXEL_FAR_STRIDE, VOXEL_RADIUS};

const EYE_HEIGHT:      f64 = 1.62;
pub const BLOCK_REACH: f64 = 4.5;

const R: i32 = VOXEL_RADIUS as i32;
const E: i32 = VOXEL_EDGE as i32;

#[inline]
fn cell_index(dx: i32, dy: i32, dz: i32) -> usize {
    ((((dy + R) * E) + (dz + R)) * E + (dx + R)) as usize
}

// near grid, blocks and bounds in one pass, one chunk lock per overlapped chunk
fn fill_near(world: &World, reg: &Registry, c: BlockPos, blocks: &mut [i32], bounds: &mut [u8]) {
    let min_y = world.chunks.min_y();
    for ccx in (c.x - R) >> 4..=(c.x + R) >> 4 {
        for ccz in (c.z - R) >> 4..=(c.z + R) >> 4 {
            let Some(chunk) = world.chunks.get(&ChunkPos::new(ccx, ccz)) else {
                continue; // unloaded stays air
            };
            let chunk = chunk.read();

            let x0 = (c.x - R).max(ccx * 16);
            let x1 = (c.x + R).min(ccx * 16 + 15);
            let z0 = (c.z - R).max(ccz * 16);
            let z1 = (c.z + R).min(ccz * 16 + 15);
            for y in c.y - R..=c.y + R {
                for z in z0..=z1 {
                    for x in x0..=x1 {
                        let Some(state) =
                            chunk.get_block_state(&ChunkBlockPos::from(&BlockPos::new(x, y, z)), min_y)
                        else {
                            continue;
                        };
                        let at     = cell_index(x - c.x, y - c.y, z - c.z);
                        blocks[at] = reg.block(state);
                        bounds[at * BOUNDS_DIMS..at * BOUNDS_DIMS + BOUNDS_DIMS]
                            .copy_from_slice(reg.bounds(state));
                    }
                }
            }
        }
    }
}

// far shell, stride 4, chunk arc cached between neighbouring cells
fn fill_far(world: &World, reg: &Registry, c: BlockPos, out: &mut [i32]) {
    let min_y = world.chunks.min_y();
    let mut last: (i32, i32, Option<Arc<RwLock<Chunk>>>) = (i32::MIN, i32::MIN, None);
    let mut at = 0;
    for dy in -R..=R {
        for dz in -R..=R {
            for dx in -R..=R {
                let pos = BlockPos::new(
                    c.x + dx * VOXEL_FAR_STRIDE,
                    c.y + dy * VOXEL_FAR_STRIDE,
                    c.z + dz * VOXEL_FAR_STRIDE,
                );
                let (ccx, ccz) = (pos.x >> 4, pos.z >> 4);
                if (ccx, ccz) != (last.0, last.1) {
                    last = (ccx, ccz, world.chunks.get(&ChunkPos::new(ccx, ccz)));
                }
                out[at] = last.2.as_ref().map_or(0, |chunk| {
                    chunk
                        .read()
                        .get_block_state(&ChunkBlockPos::from(&pos), min_y)
                        .map_or(0, |s| reg.block(s))
                });
                at += 1;
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
pub fn build_obs(ecs: &bevy_ecs::world::World, id: Entity, reg: &Registry, i: usize, tick: i64) -> Obs {
    let pos    = ecs.get::<Position>(id).expect("position");
    let phys   = ecs.get::<Physics>(id).expect("physics");
    let look   = ecs.get::<LookDirection>(id).expect("look");
    let health = ecs.get::<Health>(id).expect("health");
    let hunger = ecs.get::<Hunger>(id).expect("hunger");
    let inv    = ecs.get::<Inventory>(id).expect("inventory");
    let holder = ecs.get::<WorldHolder>(id).expect("world holder");

    let mut o = Obs {
        tick,
        agent_id:      i as i32,
        pos:           [pos.x as f32, pos.y as f32, pos.z as f32],
        vel:           [phys.velocity.x as f32, phys.velocity.y as f32, phys.velocity.z as f32],
        yaw:           look.y_rot(),
        pitch:         look.x_rot(),
        on_ground:     u8::from(phys.on_ground()),
        health:        **health,
        food:          hunger.food as f32,
        selected_slot: inv.selected_hotbar_slot,
        ..Default::default()
    };

    let world  = holder.shared.read();
    let center = BlockPos::new(pos.x.floor() as i32, pos.y.floor() as i32, pos.z.floor() as i32);
    fill_near(&world, reg, center, &mut o.voxel_blocks, &mut o.voxel_bounds);
    fill_far(&world, reg, center, &mut o.voxel_far);

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
