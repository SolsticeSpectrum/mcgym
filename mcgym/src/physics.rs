//! agent state + vanilla style player movement and aabb vs voxel collision

use pumpkin_data::BlockState;

use crate::schema::Action;
use crate::world::World;

const WIDTH:         f64 = 0.6;
const HALF_W:        f64 = WIDTH / 2.0;
const HEIGHT:        f64 = 1.8;
const GRAVITY:       f64 = 0.08;
const DRAG_Y:        f64 = 0.98;
const AIR_FRICTION:  f64 = 0.91;
const DEFAULT_SLIP:  f64 = 0.6;
const WALK_SPEED:    f64 = 0.1;
const SPRINT_MUL:    f64 = 1.3;
const JUMP_VELOCITY: f64 = 0.42;
const EPS:           f64 = 1.0e-7;

#[derive(Clone, Copy, Debug)]
struct Aabb {
    min: [f64; 3],
    max: [f64; 3],
}

impl Aabb {
    fn player(pos: [f64; 3]) -> Self {
        Self {
            min: [pos[0] - HALF_W, pos[1],          pos[2] - HALF_W],
            max: [pos[0] + HALF_W, pos[1] + HEIGHT, pos[2] + HALF_W],
        }
    }

    fn shift(&mut self, axis: usize, d: f64) {
        self.min[axis] += d;
        self.max[axis] += d;
    }

    // do the boxes overlap on the two axes other than ax
    fn overlaps_other(&self, b: &Self, ax: usize) -> bool {
        for &o in &[(ax + 1) % 3, (ax + 2) % 3] {
            if self.max[o] <= b.min[o] || self.min[o] >= b.max[o] {
                return false;
            }
        }

        true
    }

    // clamp signed motion d along ax so this box does not pass through b
    fn clamp_axis(&self, b: &Self, ax: usize, d: f64) -> f64 {
        if !self.overlaps_other(b, ax) {
            return d;
        }

        if d > 0.0 && self.max[ax] <= b.min[ax] + EPS {
            (b.min[ax] - self.max[ax]).min(d).max(0.0)
        } else if d < 0.0 && self.min[ax] >= b.max[ax] - EPS {
            (b.max[ax] - self.min[ax]).max(d).min(0.0)
        } else {
            d
        }
    }
}

pub const INV_SLOTS: usize = crate::schema::INVENTORY_SLOTS;

#[derive(Clone, Debug)]
pub struct Agent {
    pub pos:                [f64; 3],
    pub vel:                [f64; 3],
    pub yaw:                f32,
    pub pitch:              f32,
    pub on_ground:          bool,
    pub health:             f32,
    pub food:               f32,
    pub selected_slot:      u8,
    pub inv_item_id:        [i32; INV_SLOTS],
    pub inv_count:          [u8; INV_SLOTS],
    pub mine_target:        Option<[i32; 3]>, // block being mined
    pub mine_progress:      f64,              // break progress 0 to 1
    pub home:               [f64; 3],         // respawn point on death
    pub home_yaw:           f32,
    pub ref_xz:             [f64; 2],         // stuck tracking reference
    pub last_progress_tick: i64,
}

impl Agent {
    pub fn new(pos: [f64; 3], yaw: f32) -> Self {
        Self {
            pos,
            vel:                [0.0; 3],
            yaw,
            pitch:              0.0,
            on_ground:          false,
            health:             20.0,
            food:               20.0,
            selected_slot:      0,
            inv_item_id:        [0; INV_SLOTS],
            inv_count:          [0; INV_SLOTS],
            mine_target:        None,
            mine_progress:      0.0,
            home:               pos,
            home_yaw:           yaw,
            ref_xz:             [pos[0], pos[2]],
            last_progress_tick: 0,
        }
    }

    // stack onto a matching non full slot else first empty, false if full
    pub fn add_item(&mut self, id: i32) -> bool {
        for s in 0..INV_SLOTS {
            if self.inv_item_id[s] == id && self.inv_count[s] < 64 {
                self.inv_count[s] += 1;
                return true;
            }
        }

        for s in 0..INV_SLOTS {
            if self.inv_count[s] == 0 {
                self.inv_item_id[s] = id;
                self.inv_count[s]   = 1;
                return true;
            }
        }

        false
    }

    pub fn count_item(&self, id: i32) -> u32 {
        (0..INV_SLOTS)
            .filter(|&s| self.inv_item_id[s] == id)
            .map(|s| u32::from(self.inv_count[s]))
            .sum()
    }

    // feet block coords, the voxel grid centre
    pub fn block_pos(&self) -> [i32; 3] {
        [
            self.pos[0].floor() as i32,
            self.pos[1].floor() as i32,
            self.pos[2].floor() as i32,
        ]
    }

    // one tick: camera, moveRelative, jump, collide, gravity/friction
    pub fn step(&mut self, world: &World, act: &Action) {
        self.yaw  += act.yaw_delta;
        self.pitch = (self.pitch + act.pitch_delta).clamp(-90.0, 90.0);

        let friction = if self.on_ground { DEFAULT_SLIP * AIR_FRICTION }
                       else { AIR_FRICTION };
        let mut accel = if self.on_ground { WALK_SPEED * (0.216 / (friction * friction * friction)) }
                        else { 0.02 };
        if act.sprint != 0 {
            accel *= SPRINT_MUL;
        }
        self.move_relative(accel, f64::from(act.forward), f64::from(act.strafe));

        if act.jump != 0 && self.on_ground {
            self.vel[1] = JUMP_VELOCITY;
        }

        self.move_and_collide(world);

        self.vel[1]  = (self.vel[1] - GRAVITY) * DRAG_Y;
        self.vel[0] *= friction;
        self.vel[2] *= friction;
    }

    // vanilla moveRelative, yaw rotated input added to horizontal velocity
    fn move_relative(&mut self, amount: f64, forward: f64, strafe: f64) {
        let d2 = forward * forward + strafe * strafe;
        if d2 < EPS {
            return;
        }

        let scale = if d2 > 1.0 { amount / d2.sqrt() }
                    else { amount };
        let (f, s)     = (forward * scale, strafe * scale);
        let yaw        = f64::from(self.yaw).to_radians();
        let (sin, cos) = (yaw.sin(), yaw.cos());
        self.vel[0] += s * cos - f * sin;
        self.vel[2] += f * cos + s * sin;
    }

    // sweep the player box by vel against block colliders, resolve y then x then z
    fn move_and_collide(&mut self, world: &World) {
        let want   = self.vel;
        let mut bb = Aabb::player(self.pos);
        let boxes  = Self::nearby_block_boxes(world, &bb, want);

        let mut d = want;
        for axis in [1usize, 0, 2] {
            for b in &boxes {
                d[axis] = bb.clamp_axis(b, axis, d[axis]);
            }
            bb.shift(axis, d[axis]);
        }

        self.on_ground = want[1] < 0.0 && (d[1] - want[1]).abs() > EPS;
        for ax in 0..3 {
            if (d[ax] - want[ax]).abs() > EPS {
                self.vel[ax] = 0.0;
            }
            self.pos[ax] += d[ax];
        }
    }

    fn nearby_block_boxes(world: &World, bb: &Aabb, vel: [f64; 3]) -> Vec<Aabb> {
        let mut lo = [0i32; 3];
        let mut hi = [0i32; 3];
        for ax in 0..3 {
            let a = bb.min[ax].min(bb.min[ax] + vel[ax]) - 1.0;
            let b = bb.max[ax].max(bb.max[ax] + vel[ax]) + 1.0;
            lo[ax] = a.floor() as i32;
            hi[ax] = b.floor() as i32;
        }

        let mut out = Vec::new();
        for bx in lo[0]..=hi[0] {
            for by in lo[1]..=hi[1] {
                for bz in lo[2]..=hi[2] {
                    let Some(sid) = world.block_state_raw(bx, by, bz) else {
                        continue;
                    };
                    let state        = BlockState::from_id(sid);
                    let (ox, oy, oz) = (f64::from(bx), f64::from(by), f64::from(bz));
                    for shape in state.get_block_collision_shapes() {
                        out.push(Aabb {
                            min: [shape.min.x + ox, shape.min.y + oy, shape.min.z + oz],
                            max: [shape.max.x + ox, shape.max.y + oy, shape.max.z + oz],
                        });
                    }
                }
            }
        }

        out
    }
}
