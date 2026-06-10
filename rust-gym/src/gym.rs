//! The gym: N agents in one generated world, driven by the transport's RESET/STEP loop.
//!
//! Assembles world + registry + physics + obs into the `Gym` trait the transport calls. Spawning
//! here is a simple spread-on-a-grid placeholder; the full WILD curriculum (forest-seek, persistent
//! home, relocation) is a later layer. Per the project's rule, RESET does not teleport agents.

use pumpkin_data::{Block, BlockState};

use crate::obs::{BLOCK_REACH, build_obs, raycast_target};
use crate::physics::Agent;
use crate::registry::Registry;
use crate::schema::{ACTION_NBYTES, Action, OBS_NBYTES, Obs};
use crate::transport::Gym;
use crate::world::World;

/// Chunks generated around each agent (covers the stride-4 far grid's ~32-block reach).
const CHUNK_RADIUS: i32 = 2;
/// Mining speed of the agent's axe on wood vs bare hand on everything else.
const AXE_SPEED: f64 = 6.0;
const HAND_SPEED: f64 = 1.0;

fn is_wood(name: &str) -> bool {
    name.ends_with("_log")
        || name.ends_with("_wood")
        || name.ends_with("_stem")
        || name.ends_with("_planks")
}

/// One tick of mining for an agent. With `attack` held and a block in reach, accumulate break
/// progress (vanilla: speed / hardness / 30 per tick); on completion break the block and give the
/// drop to the agent. Returns the dropped item id if a block broke this tick.
pub fn mine_step(agent: &mut Agent, world: &mut World, reg: &Registry, act: &Action) -> Option<i32> {
    if act.attack == 0 {
        agent.mine_target = None;
        agent.mine_progress = 0.0;
        return None;
    }
    let target = raycast_target(world, agent.pos, agent.yaw, agent.pitch, BLOCK_REACH);
    let Some((tb, _face, _dist)) = target else {
        agent.mine_target = None;
        agent.mine_progress = 0.0;
        return None;
    };
    let Some(sid) = world.block_state_raw(tb[0], tb[1], tb[2]) else {
        return None;
    };
    let block = Block::from_state_id(sid);
    let hardness = f64::from(block.hardness);
    if hardness < 0.0 {
        return None; // unbreakable (bedrock)
    }
    if agent.mine_target != Some(tb) {
        agent.mine_target = Some(tb);
        agent.mine_progress = 0.0;
    }
    let speed = if is_wood(block.name) { AXE_SPEED } else { HAND_SPEED };
    agent.mine_progress += if hardness <= 0.0 {
        1.0
    } else {
        speed / hardness / 30.0
    };
    if agent.mine_progress >= 1.0 {
        world.break_block(tb[0], tb[1], tb[2]);
        agent.mine_target = None;
        agent.mine_progress = 0.0;
        let item = reg.item(&format!("minecraft:{}", block.name));
        if item >= 0 {
            agent.add_item(item);
            return Some(item);
        }
    }
    None
}

/// An alive agent that makes no progress (no move >= 3, no mine, no pickup) for this many ticks
/// has a hopeless home and is relocated. Dying a lot is NOT stuck.
const STUCK_GIVEUP: i64 = 1500;
const MOVE_PROGRESS: f64 = 3.0;
const SPAWN_SEARCH_RADIUS: i32 = 5;

fn is_tree_block(name: &str) -> bool {
    name.ends_with("_log") || name.ends_with("_leaves") || name.ends_with("_wood") || name.ends_with("_stem")
}

fn has_collision(world: &World, x: i32, y: i32, z: i32) -> bool {
    world
        .block_state_raw(x, y, z)
        .is_some_and(|s| BlockState::from_id(s).get_block_collision_shapes().count() > 0)
}

/// Feet Y of a standable spot at (x,z): top *terrain* (non-tree) solid with 2 air above, else None.
fn standable_feet_y(world: &World, x: i32, z: i32) -> Option<i32> {
    for y in (world.bottom_y()..world.top_y()).rev() {
        if !has_collision(world, x, y, z) {
            continue;
        }
        let name = Block::from_state_id(world.block_state_raw(x, y, z).unwrap()).name;
        if is_tree_block(name) {
            continue; // skip canopy/trunk, keep descending to ground
        }
        let fy = y + 1;
        if !has_collision(world, x, fy, z) && !has_collision(world, x, fy + 1, z) {
            return Some(fy);
        }
        return None;
    }
    None
}

/// All log block positions within `reach` blocks of (ax,az) in the generated area.
fn collect_logs(world: &World, ax: i32, az: i32, reach: i32) -> Vec<[i32; 3]> {
    let mut logs = Vec::new();
    for x in (ax - reach)..=(ax + reach) {
        for z in (az - reach)..=(az + reach) {
            for y in world.bottom_y()..world.top_y() {
                if let Some(sid) = world.block_state_raw(x, y, z) {
                    if Block::from_state_id(sid).name.ends_with("_log") {
                        logs.push([x, y, z]);
                    }
                }
            }
        }
    }
    logs
}

fn yaw_facing(from: [f64; 2], to: [f64; 2]) -> f32 {
    (-(to[0] - from[0])).atan2(to[1] - from[1]).to_degrees() as f32
}

/// Find a spawn near trees: stand by the log nearest the anchor (facing it). Falls back to the
/// anchor's own surface when no trees are within the generated area (then far-nav reward applies).
fn find_forest_spawn(world: &World, ax: i32, az: i32, reach: i32) -> ([f64; 3], f32) {
    let logs = collect_logs(world, ax, az, reach);
    if let Some(log) = logs.iter().min_by_key(|p| {
        let (dx, dz) = (p[0] - ax, p[2] - az);
        dx * dx + dz * dz
    }) {
        let (lx, lz) = (log[0], log[2]);
        let mut best: Option<(i64, [f64; 3])> = None;
        for dx in -SPAWN_SEARCH_RADIUS..=SPAWN_SEARCH_RADIUS {
            for dz in -SPAWN_SEARCH_RADIUS..=SPAWN_SEARCH_RADIUS {
                if dx == 0 && dz == 0 {
                    continue; // not inside the trunk
                }
                let (sx, sz) = (lx + dx, lz + dz);
                if let Some(fy) = standable_feet_y(world, sx, sz) {
                    let d = i64::from(dx * dx + dz * dz);
                    let pos = [f64::from(sx) + 0.5, f64::from(fy), f64::from(sz) + 0.5];
                    if best.is_none_or(|(bd, _)| d < bd) {
                        best = Some((d, pos));
                    }
                }
            }
        }
        if let Some((_, pos)) = best {
            let yaw = yaw_facing([pos[0], pos[2]], [f64::from(lx) + 0.5, f64::from(lz) + 0.5]);
            return (pos, yaw);
        }
    }
    let fy = standable_feet_y(world, ax, az).unwrap_or(world.bottom_y() + 64);
    ([f64::from(ax) + 0.5, f64::from(fy), f64::from(az) + 0.5], 0.0)
}

pub struct GymState {
    world: World,
    reg: Registry,
    agents: Vec<Agent>,
    tick: i64,
    n: usize,
    spacing: i32,
    /// Last chunk each agent occupied, to avoid re-probing the neighbourhood every tick.
    last_chunk: Vec<(i32, i32)>,
}

impl GymState {
    /// Spawn `n_agents` spread on a grid `spacing` blocks apart, each on the surface with its
    /// local chunk neighbourhood generated (with trees).
    pub fn new(n_agents: usize, seed: i64, spacing: i32) -> Self {
        let reg = Registry::load();
        let mut world = World::new(seed);
        let mut agents = Vec::with_capacity(n_agents);

        let reach = CHUNK_RADIUS * 16 - 1;
        let side = (n_agents as f64).sqrt().ceil() as i32;
        for i in 0..n_agents as i32 {
            let ax = (i % side) * spacing;
            let az = (i / side) * spacing;
            Self::ensure_around_chunk(&mut world, ax >> 4, az >> 4);
            let (pos, yaw) = find_forest_spawn(&world, ax, az, reach);
            let mut agent = Agent::new(pos, yaw);
            // Start holding an axe (as the Java gym did), so wood mines at axe speed.
            let axe = reg.item("minecraft:iron_axe");
            if axe >= 0 {
                agent.inv_item_id[0] = axe;
                agent.inv_count[0] = 1;
            }
            agents.push(agent);
        }

        Self {
            world,
            reg,
            agents,
            tick: 0,
            n: n_agents,
            spacing,
            last_chunk: vec![(i32::MIN, i32::MIN); n_agents],
        }
    }

    /// Generate the CHUNK_RADIUS neighbourhood around a chunk (cached; cheap if already present).
    fn ensure_around_chunk(world: &mut World, ccx: i32, ccz: i32) {
        for dcx in -CHUNK_RADIUS..=CHUNK_RADIUS {
            for dcz in -CHUNK_RADIUS..=CHUNK_RADIUS {
                world.ensure_chunk(ccx + dcx, ccz + dcz);
            }
        }
    }

    /// Update stuck tracking and relocate/respawn per the curriculum rules. Progress = moved
    /// >= MOVE_PROGRESS from the reference, or broke a block this tick.
    fn update_and_relocate(&mut self, i: usize, broke: bool) {
        let (px, pz) = (self.agents[i].pos[0], self.agents[i].pos[2]);
        let dref = (px - self.agents[i].ref_xz[0]).powi(2) + (pz - self.agents[i].ref_xz[1]).powi(2);
        if dref >= MOVE_PROGRESS * MOVE_PROGRESS || broke {
            self.agents[i].ref_xz = [px, pz];
            self.agents[i].last_progress_tick = self.tick;
        }

        // Death is a wanted penalty: respawn at the persistent home to re-attempt the hazard.
        // (No damage source produces death yet; the structure is in place for when it does.)
        if self.agents[i].health <= 0.0 {
            let (home, hyaw) = (self.agents[i].home, self.agents[i].home_yaw);
            let a = &mut self.agents[i];
            a.pos = home;
            a.yaw = hyaw;
            a.vel = [0.0; 3];
            a.health = 20.0;
            a.food = 20.0;
            a.ref_xz = [home[0], home[2]];
            a.last_progress_tick = self.tick;
            a.mine_target = None;
            a.mine_progress = 0.0;
            return;
        }

        // Truly stuck (alive, no progress for STUCK_GIVEUP): the home is hopeless -> new forest home.
        if self.tick - self.agents[i].last_progress_tick >= STUCK_GIVEUP {
            let old = self.agents[i].home;
            let (nax, naz) = (old[0] as i32 + self.spacing, old[2] as i32);
            Self::ensure_around_chunk(&mut self.world, nax >> 4, naz >> 4);
            let (pos, yaw) = find_forest_spawn(&self.world, nax, naz, CHUNK_RADIUS * 16 - 1);
            let a = &mut self.agents[i];
            a.home = pos;
            a.home_yaw = yaw;
            a.pos = pos;
            a.yaw = yaw;
            a.vel = [0.0; 3];
            a.ref_xz = [pos[0], pos[2]];
            a.last_progress_tick = self.tick;
            a.mine_target = None;
            a.mine_progress = 0.0;
        }
    }

    fn write_all_obs(&self, obs: &mut [u8]) {
        for i in 0..self.n {
            let o = build_obs(&self.agents[i], &self.world, &self.reg, self.tick, i as i32);
            o.encode_into(&mut obs[i * OBS_NBYTES..(i + 1) * OBS_NBYTES]);
        }
    }

    pub fn agents(&self) -> &[Agent] {
        &self.agents
    }
}

impl Gym for GymState {
    fn reset(&mut self, obs: &mut [u8]) {
        self.tick = 0;
        self.write_all_obs(obs);
    }

    fn step(&mut self, actions: &[u8], obs: &mut [u8]) {
        self.tick += 1;
        for i in 0..self.n {
            let a = Action::decode(&actions[i * ACTION_NBYTES..(i + 1) * ACTION_NBYTES]);
            // Only re-probe/generate the neighbourhood when the agent crosses a chunk boundary
            // (it moves ~0.1 block/tick, so this is ~160x less work than every tick).
            let bp = self.agents[i].block_pos();
            let cc = (bp[0] >> 4, bp[2] >> 4);
            if cc != self.last_chunk[i] {
                Self::ensure_around_chunk(&mut self.world, cc.0, cc.1);
                self.last_chunk[i] = cc;
            }
            self.agents[i].step(&self.world, &a);
            // Disjoint fields: agent[i], world, reg borrowed separately.
            let broke = mine_step(&mut self.agents[i], &mut self.world, &self.reg, &a).is_some();
            self.update_and_relocate(i, broke);
        }
        // Periodically drop chunks no agent is near, bounding memory over long roaming runs.
        if self.tick % 256 == 0 {
            let centers: Vec<(i32, i32)> =
                self.agents.iter().map(|ag| {
                    let b = ag.block_pos();
                    (b[0] >> 4, b[2] >> 4)
                }).collect();
            self.world.retain_chunks_near(&centers, CHUNK_RADIUS + 2);
        }
        self.write_all_obs(obs);
    }
}

/// Decode one agent's observation from an obs region (helper for tests/inspection).
pub fn decode_agent_obs(obs: &[u8], i: usize) -> Obs {
    Obs::decode(&obs[i * OBS_NBYTES..(i + 1) * OBS_NBYTES])
}
