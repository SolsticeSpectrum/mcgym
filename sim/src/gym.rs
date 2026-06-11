//! lockstep orchestration, trainer actions -> azalea inputs -> packets -> steel tick -> obs

use std::sync::Arc;

use azalea_client::mining::LeftClickMine;
use azalea_entity::{Jumping, LookDirection};
use azalea_physics::client_movement::{ClientMovementState, WalkDirection};
use steel_core::inventory::container::Container;
use steel_core::player::Player;
use steel_registry::RegistryExt;
use steel_protocol::packet_traits::EncodedPacket;
use glam::DVec3;

use crate::client::Swarm;
use crate::obs::build_obs;
use crate::registry::Registry;
use crate::schema::{ACTION_NBYTES, Action, OBS_NBYTES};
use crate::server::{Bridge, Steel};
use crate::transport::Gym;

// alive agent with no progress (no move >= 3, no wood) for this many ticks gets a new home
const STUCK_GIVEUP:  i64 = 1500;
const MOVE_PROGRESS: f64 = 3.0;

// settle pump bound, boot fails hard if the world is not streamed in by then
const SETTLE_MAX: usize = 4096;

pub struct Sim {
    steel:    Steel,
    tickets:  Vec<steel_core::chunk::chunk_request::ChunkRequestHandle>,
    swarm:    Swarm,
    players:  Vec<Arc<Player>>,
    bridges:  Vec<Bridge>,
    reg:      Registry,
    tick:     i64,
    n:        usize,
    mining:   bool,
    roam:     bool,
    spacing:  i32,
    homes:    Vec<(DVec3, f32)>,
    ref_xz:   Vec<[f64; 2]>,
    progress: Vec<i64>,
    wood:     Vec<i32>,
}

fn dir8(forward: f32, strafe: f32) -> WalkDirection {
    let f = forward > 0.5;
    let b = forward < -0.5;
    let l = strafe < -0.5;
    let r = strafe > 0.5;
    match (f, b, l, r) {
        (true,  _,     true,  _)     => WalkDirection::ForwardLeft,
        (true,  _,     _,     true)  => WalkDirection::ForwardRight,
        (true,  _,     _,     _)     => WalkDirection::Forward,
        (_,     true,  true,  _)     => WalkDirection::BackwardLeft,
        (_,     true,  _,     true)  => WalkDirection::BackwardRight,
        (_,     true,  _,     _)     => WalkDirection::Backward,
        (_,     _,     true,  _)     => WalkDirection::Left,
        (_,     _,     _,     true)  => WalkDirection::Right,
        _                            => WalkDirection::None,
    }
}

impl Sim {
    // worldgen mode, forest homes on a spacing grid
    pub fn new(n: usize, seed: i64, spacing: i32, view: u8) -> Self {
        let steel = Steel::boot("minecraft:overworld", seed, view);
        let mut sim = Self::assemble(steel, n, view, true, true, spacing);

        // provisional skydrop joins pull chunk tickets at each home column
        let side = (n as f64).sqrt().ceil() as i32;
        for i in 0..n {
            let ax = (i as i32 % side) * spacing;
            let az = (i as i32 / side) * spacing;
            sim.join(i, DVec3::new(f64::from(ax) + 0.5, 320.0, f64::from(az) + 0.5), 0.0, view);
        }
        sim.settle();

        // ground everyone at a forest spawn now that terrain exists
        for i in 0..n {
            let (pos, yaw) = sim.forest_spawn(i);
            sim.homes[i]   = (pos, yaw);
            sim.teleport(i, pos, yaw);
            sim.give_axe(i);
        }
        sim.settle();
        sim
    }

    // loaded map mode, blocks first so chunk streaming carries them, every agent on one spawn
    pub fn fixed(n: usize, dir: &std::path::Path, spawn: [f64; 3], yaw: f32, mining: bool, view: u8) -> Self {
        let steel   = Steel::boot("steel:empty", 0, view);
        let tickets = crate::anvil::load(&steel, dir);
        let mut sim = Self::assemble(steel, n, view, mining, false, 0);
        sim.tickets = tickets;

        let pos = DVec3::new(spawn[0], spawn[1], spawn[2]);
        for i in 0..n {
            sim.join(i, pos, yaw, view);
        }
        sim.settle();
        sim
    }

    fn assemble(steel: Steel, n: usize, _view: u8, mining: bool, roam: bool, spacing: i32) -> Self {
        let swarm = Swarm::new(n);
        let mut sim = Self {
            steel,
            tickets:  Vec::new(),
            swarm,
            players:  Vec::with_capacity(n),
            bridges:  Vec::with_capacity(n),
            reg:      Registry::load(),
            tick:     0,
            n,
            mining,
            roam,
            spacing,
            homes:    vec![(DVec3::new(0.0, 0.0, 0.0), 0.0); n],
            ref_xz:   vec![[0.0; 2]; n],
            progress: vec![0; n],
            wood:     vec![0; n],
        };
        sim.configure();
        sim
    }

    // config phase, steel registry wire data straight into each azalea client
    fn configure(&mut self) {
        let registry = self.steel.server.registry_cache.registry_packets.clone();
        let tags     = self.steel.server.registry_cache.tags_packet.clone();
        let finish   = EncodedPacket::from_bare(
            steel_protocol::packets::config::CFinishConfiguration {},
            None,
            steel_protocol::utils::ConnectionProtocol::Config,
        )
        .expect("encode finish configuration");

        for i in 0..self.n {
            let mut frames: Vec<Arc<_>> = registry.iter().map(|p| p.encoded_data.clone()).collect();
            frames.push(tags.encoded_data.clone());
            frames.push(finish.encoded_data.clone());
            self.swarm.inject(i, &frames);
        }
        self.swarm.update();
        self.swarm.update();
    }

    fn join(&mut self, i: usize, pos: DVec3, yaw: f32, view: u8) {
        let (player, bridge) = self.steel.join(i, pos, yaw, view);
        self.players.push(player);
        self.bridges.push(bridge);
        self.homes[i]  = (pos, yaw);
        self.ref_xz[i] = [pos.x, pos.z];
    }

    // pump packets without client physics until every client stands in a loaded chunk
    fn settle(&mut self) {
        let mut ready = 0;
        for _ in 0..SETTLE_MAX {
            self.steel.tick();
            self.steel.send_chunks(&self.players);

            for i in 0..self.n {
                let frames: Vec<Arc<_>> =
                    self.bridges[i].drain().into_iter().map(|p| p.encoded_data).collect();
                if !frames.is_empty() {
                    self.swarm.inject(i, &frames);
                }
            }
            self.swarm.update();
            for i in 0..self.n {
                for (id, payload) in self.swarm.outbox(i) {
                    self.steel.apply(&self.players[i], id, &payload).expect("apply packet");
                }
            }

            let landed = (0..self.n).all(|i| {
                let Some(pos) = self.swarm.try_get::<azalea_entity::Position>(i) else {
                    return false;
                };
                let Some(holder) = self.swarm.try_get::<azalea_client::local_player::WorldHolder>(i) else {
                    return false;
                };
                let chunk = azalea_core::position::ChunkPos::from(
                    &azalea_core::position::BlockPos::new(pos.x as i32, pos.y as i32, pos.z as i32));
                holder.shared.read().chunks.get(&chunk).is_some()
            });

            // a grace lap so inventory and entity sync land too
            ready = if landed { ready + 1 } else { 0 };
            if ready >= 8 {
                return;
            }
        }
        panic!("settle starved, clients never landed in loaded chunks");
    }

    fn teleport(&mut self, i: usize, pos: DVec3, yaw: f32) {
        self.players[i]
            .teleport(pos.x, pos.y, pos.z, yaw, 0.0)
            .expect("teleport");
    }

    // iron axe in hotbar slot 0 so wood mines at tool speed, synced to the client
    fn give_axe(&mut self, i: usize) {
        let axe = steel_registry::REGISTRY
            .items
            .by_key(&steel_utils::Identifier::vanilla("iron_axe".into()))
            .expect("iron_axe in steel registry");
        let stack = steel_registry::item_stack::ItemStack::with_count(axe, 1);
        self.players[i].inventory.lock().set_item(0, stack);
        self.players[i].send_inventory_to_remote();
    }

    // stand next to the log nearest home facing it, plain surface when no trees nearby
    fn forest_spawn(&mut self, i: usize) -> (DVec3, f32) {
        let (home, _) = self.homes[i];
        let (ax, az)  = (home.x as i32, home.z as i32);
        let mut area  = Vec::with_capacity(25);
        for dcx in -2..=2 {
            for dcz in -2..=2 {
                area.push(steel_utils::types::ChunkPos::new((ax >> 4) + dcx, (az >> 4) + dcz));
            }
        }
        // player tickets take over once the agent stands here, handles can drop
        let _ = self.steel.ensure_all(&area);
        crate::spawn::forest(&self.steel, ax, az)
    }

    fn drive(&mut self, i: usize, act: &Action) {
        let look: LookDirection = self.swarm.get(i);
        let yaw   = look.y_rot() + act.yaw_delta;
        let pitch = (look.x_rot() + act.pitch_delta).clamp(-90.0, 90.0);
        self.swarm.set::<LookDirection>(i, |l| *l = LookDirection::new(yaw, pitch));

        let dir    = dir8(act.forward, act.strafe);
        let sprint = act.sprint != 0 && matches!(dir, WalkDirection::Forward | WalkDirection::ForwardLeft | WalkDirection::ForwardRight);
        self.swarm.set::<ClientMovementState>(i, |m| {
            m.move_direction   = dir;
            m.trying_to_sprint = sprint;
            m.trying_to_crouch = act.sneak != 0;
        });
        self.swarm.set::<Jumping>(i, |j| *j = Jumping(act.jump != 0));
        self.swarm.mark::<LeftClickMine>(i, self.mining && act.attack != 0, || LeftClickMine);
    }

    // progress = moved >= MOVE_PROGRESS or gained wood, hopeless homes get replaced
    fn relocate(&mut self, i: usize) {
        let pos: azalea_entity::Position = self.swarm.get(i);
        let dref = (pos.x - self.ref_xz[i][0]).powi(2) + (pos.z - self.ref_xz[i][1]).powi(2);
        let wood = crate::spawn::wood_count(&self.swarm, &self.reg, i);
        if dref >= MOVE_PROGRESS * MOVE_PROGRESS || wood > self.wood[i] {
            self.ref_xz[i]   = [pos.x, pos.z];
            self.progress[i] = self.tick;
            self.wood[i]     = wood;
        }

        let health: azalea_entity::metadata::Health = self.swarm.get(i);
        if *health <= 0.0 {
            let (home, yaw) = self.homes[i];
            self.players[i].handle_client_command(steel_protocol::packets::game::ClientCommandAction::PerformRespawn);
            self.teleport(i, home, yaw);
            self.ref_xz[i]   = [home.x, home.z];
            self.progress[i] = self.tick;
            return;
        }

        if self.roam && self.tick - self.progress[i] >= STUCK_GIVEUP {
            let old = self.homes[i].0;
            self.homes[i].0 = DVec3::new(old.x + f64::from(self.spacing), old.y, old.z);
            let (pos, yaw)  = self.forest_spawn(i);
            self.homes[i]   = (pos, yaw);
            self.teleport(i, pos, yaw);
            self.ref_xz[i]   = [pos.x, pos.z];
            self.progress[i] = self.tick;
        }
    }

    // read only over the shared client world, agents fan out across cores
    fn write_all_obs(&self, obs: &mut [u8]) {
        let ecs   = self.swarm.app.world();
        let tick  = self.tick;
        let reg   = &self.reg;
        let ids   = &self.swarm.ids;
        let slots: Vec<&mut [u8]> = obs.chunks_mut(OBS_NBYTES).take(self.n).collect();
        std::thread::scope(|scope| {
            for (i, slot) in slots.into_iter().enumerate() {
                let id = ids[i];
                scope.spawn(move || {
                    build_obs(ecs, id, reg, i, tick).encode_into(slot);
                });
            }
        });
    }
}

impl Gym for Sim {
    fn reset(&mut self, obs: &mut [u8]) {
        // reset never teleports agents
        self.tick = 0;
        self.write_all_obs(obs);
    }

    fn step(&mut self, actions: &[u8], obs: &mut [u8]) {
        self.tick += 1;

        for i in 0..self.n {
            let act = Action::decode(&actions[i * ACTION_NBYTES..(i + 1) * ACTION_NBYTES]);
            self.drive(i, &act);
        }

        // client tick authors movement, server tick validates and reacts, replies flow back
        self.swarm.game_tick();
        for i in 0..self.n {
            for (id, payload) in self.swarm.outbox(i) {
                self.steel.apply(&self.players[i], id, &payload).expect("apply packet");
            }
        }

        self.steel.tick();
        self.steel.send_chunks(&self.players);

        for i in 0..self.n {
            let frames: Vec<Arc<_>> =
                self.bridges[i].drain().into_iter().map(|p| p.encoded_data).collect();
            if !frames.is_empty() {
                self.swarm.inject(i, &frames);
            }
        }
        self.swarm.update();

        for i in 0..self.n {
            self.relocate(i);
        }

        self.write_all_obs(obs);
    }
}
