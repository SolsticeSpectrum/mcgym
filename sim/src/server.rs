//! steel side, headless server, bridge connection instead of sockets, manual lockstep ticking

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use steel_core::chunk::chunk_access::ChunkStatus;
use steel_core::chunk::chunk_map::GenerationTaskCap;
use steel_core::chunk::chunk_request::{ChunkRequestHandle, ChunkRequestState, ChunkTicketKind};
use steel_core::config::{RuntimeConfig, WorldsConfig};
use steel_core::entity::next_entity_id;
use steel_core::player::chunk_sender::ChunkSender;
use steel_core::player::connection::NetworkConnection;
use steel_core::player::{ClientInformation, GameProfile, Player, PlayerConnection, ResetReason};
use steel_core::entity::Entity;
use steel_core::server::Server;
use steel_registry::RegistryEntry;
use steel_core::world::World;
use steel_protocol::packet_traits::{CompressionInfo, EncodedPacket, ServerPacket};
use steel_protocol::packets::common::{SClientInformation, SCustomPayload, SKeepAlive};
use steel_protocol::packets::game::*;
use steel_protocol::utils::PacketError;
use steel_registry::packets::play;
use glam::DVec3;
use text_components::TextComponent;
use tokio::runtime::Runtime;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

// clientbound wire bytes pile up here until the gym pumps them to azalea
#[derive(Clone)]
pub struct Bridge {
    queue:  Arc<Mutex<Vec<EncodedPacket>>>,
    closed: Arc<AtomicBool>,
}

impl Bridge {
    fn new() -> Self {
        Self { queue: Arc::new(Mutex::new(Vec::new())), closed: Arc::new(AtomicBool::new(false)) }
    }

    pub fn drain(&self) -> Vec<EncodedPacket> {
        std::mem::take(&mut *self.queue.lock())
    }
}

impl NetworkConnection for Bridge {
    fn compression(&self) -> Option<CompressionInfo> { None }
    fn send_encoded(&self, packet: EncodedPacket) { self.queue.lock().push(packet); }
    fn send_encoded_bundle(&self, packets: Vec<EncodedPacket>) { self.queue.lock().extend(packets); }
    fn disconnect_with_reason(&self, reason: TextComponent) {
        panic!("steel disconnected a sim agent: {reason:?}");
    }
    fn tick(&self) {}
    fn latency(&self) -> i32 { 0 }
    fn close(&self) { self.closed.store(true, Ordering::SeqCst); }
    fn closed(&self) -> bool { self.closed.load(Ordering::SeqCst) }
}

pub struct Steel {
    pub server: Arc<Server>,
    pub world:  Arc<World>,
    pub rt:     Arc<Runtime>,
    tick:       u64,
}

// worlds.toml equivalent, ram storage, no disk anywhere
fn worlds_toml(generator: &str, seed: i64) -> String {
    // strict per generator config, empty needs the dimension type, overworld takes none
    let config = match generator {
        "steel:empty" => "\n[domains.minecraft.worlds.config]\ndimension_type = \"minecraft:overworld\"",
        _             => "",
    };
    format!(
        r#"
        save_path = "sim-void"
        seed = "{seed}"
        default_gamemode = "survival"
        difficulty = "normal"

        [storage]
        type = "steel:ram"

        [player_storage]
        type = "steel:file"

        [domains.minecraft]
        default = true

        [[domains.minecraft.worlds]]
        name = "overworld"
        generator = "{generator}"
        default = true
        {config}
        "#
    )
}

impl Steel {
    // generator is "minecraft:overworld" for worldgen or "steel:empty" for loaded maps
    pub fn boot(generator: &str, seed: i64, view: u8) -> Self {
        let rt = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .thread_stack_size(16 << 20) // worldgen density functions go deep unoptimized
                .enable_all()
                .build()
                .expect("tokio runtime"),
        );

        let config = RuntimeConfig {
            max_players:         1024,
            view_distance:       view,
            simulation_distance: 2,
            online_mode:         false,
            allow_flight:        false,
            encryption:          false,
            motd:                String::new(),
            use_favicon:         false,
            favicon:             String::new(),
            enforce_secure_chat: false,
            compression:         None,
            server_links:        None,
        };

        let worlds: WorldsConfig =
            toml::from_str(&worlds_toml(generator, seed)).expect("worlds config");

        let server = rt
            .block_on(Server::new(rt.clone(), CancellationToken::new(), config, worlds))
            .expect("steel server boot");
        let server = Arc::new(server);
        let world  = server.worlds.values().next().expect("overworld").clone();

        Self { server, world, rt, tick: 0 }
    }

    // one lockstep game tick, scheduling pump included (stock server runs it at the same rate)
    pub fn tick(&mut self) {
        self.tick += 1;
        let _guard = self.rt.enter();
        self.world.chunk_map.tick_scheduling(GenerationTaskCap::RespectMaxCap);
        self.world.tick_game(self.tick, true);
    }

    // ticket a chunk and pump scheduling until it is fully generated, flint pattern
    pub fn ensure(&self, pos: steel_utils::types::ChunkPos) -> ChunkRequestHandle {
        self.ensure_all(&[pos]).pop().expect("one handle")
    }

    // batch form, all tickets first so the generation pool works in parallel
    pub fn ensure_all(&self, all: &[steel_utils::types::ChunkPos]) -> Vec<ChunkRequestHandle> {
        let map = &self.world.chunk_map;
        let handles: Vec<_> = all
            .iter()
            .map(|&pos| map.request_chunk(pos, ChunkStatus::Full, ChunkTicketKind::Command))
            .collect();
        let _guard = self.rt.enter();
        for _ in 0..120_000 {
            map.tick_scheduling(GenerationTaskCap::RespectMaxCap);
            if handles.iter().all(|h| h.poll() == ChunkRequestState::Ready) {
                return handles;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        panic!("{} chunks never reached full", all.len());
    }

    // mirror of the private Server::tick_chunk_sending, public pieces only
    pub fn send_chunks(&self, players: &[Arc<Player>]) {
        let mut cache = FxHashMap::default();
        for player in players {
            let pos      = *player.last_chunk_pos.lock();
            let prepared = player.chunk_sender.lock().prepare_batch(&self.world, pos, &player.chunk_send_epoch);
            let Some(batch) = prepared else { continue };
            let encoded = ChunkSender::encode_batch(&batch, &mut cache, None);
            player.chunk_sender.lock().commit_batch(&batch, encoded, &player.connection, &player.chunk_send_epoch);
        }
    }

    // mirror of the private Server::finish_prepared_player_join, fresh player, no saved state
    pub fn join(&self, idx: usize, pos: DVec3, yaw: f32, view: u8) -> (Arc<Player>, Bridge) {
        let bridge  = Bridge::new();
        let profile = GameProfile {
            id:              Uuid::from_u128(0x53494D00 + idx as u128),
            name:            format!("agent{idx}"),
            properties:      vec![],
            profile_actions: None,
        };
        let info = ClientInformation { view_distance: view, ..ClientInformation::default() };

        let player = Arc::new_cyclic(|weak| {
            Player::new(
                profile,
                Arc::new(PlayerConnection::Other(Box::new(bridge.clone()))),
                self.world.clone(),
                Arc::downgrade(&self.server),
                self.server.config.clone(),
                next_entity_id(),
                weak,
                info,
            )
        });

        player.send_packet(CLogin {
            player_id:           player.id(),
            hardcore:            false,
            levels:              self.server.worlds.keys().cloned().collect(),
            max_players:         self.server.config.max_players as i32,
            chunk_radius:        player.view_distance().into(),
            simulation_distance: self.server.config.simulation_distance.into(),
            reduced_debug_info:  false,
            show_death_screen:   true,
            do_limited_crafting: false,
            common_player_spawn_info: CommonPlayerSpawnInfo {
                dimension_type:      self.world.dimension_type.id() as i32,
                dimension:           self.world.key.clone(),
                seed:                self.world.obfuscated_seed(),
                game_type:           player.game_mode(),
                previous_game_type:  Some(player.previous_game_mode()),
                is_debug:            false,
                is_flat:             self.world.is_flat,
                last_death_location: None,
                portal_cooldown:     0,
                sea_level:           self.world.sea_level,
            },
            enforces_secure_chat: false,
        });

        player.reset(self.world.clone(), ResetReason::InitialJoin);
        assert!(player.spawn(pos, (yaw, 0.0), ResetReason::InitialJoin), "spawn failed");

        (player, bridge)
    }

    // mirror of the private JavaConnection::process_packet dispatch, sans keep alive
    pub fn apply(&self, player: &Arc<Player>, id: i32, payload: &[u8]) -> Result<(), PacketError> {
        let data = &mut std::io::Cursor::new(payload);
        match id {
            play::S_ACCEPT_TELEPORTATION => player.handle_accept_teleportation(SAcceptTeleportation::read_packet(data)?),
            play::S_ATTACK               => player.handle_attack(SAttack::read_packet(data)?),
            play::S_INTERACT             => player.handle_interact(SInteract::read_packet(data)?),
            play::S_CUSTOM_PAYLOAD       => player.handle_custom_payload(SCustomPayload::read_packet(data)?),
            play::S_CLIENT_INFORMATION   => player.handle_client_information(SClientInformation::read_packet(data)?),
            play::S_CLIENT_TICK_END      => player.handle_client_tick_end(),
            play::S_CHUNK_BATCH_RECEIVED => {
                let packet = SChunkBatchReceived::read_packet(data)?;
                player.chunk_sender.lock().on_chunk_batch_received_by_client(packet.desired_chunks_per_tick);
            }
            play::S_KEEP_ALIVE             => { let _ = SKeepAlive::read_packet(data)?; }
            play::S_MOVE_PLAYER_POS        => player.handle_move_player(SMovePlayerPos::read_packet(data)?.into()),
            play::S_MOVE_PLAYER_POS_ROT    => player.handle_move_player(SMovePlayerPosRot::read_packet(data)?.into()),
            play::S_MOVE_PLAYER_ROT        => player.handle_move_player(SMovePlayerRot::read_packet(data)?.into()),
            play::S_MOVE_PLAYER_STATUS_ONLY => player.handle_move_player(SMovePlayerStatusOnly::read_packet(data)?.into()),
            play::S_PLAYER_LOADED          => {
                if player.mark_client_loaded_from_network() {
                    player.send_inventory_to_remote();
                }
            }
            play::S_CONTAINER_CLICK     => player.handle_container_click(SContainerClick::read_packet(data)?),
            play::S_CONTAINER_CLOSE     => player.handle_container_close(SContainerClose::read_packet(data)?),
            play::S_PLAYER_INPUT        => player.handle_player_input(SPlayerInput::read_packet(data)?),
            play::S_PLAYER_COMMAND      => player.handle_player_command(SPlayerCommand::read_packet(data)?),
            play::S_PLAYER_ABILITIES    => player.handle_player_abilities(SPlayerAbilities::read_packet(data)?),
            play::S_USE_ITEM_ON         => player.handle_use_item_on(SUseItemOn::read_packet(data)?),
            play::S_USE_ITEM            => player.handle_use_item(SUseItem::read_packet(data)?),
            play::S_SET_CARRIED_ITEM    => player.handle_set_carried_item(SSetCarriedItem::read_packet(data)?),
            play::S_SWING               => { let packet = SSwing::read_packet(data)?; player.swing(packet.hand, false); }
            play::S_PLAYER_ACTION       => player.handle_player_action(SPlayerAction::read_packet(data)?),
            play::S_CLIENT_COMMAND      => { let packet = SClientCommand::read_packet(data)?; player.handle_client_command(packet.action); }
            play::S_PONG                => {}
            other                       => panic!("sim agent sent unhandled packet id {other}"),
        }
        Ok(())
    }
}
