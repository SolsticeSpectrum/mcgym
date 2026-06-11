//! azalea side, one bevy app, networkless clients, manual schedules, raw byte io

use std::sync::Arc;

use azalea_auth::game_profile::GameProfile;
use azalea_client::{InConfigState, LocalPlayerBundle};
use azalea_client::connection::RawConnection;
use azalea_client::local_player::WorldHolder;
use azalea_client::packet::game::SendGamePacketEvent;
use azalea_client::player::GameProfileComponent;
use azalea_core::tick::GameTick;
use azalea_protocol::common::client_information::ClientInformation;
use azalea_protocol::packets::ConnectionProtocol;
use bevy_app::App;
use bevy_ecs::entity::Entity;
use bevy_ecs::observer::On;
use bevy_ecs::schedule::ExecutorKind;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use uuid::Uuid;

// serverbound frames per client entity, (packet id, payload)
type Outbox = Arc<Mutex<FxHashMap<Entity, Vec<(i32, Vec<u8>)>>>>;

pub struct Swarm {
    pub app: App,
    pub ids: Vec<Entity>,
    out:     Outbox,
    _rt:     tokio::runtime::Runtime,
}

// minecraft varint, returns (value, bytes consumed)
fn varint(buf: &[u8]) -> (i32, usize) {
    let mut value = 0i32;
    let mut at    = 0usize;
    loop {
        let byte = buf[at];
        value |= i32::from(byte & 0x7f) << (7 * at);
        at += 1;
        if byte & 0x80 == 0 {
            return (value, at);
        }
    }
}

impl Swarm {
    pub fn new(n: usize) -> Self {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");

        let mut app = App::new();
        let plugins = bevy_app::PluginGroup::build(azalea_client::DefaultPlugins)
            .disable::<bevy_log::LogPlugin>();
        app.add_plugins(plugins);
        app.edit_schedule(bevy_app::Main, |schedule| {
            schedule.set_executor_kind(ExecutorKind::SingleThreaded);
        });

        let out: Outbox = Arc::default();
        let sink        = out.clone();
        app.add_observer(move |ev: On<SendGamePacketEvent>| {
            let bytes     = azalea_protocol::write::serialize_packet(&ev.packet).expect("serialize");
            let (id, len) = varint(&bytes);
            sink.lock().entry(ev.sent_by).or_default().push((id, bytes[len..].to_vec()));
        });

        // clients start in the config state like a real join, login phase skipped on both sides
        let guard   = rt.enter();
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            let mut entity = app.world_mut().spawn_empty();
            let connection = RawConnection::new_networkless(ConnectionProtocol::Configuration);
            let holder     = WorldHolder::new(entity.id(), Arc::default());
            entity.insert((
                LocalPlayerBundle {
                    raw_connection: connection,
                    world_holder:   holder,
                    metadata:       azalea_entity::metadata::PlayerMetadataBundle::default(),
                },
                ClientInformation::default(),
                InConfigState,
                GameProfileComponent(GameProfile::new(
                    Uuid::from_u128(0x53494D00 + i as u128),
                    format!("agent{i}"),
                )),
            ));
            ids.push(entity.id());
        }
        drop(guard);

        let mut swarm = Self { app, ids, out, _rt: rt };
        swarm.update();
        swarm
    }

    // steel wire frames are length prefixed, azalea wants bare id + payload
    pub fn inject(&mut self, i: usize, frames: &[Arc<impl std::ops::Deref<Target = [u8]>>]) {
        let id = self.ids[i];
        let mut conn = self.app.world_mut().entity_mut(id);
        let mut conn = conn.get_mut::<RawConnection>().expect("raw connection");
        for frame in frames {
            let bytes: &[u8] = frame;
            let (len, at)  = varint(bytes);
            assert_eq!(at + len as usize, bytes.len(), "frame length mismatch");
            conn.injected_clientbound_packets.push(bytes[at..].into());
        }
    }

    // serverbound frames emitted since the last call
    pub fn outbox(&mut self, i: usize) -> Vec<(i32, Vec<u8>)> {
        self.out.lock().remove(&self.ids[i]).unwrap_or_default()
    }

    // one render frame, drains injected packets and applies them
    pub fn update(&mut self) {
        self.app.update();
    }

    // one game tick, physics and outgoing movement packets
    pub fn game_tick(&mut self) {
        self.app.update();
        self.app.world_mut().run_schedule(GameTick);
    }

    pub fn get<T: bevy_ecs::component::Component + Clone>(&self, i: usize) -> T {
        self.app.world().get::<T>(self.ids[i]).expect("component").clone()
    }

    pub fn try_get<T: bevy_ecs::component::Component + Clone>(&self, i: usize) -> Option<T> {
        self.app.world().get::<T>(self.ids[i]).cloned()
    }

    pub fn set<T: bevy_ecs::component::Component<Mutability = bevy_ecs::component::Mutable>>(
        &mut self,
        i: usize,
        f: impl FnOnce(&mut T),
    ) {
        f(&mut self.app.world_mut().entity_mut(self.ids[i]).get_mut::<T>().expect("component"));
    }

    pub fn mark<T: bevy_ecs::component::Component>(&mut self, i: usize, on: bool, make: impl FnOnce() -> T) {
        let mut entity = self.app.world_mut().entity_mut(self.ids[i]);
        let has = entity.contains::<T>();
        if on && !has {
            entity.insert(make());
        } else if !on && has {
            entity.remove::<T>();
        }
    }
}
