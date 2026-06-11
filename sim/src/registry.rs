//! azalea state/item ids to schema registry ints, plus per state collision bounds luts

use std::collections::HashMap;

use azalea_block::BlockState;
use azalea_core::position::BlockPos;
use azalea_physics::collision::BlockWithShape;
use azalea_registry::builtin::{BlockKind, ItemKind};

use crate::schema::BOUNDS_DIMS;

const REGISTRY_JSON: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../schema/registry.json"));

pub struct Registry {
    state_to_block: Vec<i32>,            // by azalea block state id, 0 = air
    state_bounds:   Vec<[u8; BOUNDS_DIMS]>, // union collision aabb in 16ths
    item_to_int:    Vec<i32>,            // by azalea item kind id
    block_by_name:  HashMap<String, i32>,
}

impl Registry {
    pub fn load() -> Self {
        let doc: serde_json::Value =
            serde_json::from_str(REGISTRY_JSON).expect("parse registry.json");

        let blocks = doc["blocks"].as_object().expect("registry.json blocks");
        let by_name: HashMap<&str, i32> = blocks
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_i64().expect("block id") as i32))
            .collect();
        let items = doc["items"].as_object().expect("registry.json items");
        let item_by_name: HashMap<&str, i32> = items
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_i64().expect("item id") as i32))
            .collect();

        let states = BlockState::MAX_STATE as usize + 1;
        let mut state_to_block = vec![0i32; states];
        let mut state_bounds   = vec![[0u8; BOUNDS_DIMS]; states];
        for sid in 0..states {
            let Some(state) = BlockState::try_from(sid as u32).ok() else { continue };
            let kind = BlockKind::from(state);
            if let Some(&id) = by_name.get(kind.to_str()) {
                state_to_block[sid] = id;
            }
            state_bounds[sid] = bounds(state);
        }

        let mut item_to_int = Vec::new();
        while let Ok(kind) = ItemKind::try_from(item_to_int.len() as u32) {
            item_to_int.push(item_by_name.get(kind.to_str()).copied().unwrap_or(-1));
        }

        let block_by_name = blocks
            .iter()
            .map(|(k, v)| (k.clone(), v.as_i64().expect("block id") as i32))
            .collect();

        Self { state_to_block, state_bounds, item_to_int, block_by_name }
    }

    // obs hot paths
    #[inline]
    pub fn block(&self, state: BlockState) -> i32 {
        self.state_to_block[state.id() as usize]
    }

    #[inline]
    pub fn bounds(&self, state: BlockState) -> &[u8; BOUNDS_DIMS] {
        &self.state_bounds[state.id() as usize]
    }

    #[inline]
    pub fn item(&self, kind: ItemKind) -> i32 {
        self.item_to_int[kind as usize]
    }

    pub fn block_id(&self, name: &str) -> i32 {
        self.block_by_name.get(name).copied().unwrap_or(-1)
    }

    #[inline]
    pub fn solid(&self, state: BlockState) -> bool {
        let b = &self.state_bounds[state.id() as usize];
        b[3] > b[0]
    }
}

// union of the state collision aabbs quantized to 16ths, air stays all zero
fn bounds(state: BlockState) -> [u8; BOUNDS_DIMS] {
    let shape = state.collision_shape(BlockPos::new(0, 0, 0));
    let boxes = shape.to_aabbs();
    if boxes.is_empty() {
        return [0; BOUNDS_DIMS];
    }

    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for b in &boxes {
        lo[0] = lo[0].min(b.min.x); hi[0] = hi[0].max(b.max.x);
        lo[1] = lo[1].min(b.min.y); hi[1] = hi[1].max(b.max.y);
        lo[2] = lo[2].min(b.min.z); hi[2] = hi[2].max(b.max.z);
    }

    let mut at = [0u8; BOUNDS_DIMS];
    for ax in 0..3 {
        at[ax]     = (lo[ax] * 16.0).clamp(0.0, 255.0) as u8;
        at[ax + 3] = (hi[ax] * 16.0).clamp(0.0, 255.0) as u8;
    }
    at
}
