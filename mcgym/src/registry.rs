//! Bridge from Pumpkin's block/item ids to the MCAI schema registry ints.
//!
//! The policy was trained against the integer ids in `schema/registry.json` (block name ->
//! int). Pumpkin gives us a block *state* id per voxel; `Block::from_state_id(id).name`
//! yields the namespaced-less identifier ("oak_log"), which we prefix with "minecraft:" and
//! look up. We precompute a dense `state_id -> mcai int` table so the obs hot path is a single
//! array index.
//!
//! `registry.json` is embedded at build time so the gym binary is self-contained.

use std::collections::HashMap;

use pumpkin_data::Block;

use crate::schema::HIDDEN_BLOCK_ID;

const REGISTRY_JSON: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../schema/registry.json"));

pub struct Registry {
    /// Indexed by block state id; value is the MCAI block int (0 = air).
    state_to_block: Vec<i32>,
    /// MCAI block int by namespaced name ("minecraft:oak_log" -> int).
    block_by_name: HashMap<String, i32>,
    /// MCAI item int by name ("minecraft:oak_log" -> int), for inventory mapping.
    item_by_name: HashMap<String, i32>,
    /// Pumpkin block names with no MCAI registry entry (diagnostics).
    unmapped: usize,
}

impl Registry {
    pub fn load() -> Self {
        let doc: serde_json::Value =
            serde_json::from_str(REGISTRY_JSON).expect("parse registry.json");

        let blocks = doc
            .get("blocks")
            .and_then(serde_json::Value::as_object)
            .expect("registry.json blocks");
        let block_by_name: HashMap<&str, i32> = blocks
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_i64().expect("block id") as i32))
            .collect();

        let item_by_name: HashMap<String, i32> = doc
            .get("items")
            .and_then(serde_json::Value::as_object)
            .map(|items| {
                items
                    .iter()
                    .map(|(k, v)| (k.clone(), v.as_i64().expect("item id") as i32))
                    .collect()
            })
            .unwrap_or_default();

        // from_state_id is total over u16 (returns AIR out of range), so this is safe.
        let mut state_to_block = vec![0i32; u16::MAX as usize + 1];
        let mut seen_unmapped: HashMap<&str, ()> = HashMap::new();
        for sid in 0..=u16::MAX {
            let name = Block::from_state_id(sid).name;
            let key = format!("minecraft:{name}");
            match block_by_name.get(key.as_str()) {
                Some(&id) => state_to_block[sid as usize] = id,
                None => {
                    seen_unmapped.insert(name, ());
                }
            }
        }

        let block_by_name = blocks
            .iter()
            .map(|(k, v)| (k.clone(), v.as_i64().expect("block id") as i32))
            .collect();

        Self {
            state_to_block,
            block_by_name,
            item_by_name,
            unmapped: seen_unmapped.len(),
        }
    }

    /// MCAI block int for a namespaced block name, or -1 if unknown.
    pub fn block_id(&self, namespaced: &str) -> i32 {
        self.block_by_name.get(namespaced).copied().unwrap_or(-1)
    }

    /// MCAI block int for a Pumpkin block state id (obs hot path).
    #[inline]
    pub fn block(&self, state_id: u16) -> i32 {
        self.state_to_block[state_id as usize]
    }

    /// MCAI item int for a namespaced item id, or `HIDDEN_BLOCK_ID` if unknown.
    pub fn item(&self, namespaced: &str) -> i32 {
        self.item_by_name
            .get(namespaced)
            .copied()
            .unwrap_or(HIDDEN_BLOCK_ID)
    }

    pub fn unmapped_block_count(&self) -> usize {
        self.unmapped
    }
}
