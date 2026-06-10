//! The cross-language byte contract: the Rust codec must reproduce the canonical
//! golden fixtures byte-for-byte, and round-trip them losslessly. These fixtures
//! are the same bytes the Java gym, the Python trainer, and the Fabric mod agree on.

use std::path::PathBuf;

use mcai_gym::schema::{Action, INVENTORY_SLOTS, MAX_ENTITIES, Obs, VOXEL_CELLS};

fn fixture(name: &str) -> Vec<u8> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("schema/fixtures")
        .join(name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// Mirror of `trainer/mcai_train/schema/golden.py::golden_obs_record`.
fn golden_obs() -> Obs {
    let mut o = Obs {
        tick: 7,
        agent_id: 1,
        pos: [10.5, 80.0, -3.25],
        vel: [0.1, -0.05, 0.2],
        yaw: 45.0,
        pitch: -10.0,
        on_ground: 1,
        health: 20.0,
        food: 18.0,
        selected_slot: 4,
        voxel_blocks: (0..VOXEL_CELLS as i32).map(|i| i % 97).collect(),
        voxel_far: (0..VOXEL_CELLS as i32).map(|i| i % 89).collect(),
        target_block: 42,
        target_face: 2,
        target_distance: 2.75,
        target_in_range: 1,
        ..Default::default()
    };
    for i in 0..MAX_ENTITIES {
        o.entity_type_id[i] = i as i32;
        o.entity_health[i] = 20.0;
    }
    for i in 0..INVENTORY_SLOTS {
        o.inv_item_id[i] = i as i32;
        o.inv_count[i] = 1;
    }
    o
}

/// Mirror of `golden.py::golden_action_record`.
fn golden_action() -> Action {
    Action {
        forward: 1.0,
        strafe: 0.0,
        jump: 0,
        sneak: 0,
        sprint: 1,
        yaw_delta: 3.0,
        pitch_delta: -1.0,
        attack: 1,
        use_: 0,
        selected_slot: 4,
        inv_op_type: 0,
        inv_slot_a: 0,
        inv_slot_b: 0,
    }
}

#[test]
fn obs_matches_golden_bytes() {
    let want = fixture("golden_obs.bin");
    let got = golden_obs().encode();
    assert_eq!(got.len(), want.len(), "obs byte length");
    assert_eq!(got, want, "obs bytes diverge from canonical golden fixture");
}

#[test]
fn action_matches_golden_bytes() {
    let want = fixture("golden_action.bin");
    let got = golden_action().encode();
    assert_eq!(got.len(), want.len(), "action byte length");
    assert_eq!(got, want, "action bytes diverge from canonical golden fixture");
}

#[test]
fn obs_round_trips() {
    let bytes = fixture("golden_obs.bin");
    let decoded = Obs::decode(&bytes);
    assert_eq!(decoded.encode(), bytes, "obs decode->encode not identity");
}

#[test]
fn action_round_trips() {
    let bytes = fixture("golden_action.bin");
    let decoded = Action::decode(&bytes);
    assert_eq!(decoded.encode(), bytes, "action decode->encode not identity");
}
