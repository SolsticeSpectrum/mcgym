//! Forest-seek spawning: agents should start near trees, so logs appear in their observations —
//! the precondition for ever learning to gather wood.

use std::collections::HashSet;

use mcai_gym::gym::{GymState, decode_agent_obs};
use mcai_gym::registry::Registry;
use mcai_gym::schema::OBS_NBYTES;
use mcai_gym::transport::Gym;

#[test]
fn forest_seek_puts_logs_in_observations() {
    let reg = Registry::load();
    let log_ids: HashSet<i32> = [
        "minecraft:oak_log",
        "minecraft:birch_log",
        "minecraft:spruce_log",
        "minecraft:jungle_log",
        "minecraft:acacia_log",
        "minecraft:dark_oak_log",
        "minecraft:mangrove_log",
        "minecraft:cherry_log",
        "minecraft:pale_oak_log",
    ]
    .iter()
    .map(|n| reg.block_id(n))
    .filter(|&v| v > 0)
    .collect();
    assert!(!log_ids.is_empty(), "registry should know log block ids");

    let n = 6;
    let mut gym = GymState::new(n, 42, 64);
    let mut obs = vec![0u8; n * OBS_NBYTES];
    gym.reset(&mut obs);

    let near_trees = (0..n)
        .filter(|&i| {
            let o = decode_agent_obs(&obs, i);
            o.voxel_blocks.iter().any(|v| log_ids.contains(v))
        })
        .count();

    // Forest-seek can't conjure trees in a desert, but across spread spawns most should land in
    // forests with logs in view.
    assert!(
        near_trees >= n / 2,
        "expected most agents near trees, only {near_trees}/{n} had logs in view"
    );
}
