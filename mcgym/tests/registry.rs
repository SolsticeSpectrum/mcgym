//! The world->schema block-id bridge must agree with `schema/registry.json` for the blocks
//! the policy cares about, and Pumpkin's vanilla blocks should mostly map cleanly.

use mcgym::registry::Registry;
use pumpkin_data::Block;

#[test]
fn maps_known_blocks() {
    let r = Registry::load();
    // registry.json: air=0, stone=1, oak_log=49.
    assert_eq!(r.block(Block::AIR.default_state.id), 0, "air");
    assert_eq!(r.block(Block::STONE.default_state.id), 1, "stone");
    assert_eq!(r.block(Block::OAK_LOG.default_state.id), 49, "oak_log");
    assert_eq!(r.block(Block::SPRUCE_LOG.default_state.id), 50, "spruce_log");
}

#[test]
fn vanilla_coverage_is_high() {
    let r = Registry::load();
    // Both sides are vanilla 1.21.x; a handful of Pumpkin-only names may not map, but the
    // overwhelming majority should. This guards against a namespace/format regression.
    let unmapped = r.unmapped_block_count();
    assert!(
        unmapped < 80,
        "too many unmapped Pumpkin blocks ({unmapped}) — likely a name/format mismatch"
    );
}
