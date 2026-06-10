//! voxel extraction, centre cell equals the block the agent stands in and tree logs land in grid

use mcgym::obs::{center_index, fill_voxels};
use mcgym::registry::Registry;
use mcgym::schema::VOXEL_CELLS;
use mcgym::world::World;
use pumpkin_data::Block;

#[test]
fn voxel_center_matches_world_and_finds_logs() {
    let reg = Registry::load();
    let mut world = World::new(42);
    // generate a forested area (features stage -> trees)
    for cx in 0..4 {
        for cz in 0..4 {
            world.ensure_chunk(cx, cz);
        }
    }

    // find any log block in the generated area
    let mut log_pos = None;
    'scan: for x in 0..64 {
        for z in 0..64 {
            for y in world.bottom_y()..world.top_y() {
                if let Some(id) = world.block_state_raw(x, y, z) {
                    if Block::from_state_id(id).name.ends_with("_log") {
                        log_pos = Some((x, y, z, reg.block(id)));
                        break 'scan;
                    }
                }
            }
        }
    }
    let (lx, ly, lz, log_id) = log_pos.expect("a forest area should contain at least one log");
    assert!(log_id > 0, "log should map to a real registry id");

    // centre the near grid on that log, the centre cell must be exactly that block
    let mut grid = vec![0i32; VOXEL_CELLS];
    fill_voxels(&world, &reg, lx, ly, lz, 1, &mut grid);
    assert_eq!(
        grid[center_index()],
        log_id,
        "voxel centre cell must equal the block at the agent's position"
    );

    // grid around a tree should contain multiple log cells
    let logs_in_grid = grid.iter().filter(|&&v| v == log_id).count();
    assert!(logs_in_grid >= 2, "expected the trunk to span multiple voxel cells, got {logs_in_grid}");
}
