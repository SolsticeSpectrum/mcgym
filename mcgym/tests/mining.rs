//! mining, agent aimed at a log holding attack breaks it in vanilla-ish time and gains the log

use mcgym::gym::mine_step;
use mcgym::obs::raycast_target;
use mcgym::physics::Agent;
use mcgym::registry::Registry;
use mcgym::schema::Action;
use mcgym::world::World;
use pumpkin_data::BlockState;

fn is_air_like(world: &World, x: i32, y: i32, z: i32) -> bool {
    world
        .block_state_raw(x, y, z)
        .is_none_or(|s| BlockState::from_id(s).get_block_collision_shapes().count() == 0)
}

#[test]
fn agent_mines_a_log_into_inventory() {
    let reg = Registry::load();
    let mut world = World::new(42);
    for cx in 0..4 {
        for cz in 0..4 {
            world.ensure_chunk(cx, cz);
        }
    }

    // find a log with a clear air approach from its -z side so we can aim at it
    let mut found = None;
    'scan: for x in 0..64 {
        for z in 2..64 {
            for y in world.bottom_y()..world.top_y() {
                if let Some(sid) = world.block_state_raw(x, y, z) {
                    if pumpkin_data::Block::from_state_id(sid).name.ends_with("_log")
                        && is_air_like(&world, x, y, z - 1)
                    {
                        found = Some((x, y, z));
                        break 'scan;
                    }
                }
            }
        }
    }
    let (lx, ly, lz) = found.expect("a log with a clear -z approach");

    // stand just south of the log (yaw 0 => looking +z), eye at the log's mid height
    let mut agent = Agent::new(
        [f64::from(lx) + 0.5, f64::from(ly) - 1.12, f64::from(lz) - 1.5],
        0.0,
    );
    
    let aim = raycast_target(&world, agent.pos, agent.yaw, agent.pitch, 4.5);
    assert_eq!(
        aim.map(|(b, _, _)| b),
        Some([lx, ly, lz]),
        "agent should be aimed at the log, got {aim:?}"
    );

    // hold attack, axe (speed 6) on a log (hardness 2) is ~0.1/tick -> ~10 ticks
    let attack    = Action { attack: 1, ..Default::default() };
    let mut broke = None;
    for t in 0..30 {
        if let Some(item) = mine_step(&mut agent, &mut world, &reg, &attack) {
            broke = Some((t, item));
            break;
        }
    }

    let (ticks, item) = broke.expect("log should break within 30 ticks");
    assert!((5..=20).contains(&ticks), "break took {ticks} ticks (expected ~10)");
    assert!(item >= 0, "log should map to a real item id");
    assert!(agent.count_item(item) >= 1, "log should be in inventory");
    assert_eq!(reg.block(world.block_state_raw(lx, ly, lz).unwrap()), 0, "broken block is now air");
}
