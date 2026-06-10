//! physics, agent dropped over real terrain falls under gravity and lands on the surface

use mcgym::physics::Agent;
use mcgym::schema::Action;
use mcgym::world::World;

/// highest solid block at (x,z)
fn surface_top(world: &World, x: i32, z: i32) -> i32 {
    use pumpkin_data::BlockState;
    for y in (world.bottom_y()..world.top_y()).rev() {
        if let Some(sid) = world.block_state_raw(x, y, z) {
            if BlockState::from_id(sid).get_block_collision_shapes().count() > 0 {
                return y;
            }
        }
    }
    world.bottom_y()
}

#[test]
fn agent_falls_and_lands_on_terrain() {
    let mut world = World::new(42);
    for cx in 0..2 {
        for cz in 0..2 {
            world.ensure_terrain_chunk(cx, cz);
        }
    }

    let (x, z)  = (8, 8);
    let top     = surface_top(&world, x, z);
    let stand_y = (top + 1) as f64; // feet rest on top of that block

    // drop from 6 blocks up, no input
    let mut agent = Agent::new([x as f64 + 0.5, stand_y + 6.0, z as f64 + 0.5], 0.0);
    let idle = Action::default();

    let mut landed_tick = None;
    for t in 0..120 {
        agent.step(&world, &idle);
        if agent.on_ground {
            landed_tick = Some(t);
            break;
        }
    }

    assert!(landed_tick.is_some(), "agent never landed");
    assert!(
        (agent.pos[1] - stand_y).abs() < 0.05,
        "landed at y={:.3}, expected ~{:.3} (top of surface block)",
        agent.pos[1],
        stand_y
    );

    // grounded and idle it should stay put, vanilla keeps a resting entity's vel.y at
    // -gravity*drag and collision cancels it each tick, so position is the invariant
    let resting = agent.pos;
    for _ in 0..20 {
        agent.step(&world, &idle);
    }
    assert!((agent.pos[1] - resting[1]).abs() < 1e-6, "agent sank/rose while resting");
}

#[test]
fn agent_walks_forward_on_flat_ground() {
    let mut world = World::new(7);
    for cx in -1..2 {
        for cz in -1..2 {
            world.ensure_terrain_chunk(cx, cz);
        }
    }
    let (x, z) = (4, 4);
    let top    = surface_top(&world, x, z);
    let mut agent = Agent::new([x as f64 + 0.5, (top + 1) as f64, z as f64 + 0.5], 0.0);

    // settle on the ground first
    let idle = Action::default();
    for _ in 0..10 {
        agent.step(&world, &idle);
    }
    let start = agent.pos;

    // hold forward for a while (yaw 0 => +z)
    let fwd = Action { forward: 1.0, ..Default::default() };
    for _ in 0..20 {
        agent.step(&world, &fwd);
    }
    let moved = (agent.pos[0] - start[0]).hypot(agent.pos[2] - start[2]);
    assert!(moved > 1.0, "agent barely moved forward ({moved:.3} blocks)");
}
