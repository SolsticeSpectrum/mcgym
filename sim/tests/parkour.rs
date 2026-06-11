//! spiral map boots, agents stand in the start pool, water slows the sprint

use std::path::Path;

use sim::gym::Sim;
use sim::obs::center_index;
use sim::schema::{ACTION_NBYTES, Action, OBS_NBYTES, Obs};
use sim::transport::Gym;

#[test]
fn spiral_boot() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("worlds/spiral");
    let mut sim = Sim::fixed(2, &dir, [100.5, -60.0, -59.5], 90.0, false, 4);
    let mut obs = vec![0u8; 2 * OBS_NBYTES];
    sim.reset(&mut obs);

    let o = Obs::decode(&obs[..OBS_NBYTES]);
    let ground: usize = o.voxel_blocks.iter().filter(|&&b| b != 0).count();
    assert!(ground > 50, "course must be visible, got {ground} non air cells");

    // bounded cells must carry real boxes, slabs and stairs are not full cubes
    let partial = o
        .voxel_bounds
        .chunks(6)
        .filter(|b| b[4] > b[1] && (b[4] - b[1]) % 16 != 0)
        .count();
    assert!(partial > 0, "course has slabs and stairs, bounds must show them");

    // agents must stand on the hut floor, not inside anything
    assert!(o.pos[1] > -60.5 && o.pos[1] < -59.5, "feet on the hut floor, got y={}", o.pos[1]);
    let head = o.voxel_blocks[center_index() + sim::schema::VOXEL_EDGE * sim::schema::VOXEL_EDGE];
    assert_eq!(head, 0, "head cell must be air, agent embedded in {head}");

    // sprint west off the dock, through the pool, water must hold the agent back
    let start = o.pos;
    let act   = Action { forward: 1.0, sprint: 1, ..Default::default() };
    let mut actions = vec![0u8; 2 * ACTION_NBYTES];
    act.encode_into(&mut actions[..ACTION_NBYTES]);
    act.encode_into(&mut actions[ACTION_NBYTES..]);
    for _ in 0..40 {
        sim.step(&actions, &mut obs);
    }
    let end = Obs::decode(&obs[..OBS_NBYTES]).pos;
    let d   = ((end[0] - start[0]).powi(2) + (end[2] - start[2]).powi(2)).sqrt();
    assert!(d > 1.0, "agent must move, went {d}");
    assert!(d < 25.0, "sprint cap sanity, went {d} in 40 ticks");
}
