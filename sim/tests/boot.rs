// boots the pair, walks an agent forward, the whole loop must hold together

use sim::gym::Sim;
use sim::schema::{ACTION_NBYTES, Action, OBS_NBYTES, Obs};
use sim::transport::Gym;

#[test]
fn boot_walk() {
    let mut sim = Sim::new(2, 7, 64, 4);
    let mut obs = vec![0u8; 2 * OBS_NBYTES];
    sim.reset(&mut obs);

    let o = Obs::decode(&obs[..OBS_NBYTES]);
    assert_eq!(o.schema_version, 2);
    assert!(o.pos[1] > 0.0, "agent must stand on terrain, got y={}", o.pos[1]);
    let ground: usize = o.voxel_blocks.iter().filter(|&&b| b != 0).count();
    assert!(ground > 50, "near grid must see terrain, got {ground} non air cells");

    let mut act = Action { forward: 1.0, sprint: 1, ..Default::default() };
    let mut actions = vec![0u8; 2 * ACTION_NBYTES];
    let start = Obs::decode(&obs[..OBS_NBYTES]).pos;
    for _ in 0..100 {
        act.jump = 1;
        act.encode_into(&mut actions[..ACTION_NBYTES]);
        act.encode_into(&mut actions[ACTION_NBYTES..]);
        sim.step(&actions, &mut obs);
    }
    let end = Obs::decode(&obs[..OBS_NBYTES]).pos;
    let d   = ((end[0] - start[0]).powi(2) + (end[2] - start[2]).powi(2)).sqrt();
    assert!(d > 5.0, "agent must move under sprint, went {d} blocks");
}
