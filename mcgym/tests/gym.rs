//! end to end gym assembly, spawn agents and RESET/STEP through the Gym trait
//! emitted observations must be valid schema records reflecting real terrain

use mcgym::gym::{GymState, decode_agent_obs};
use mcgym::schema::{ACTION_NBYTES, Action, OBS_NBYTES};
use mcgym::transport::Gym;

#[test]
fn gym_resets_steps_and_emits_valid_obs() {
    let n = 2;
    let mut gym     = GymState::new(n, 42, 32);
    let mut obs     = vec![0u8; n * OBS_NBYTES];
    let mut actions = vec![0u8; n * ACTION_NBYTES];

    // reset -> initial observations
    gym.reset(&mut obs);
    for i in 0..n {
        let o = decode_agent_obs(&obs, i);
        assert_eq!(o.schema_version, 1, "schema_version");
        assert_eq!(o.agent_id, i as i32, "agent_id");
        assert_eq!(o.tick, 0, "tick");
        assert!(o.pos[1] > 0.0 && o.pos[1] < 320.0, "sane y: {}", o.pos[1]);
        assert_eq!(o.health, 20.0, "health");
        let nonair = o.voxel_blocks.iter().filter(|&&v| v != 0).count();
        assert!(nonair > 100, "voxel grid should see terrain, got {nonair} non-air");
    }

    // step forward for a while (forest seek spawns agents facing a tree so they may collide
    // with the trunk, free walking is covered by the physics test, here the step loop must
    // keep producing valid observations)
    let enc = Action { forward: 1.0, ..Default::default() }.encode();
    for i in 0..n {
        actions[i * ACTION_NBYTES..(i + 1) * ACTION_NBYTES].copy_from_slice(&enc);
    }
    for _ in 0..25 {
        gym.step(&actions, &mut obs);
    }

    for i in 0..n {
        let o = decode_agent_obs(&obs, i);
        assert_eq!(o.tick, 25, "tick advanced");
        assert!(o.pos[1] > 0.0 && o.pos[1] < 320.0, "agent {i} y stayed sane: {}", o.pos[1]);
        let nonair = o.voxel_blocks.iter().filter(|&&v| v != 0).count();
        assert!(nonair > 100, "agent {i} obs still sees terrain after stepping");
    }
}
