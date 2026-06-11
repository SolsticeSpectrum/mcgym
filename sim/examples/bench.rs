// 64 agents sprint roaming, prints world ticks per second and agent steps per second

use std::time::Instant;

use sim::gym::Sim;
use sim::schema::{ACTION_NBYTES, Action, OBS_NBYTES};
use sim::transport::Gym;

fn main() {
    let n     = 64usize;
    let ticks = 1000i64;

    let boot    = Instant::now();
    let mut sim = Sim::new(n, 3, 128, 3);
    let mut obs = vec![0u8; n * OBS_NBYTES];
    sim.reset(&mut obs);
    eprintln!("[bench] boot {:.1}s", boot.elapsed().as_secs_f64());

    let mut actions = vec![0u8; n * ACTION_NBYTES];
    let start = Instant::now();
    for t in 0..ticks {
        for i in 0..n {
            let act = Action {
                forward:   1.0,
                sprint:    1,
                jump:      u8::from(t % 7 == 0),
                yaw_delta: ((i as i64 * 13 + t) % 11 - 5) as f32,
                ..Default::default()
            };
            act.encode_into(&mut actions[i * ACTION_NBYTES..(i + 1) * ACTION_NBYTES]);
        }
        sim.step(&actions, &mut obs);
    }

    let dt = start.elapsed().as_secs_f64();
    eprintln!("[bench] {ticks} ticks x {n} agents in {dt:.1}s = {:.0} tps, {:.0} agent steps/s",
              ticks as f64 / dt, ticks as f64 * n as f64 / dt);
}
