//! the full wood loop, aim at a forest log, hold attack, wood must land in the inventory

use sim::gym::Sim;
use sim::schema::{ACTION_NBYTES, Action, OBS_NBYTES, Obs, VOXEL_EDGE, VOXEL_RADIUS};
use sim::transport::Gym;

fn log_ids() -> Vec<i32> {
    let doc: serde_json::Value = serde_json::from_str(include_str!("../../schema/registry.json")).unwrap();
    doc["blocks"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, _)| k.ends_with("_log"))
        .map(|(_, v)| v.as_i64().unwrap() as i32)
        .collect()
}

// nearest log cell in the near grid, offsets relative to the agent block
fn nearest_log(o: &Obs, logs: &[i32]) -> Option<[i32; 3]> {
    let r = VOXEL_RADIUS as i32;
    let e = VOXEL_EDGE as i32;
    let mut best: Option<(i32, [i32; 3])> = None;
    for (at, id) in o.voxel_blocks.iter().enumerate() {
        if !logs.contains(id) {
            continue;
        }
        let at = at as i32;
        let dy = at / (e * e) - r;
        let dz = (at % (e * e)) / e - r;
        let dx = at % e - r;
        let d  = dx * dx + dy * dy + dz * dz;
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, [dx, dy, dz]));
        }
    }
    best.map(|(_, p)| p)
}

#[test]
fn mine_log() {
    let logs    = log_ids();
    let mut sim = Sim::new(1, 7, 64, 4);
    let mut obs = vec![0u8; OBS_NBYTES];
    sim.reset(&mut obs);

    let mut actions = vec![0u8; ACTION_NBYTES];
    for tick in 0..1200 {
        let o   = Obs::decode(&obs);
        let got: i32 = o
            .inv_item_id
            .iter()
            .zip(o.inv_count)
            .filter(|(id, _)| **id > 0 && logs_items(**id))
            .map(|(_, c)| i32::from(c))
            .sum();
        if got > 0 {
            return;
        }

        let target = nearest_log(&o, &logs).unwrap_or_else(|| panic!("no log near forest spawn at tick {tick}"));

        // aim at the cell center from the eye, vanilla yaw 0 => +z
        let feet = [o.pos[0], o.pos[1], o.pos[2]];
        let bx   = feet[0].floor() + target[0] as f32 + 0.5;
        let by   = feet[1].floor() + target[1] as f32 + 0.5;
        let bz   = feet[2].floor() + target[2] as f32 + 0.5;
        let dx   = bx - feet[0];
        let dy   = by - (feet[1] + 1.62);
        let dz   = bz - feet[2];
        let len  = (dx * dx + dy * dy + dz * dz).sqrt();
        let yaw   = (-dx).atan2(dz).to_degrees();
        let pitch = (-dy / len).asin().to_degrees();

        let act = Action {
            forward:     if len > 3.0 { 1.0 } else { 0.0 },
            yaw_delta:   yaw - o.yaw,
            pitch_delta: pitch - o.pitch,
            attack:      1,
            ..Default::default()
        };
        act.encode_into(&mut actions);
        sim.step(&actions, &mut obs);
    }

    let o = Obs::decode(&obs);
    panic!(
        "no wood after 1200 ticks, target_block={} in_range={} pos={:?}",
        o.target_block, o.target_in_range, o.pos
    );
}

// item ids for logs differ from block ids, anything nonzero picked up counts once
// the mined block id space confirms the break, the item space confirms the pickup
fn logs_items(id: i32) -> bool {
    let doc: serde_json::Value = serde_json::from_str(include_str!("../../schema/registry.json")).unwrap();
    doc["items"]
        .as_object()
        .unwrap()
        .iter()
        .any(|(k, v)| k.ends_with("_log") && v.as_i64().unwrap() as i32 == id)
}
