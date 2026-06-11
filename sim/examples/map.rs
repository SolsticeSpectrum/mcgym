// top down ascii of the spiral bottom, find the hut, the pool and valid spawns

use sim::server::Steel;
use steel_registry::{REGISTRY, RegistryExt};
use steel_utils::BlockPos;

fn main() {
    let steel = Steel::boot("steel:empty", 0, 3);
    let dir   = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("worlds/spiral");
    let _hold = sim::anvil::load(&steel, &dir);

    let name = |x: i32, y: i32, z: i32| -> &'static str {
        let sid = steel.world.get_block_state(BlockPos::new(x, y, z));
        REGISTRY.blocks.by_state_id(sid).map_or("?", |b| b.key.path.as_ref())
    };

    // what is at the bad spawn
    for y in [-63, -62, -61, -60, -59] {
        println!("col(38,{y},-63) = {}", name(38, y, -63));
    }

    // the goal platform
    for y in 100..=106 {
        println!("goal col(38,{y},-64) = {}", name(38, y, -64));
    }
    // exact hut floor cells, standable solid with 2 air above
    let airy = |n: &str| matches!(n, "air" | "void_air" | "cave_air");
    for z in -65..=-42 {
        for x in 90..=115 {
            for y in -63..=-57 {
                let n = name(x, y, z);
                if n != "water" && !airy(n)
                    && airy(name(x, y + 1, z)) && airy(name(x, y + 2, z)) {
                    println!("stand x={x} y_feet={} z={z} on {n}", y + 1);
                }
            }
        }
    }
}
