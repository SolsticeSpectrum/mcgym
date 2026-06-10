//! Benchmark headless vanilla worldgen + block-read throughput — the two costs that
//! made the Java gym CPU-bound (~14 TPS/world). Run: `cargo run --release --bin genbench`.

use std::time::Instant;

use mcai_gym::world::World;
use pumpkin_data::Block;

fn main() {
    let seed = 42i64;
    let radius_chunks = 8; // 16x16 = 256 chunks ~ a 256-block square
    let side = radius_chunks * 2;

    // --- worldgen ---
    let mut world = World::new(seed);
    let t0 = Instant::now();
    for cx in -radius_chunks..radius_chunks {
        for cz in -radius_chunks..radius_chunks {
            world.ensure_terrain_chunk(cx, cz);
        }
    }
    let gen_dt = t0.elapsed();
    let n_chunks = world.chunk_count();
    println!(
        "gen: {n_chunks} chunks in {:.3}s = {:.0} chunks/s ({:.2} ms/chunk)",
        gen_dt.as_secs_f64(),
        n_chunks as f64 / gen_dt.as_secs_f64(),
        gen_dt.as_secs_f64() * 1000.0 / n_chunks as f64,
    );

    // --- full-volume block reads (every block in every generated chunk) ---
    let (bx0, bx1) = (-radius_chunks * 16, radius_chunks * 16);
    let (bz0, bz1) = (-radius_chunks * 16, radius_chunks * 16);
    let (y0, y1) = (world.bottom_y(), world.top_y());
    let mut nonair = 0u64;
    let mut reads = 0u64;
    let t1 = Instant::now();
    for x in bx0..bx1 {
        for z in bz0..bz1 {
            for y in y0..y1 {
                if let Some(id) = world.block_state_raw(x, y, z) {
                    reads += 1;
                    if id != 0 {
                        nonair += 1;
                    }
                }
            }
        }
    }
    let read_dt = t1.elapsed();
    println!(
        "read: {reads} blocks in {:.3}s = {:.1} M blocks/s ({nonair} non-air)",
        read_dt.as_secs_f64(),
        reads as f64 / read_dt.as_secs_f64() / 1e6,
    );

    // --- obs-shaped reads: a 17^3 voxel cube per agent, many agents ---
    let agents = 64;
    let r = 8i32;
    let mut sink = 0u64;
    let t2 = Instant::now();
    for a in 0..agents {
        // spread sample centers across the generated area, near the surface
        let cx = bx0 + 8 + (a * 37) % (side * 16 - 17);
        let cz = bz0 + 8 + (a * 53) % (side * 16 - 17);
        let cy = 72;
        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    if let Some(id) = world.block_state_raw(cx + dx, cy + dy, cz + dz) {
                        sink = sink.wrapping_add(id as u64);
                    }
                }
            }
        }
    }
    let obs_dt = t2.elapsed();
    let cells = agents as f64 * 17.0 * 17.0 * 17.0;
    println!(
        "obs: {agents} agents x 17^3 voxel cube in {:.3} ms = {:.1} M cells/s (sink={sink})",
        obs_dt.as_secs_f64() * 1000.0,
        cells / obs_dt.as_secs_f64() / 1e6,
    );
    println!(
        "  -> at this read rate, one 64-agent voxel fill costs ~{:.3} ms (excl. physics/sim)",
        obs_dt.as_secs_f64() * 1000.0,
    );

    // --- full vanilla gen WITH features (trees) + confirm logs appear ---
    let mut fworld = World::new(seed);
    let fr = 3i32; // 6x6 chunks generated to the Features stage
    let t3 = Instant::now();
    for cx in 0..fr * 2 {
        for cz in 0..fr * 2 {
            fworld.ensure_chunk(cx, cz);
        }
    }
    let fdt = t3.elapsed();
    let mut logs = 0u64;
    let mut leaves = 0u64;
    for x in 0..fr * 2 * 16 {
        for z in 0..fr * 2 * 16 {
            for y in (fworld.bottom_y())..(fworld.top_y()) {
                if let Some(id) = fworld.block_state_raw(x, y, z) {
                    let name = Block::from_state_id(id).name;
                    if name.ends_with("_log") || name.ends_with("_stem") {
                        logs += 1;
                    } else if name.ends_with("_leaves") {
                        leaves += 1;
                    }
                }
            }
        }
    }
    let nfc = fworld.chunk_count();
    println!(
        "feat: {nfc} chunks w/ features in {:.2}s ({:.0} ms/chunk), {logs} log + {leaves} leaf blocks",
        fdt.as_secs_f64(),
        fdt.as_secs_f64() * 1000.0 / nfc as f64,
    );
}
