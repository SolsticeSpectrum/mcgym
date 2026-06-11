//! forest spawn picking and wood progress, steel side world queries

use azalea_entity::inventory::Inventory;
use azalea_inventory::ItemStack;
use steel_registry::REGISTRY;
use steel_utils::BlockPos;
use glam::DVec3;

use crate::client::Swarm;
use crate::registry::Registry;
use crate::server::Steel;

const SEARCH:        i32 = 5;
const REACH:         i32 = 31;
const SCAN_TOP:      i32 = 200;
const SCAN_BOTTOM:   i32 = 0;

fn name(steel: &Steel, x: i32, y: i32, z: i32) -> &'static str {
    let sid = steel.world.get_block_state(BlockPos::new(x, y, z));
    REGISTRY.blocks.by_state_id(sid).map_or("air", |b| b.key.path.as_ref())
}

fn airy(n: &str) -> bool {
    n == "air" || n == "cave_air" || n == "void_air"
        || n == "short_grass" || n == "tall_grass" || n == "fern"
}

fn tree(n: &str) -> bool {
    n.ends_with("_log") || n.ends_with("_leaves") || n.ends_with("_wood") || n.ends_with("_stem")
}

// top non tree ground with 2 clear above
fn standable(steel: &Steel, x: i32, z: i32) -> Option<i32> {
    for y in (SCAN_BOTTOM..SCAN_TOP).rev() {
        let n = name(steel, x, y, z);
        if airy(n) {
            continue;
        }
        if tree(n) {
            continue; // canopy or trunk, keep descending
        }
        let fy = y + 1;
        if airy(name(steel, x, fy, z)) && airy(name(steel, x, fy + 1, z)) {
            return Some(fy);
        }
        return None;
    }
    None
}

fn logs(steel: &Steel, ax: i32, az: i32) -> Vec<[i32; 3]> {
    let mut found = Vec::new();
    for x in (ax - REACH)..=(ax + REACH) {
        for z in (az - REACH)..=(az + REACH) {
            for y in SCAN_BOTTOM..SCAN_TOP {
                if name(steel, x, y, z).ends_with("_log") {
                    found.push([x, y, z]);
                }
            }
        }
    }
    found
}

fn yaw_facing(from: [f64; 2], to: [f64; 2]) -> f32 {
    (-(to[0] - from[0])).atan2(to[1] - from[1]).to_degrees() as f32
}

// stand by the log nearest the anchor facing it, anchor surface when no trees in range
pub fn forest(steel: &Steel, ax: i32, az: i32) -> (DVec3, f32) {
    if let Some(log) = logs(steel, ax, az).iter().min_by_key(|p| {
        let (dx, dz) = (p[0] - ax, p[2] - az);
        dx * dx + dz * dz
    }) {
        let (lx, lz) = (log[0], log[2]);
        let mut best: Option<(i64, DVec3)> = None;
        for dx in -SEARCH..=SEARCH {
            for dz in -SEARCH..=SEARCH {
                if dx == 0 && dz == 0 {
                    continue; // not inside the trunk
                }
                let (sx, sz) = (lx + dx, lz + dz);
                if let Some(fy) = standable(steel, sx, sz) {
                    let d   = i64::from(dx * dx + dz * dz);
                    let pos = DVec3::new(f64::from(sx) + 0.5, f64::from(fy), f64::from(sz) + 0.5);
                    if best.is_none_or(|(bd, _)| d < bd) {
                        best = Some((d, pos));
                    }
                }
            }
        }

        if let Some((_, pos)) = best {
            let yaw = yaw_facing([pos.x, pos.z], [f64::from(lx) + 0.5, f64::from(lz) + 0.5]);
            return (pos, yaw);
        }
    }

    let fy = standable(steel, ax, az).unwrap_or(96);
    (DVec3::new(f64::from(ax) + 0.5, f64::from(fy), f64::from(az) + 0.5), 0.0)
}

fn fluid(n: &str) -> bool {
    matches!(n, "water" | "bubble_column" | "kelp" | "kelp_plant" | "seagrass" | "tall_seagrass")
}

// standable cells near the anchor, real floor only, nearest first, fixed map spawns
pub fn pad(steel: &Steel, anchor: DVec3, r: i32) -> Vec<DVec3> {
    let (ax, ay, az) = (anchor.x.floor() as i32, anchor.y.floor() as i32, anchor.z.floor() as i32);
    let mut cells = Vec::new();
    for x in ax - r..=ax + r {
        for z in az - r..=az + r {
            for fy in ay - 2..=ay + 2 {
                let ground = name(steel, x, fy - 1, z);
                if airy(ground) || fluid(ground) || tree(ground) {
                    continue;
                }
                if airy(name(steel, x, fy, z)) && airy(name(steel, x, fy + 1, z)) {
                    cells.push(DVec3::new(f64::from(x) + 0.5, f64::from(fy), f64::from(z) + 0.5));
                    break;
                }
            }
        }
    }
    cells.sort_by(|a, b| {
        let da = (a.x - anchor.x).powi(2) + (a.z - anchor.z).powi(2);
        let db = (b.x - anchor.x).powi(2) + (b.z - anchor.z).powi(2);
        da.total_cmp(&db)
    });
    cells
}

// log count in the azalea inventory, the gym side progress signal
pub fn wood_count(swarm: &Swarm, _reg: &Registry, i: usize) -> i32 {
    let inv: Inventory = swarm.get(i);
    inv.inventory_menu
        .slots()
        .iter()
        .filter_map(|s| match s {
            ItemStack::Present(stack) if stack.kind.to_str().ends_with("_log") => {
                Some(i32::from(stack.count))
            }
            _ => None,
        })
        .sum()
}
