//! vanilla region files into the steel world, just enough for the parkour map

use std::io::Read;
use std::path::Path;

use simdnbt::borrow::NbtCompound;
use steel_core::chunk::chunk_request::ChunkRequestHandle;
use steel_registry::{REGISTRY, RegistryExt};
use steel_utils::types::{ChunkPos, UpdateFlags};
use steel_utils::{BlockPos, BlockStateId, Identifier};

use crate::server::Steel;

// handles own the tickets, dropping them would unload the course
pub fn load(steel: &Steel, dir: &Path) -> Vec<ChunkRequestHandle> {
    let mut handles = Vec::new();
    let mut stubs   = 0usize;
    let mut walls   = 0usize;
    let mut blocks  = 0usize;

    for entry in std::fs::read_dir(dir.join("region")).expect("region dir") {
        let path = entry.expect("region entry").path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let mut it = name.split('.');
        let (Some("r"), Some(rx), Some(rz), Some("mca")) =
            (it.next(), it.next(), it.next(), it.next()) else {
            continue;
        };
        let rx: i32 = rx.parse().expect("region x");
        let rz: i32 = rz.parse().expect("region z");

        let data = std::fs::read(&path).expect("region read");
        for idx in 0..1024usize {
            let loc = u32::from_be_bytes(data[idx * 4..idx * 4 + 4].try_into().unwrap());
            if loc == 0 {
                continue; // chunk absent
            }

            let at   = (loc >> 8) as usize * 4096;
            let len  = u32::from_be_bytes(data[at..at + 4].try_into().unwrap()) as usize;
            let comp = data[at + 4];
            let body = &data[at + 5..at + 4 + len];

            let mut raw = Vec::new();
            match comp {
                2 => { flate2::read::ZlibDecoder::new(body).read_to_end(&mut raw).expect("zlib"); }
                1 => { flate2::read::GzDecoder::new(body).read_to_end(&mut raw).expect("gzip"); }
                3 => raw.extend_from_slice(body),
                c => panic!("unknown region compression {c}"),
            }

            let nbt  = simdnbt::borrow::read(&mut std::io::Cursor::new(&raw[..]))
                .expect("chunk nbt")
                .unwrap();
            let root = nbt.as_compound();
            if root.string("Status").map(|s| s.to_string()) != Some("minecraft:full".into()) {
                stubs += 1;
                continue; // ungenerated border stub
            }

            let cx = rx * 32 + (idx % 32) as i32;
            let cz = rz * 32 + (idx / 32) as i32;
            handles.push(steel.ensure(ChunkPos::new(cx, cz)));
            blocks += fill(steel, &root, cx, cz, &mut walls);
        }
    }

    eprintln!("[sim] map loaded, {blocks} blocks set ({stubs} stubs, {walls} wall blocks skipped)");
    handles
}

// vanilla section format, name+properties palette and packed indices
fn fill(steel: &Steel, root: &NbtCompound, cx: i32, cz: i32, walls: &mut usize) -> usize {
    let mut set = 0usize;
    let air     = REGISTRY
        .blocks
        .by_key(&Identifier::vanilla("air".into()))
        .expect("air block")
        .default_state();

    for section in root.list("sections").expect("sections").compounds().expect("section compounds") {
        let base = i32::from(section.byte("Y").expect("section y")) * 16;
        let Some(bs) = section.compound("block_states") else {
            continue; // no blocks in this section
        };

        let mut states: Vec<Option<BlockStateId>> = Vec::new();
        for entry in bs.list("palette").expect("palette").compounds().expect("palette compounds") {
            let bname = entry.string("Name").expect("palette name").to_string();
            let bname = match bname.strip_prefix("minecraft:").unwrap_or(&bname) {
                // mojang renames since the map's data version
                "chain" => "iron_chain",
                n       => n,
            };
            if bname.ends_with("_wall") {
                // steel has no wall behavior yet, course renders without them
                *walls += 1;
                states.push(None);
                continue;
            }

            let state = match entry.compound("Properties") {
                None        => REGISTRY
                    .blocks
                    .by_key(&Identifier::vanilla(bname.into()))
                    .map(|b| b.default_state()),
                Some(props) => {
                    let kv: Vec<(String, String)> = props
                        .iter()
                        .filter_map(|(k, v)| v.string().map(|s| (k.to_string(), s.to_string())))
                        .collect();
                    let kv: Vec<(&str, &str)> =
                        kv.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
                    REGISTRY
                        .blocks
                        .state_id_from_properties(&Identifier::vanilla(bname.into()), &kv)
                }
            };
            if state.is_none() {
                panic!("unknown block {bname} in chunk {cx},{cz}");
            }
            states.push(state);
        }

        let mut put = |cell: usize, state: BlockStateId| {
            if state == air {
                return;
            }
            let (x, z, y) = ((cell & 15) as i32, ((cell >> 4) & 15) as i32, (cell >> 8) as i32);
            let pos = BlockPos::new(cx * 16 + x, base + y, cz * 16 + z);
            steel.world.set_block(pos, state, UpdateFlags::UPDATE_NONE);
            set += 1;
        };

        match bs.long_array("data") {
            None => {
                // single entry palette fills the whole section
                if let Some(state) = states[0] {
                    for cell in 0..4096 {
                        put(cell, state);
                    }
                }
            }
            Some(packed) => {
                let bits = (usize::BITS - (states.len() - 1).leading_zeros()).max(4) as usize;
                let per  = 64 / bits;
                let mask = (1u64 << bits) - 1;
                for cell in 0..4096usize {
                    let word = packed[cell / per] as u64;
                    let pi   = ((word >> ((cell % per) * bits)) & mask) as usize;
                    if let Some(state) = states[pi] {
                        put(cell, state);
                    }
                }
            }
        }
    }

    set
}
