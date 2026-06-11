//! headless vanilla worldgen via pumpkin, chunks generated straight into an in
//! memory cache, or loaded read only from an anvil world

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use crate::registry::Registry;
use crate::schema::{VOXEL_EDGE, VOXEL_RADIUS};
use pumpkin_data::chunk::Biome;
use pumpkin_data::dimension::Dimension;
use pumpkin_data::{Block, BlockState};
use crate::schema::BOUNDS_DIMS;
use pumpkin_util::math::position::BlockPos;
use pumpkin_util::world_seed::Seed;
use pumpkin_nbt::Nbt;
use pumpkin_nbt::compound::NbtCompound;
use pumpkin_nbt::deserializer::NbtReadHelperJava;
use pumpkin_nbt::tag::NbtTag;
use pumpkin_world::ProtoChunk;
use pumpkin_world::chunk_system::{Chunk, StagedChunkEnum, generate_single_chunk};
use pumpkin_world::generation::generator::{GeneratorInit, VanillaGenerator};
use pumpkin_world::generation::proto_chunk::GenerationCache;
use pumpkin_world::world::{BlockAccessor, WorldPortalExt};

// generation hooks: allow all placement, no mob spawning
struct Hooks;

impl WorldPortalExt for Hooks {
    fn can_place_at(
        &self,
        _block:    &Block,
        _state:    &BlockState,
        _accessor: &dyn BlockAccessor,
        _pos:      &BlockPos,
    ) -> bool {
        true
    }

    fn spawn_mobs_for_chunk_generation(
        &self,
        _cache:   &mut dyn GenerationCache,
        _biome:   &'static Biome,
        _chunk_x: i32,
        _chunk_z: i32,
    ) {
    }
}

pub struct World {
    generator: VanillaGenerator,
    hooks:     Hooks,
    chunks:    HashMap<(i32, i32), ProtoChunk>,
    bottom_y:  i32,
    height:    i32,
    fixed:     bool, // loaded map, never generate
}

impl World {
    pub fn new(seed: i64) -> Self {
        let generator = VanillaGenerator::new(Seed(seed as u64), Dimension::OVERWORLD);
        let probe     = ProtoChunk::new(0, 0, &generator);
        let bottom_y  = i32::from(probe.bottom_y());
        let height    = i32::from(probe.height());
        Self {
            generator,
            hooks:    Hooks,
            chunks:   HashMap::new(),
            bottom_y,
            height,
            fixed:    false,
        }
    }

    // anvil world loaded read only, missing chunks read as air, vanilla saves
    // use compound palettes which pumpkins own disk dialect does not, so chunks
    // are built from the generic nbt tree and resolved through pumpkin data
    pub fn load(dir: &Path) -> std::io::Result<Self> {
        let mut w = Self::new(0);
        w.fixed   = true;

        let mut stubs = 0usize;
        for entry in std::fs::read_dir(dir.join("region"))? {
            let path = entry?.path();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let mut it = name.split('.');
            let (Some("r"), Some(rx), Some(rz), Some("mca")) =
                (it.next(), it.next(), it.next(), it.next()) else {
                continue;
            };
            let rx: i32 = rx.parse().map_err(std::io::Error::other)?;
            let rz: i32 = rz.parse().map_err(std::io::Error::other)?;

            let data = std::fs::read(&path)?;
            for idx in 0..1024usize {
                let loc = u32::from_be_bytes(data[idx * 4..idx * 4 + 4].try_into().unwrap());
                if loc == 0 {
                    continue; // chunk absent
                }

                let at   = (loc >> 8) as usize * 4096;
                let len  = u32::from_be_bytes(data[at..at + 4].try_into().unwrap()) as usize;
                let comp = data[at + 4];
                let body = &data[at + 5..at + 4 + len];

                let mut nbt = Vec::new();
                match comp {
                    2 => { flate2::read::ZlibDecoder::new(body).read_to_end(&mut nbt)?; }
                    1 => { flate2::read::GzDecoder::new(body).read_to_end(&mut nbt)?; }
                    3 => nbt.extend_from_slice(body),
                    c => return Err(std::io::Error::other(format!("unknown compression {c}"))),
                }

                let cx   = rx * 32 + (idx % 32) as i32;
                let cz   = rz * 32 + (idx / 32) as i32;
                let tree = Nbt::read(&mut NbtReadHelperJava::new(std::io::Cursor::new(&nbt[..])))
                    .map_err(|e| std::io::Error::other(format!("chunk {cx},{cz} nbt: {e}")))?;
                if tree.root_tag.get_string("Status") != Some("minecraft:full") {
                    stubs += 1;
                    continue; // ungenerated border stub
                }

                let mut proto = ProtoChunk::new(cx, cz, &w.generator);
                if Self::fill_proto(&tree.root_tag, cx, cz, &mut proto).is_none() {
                    return Err(std::io::Error::other(format!("chunk {cx},{cz} malformed")));
                }
                w.chunks.insert((cx, cz), proto);
            }
        }

        eprintln!("[mcgym] map loaded, {} chunks ({stubs} stubs skipped)", w.chunks.len());
        Ok(w)
    }

    // vanilla section format, palette of name+properties compounds and packed
    // indices, 1.16 style with no entries crossing longs
    fn fill_proto(root: &NbtCompound, cx: i32, cz: i32, proto: &mut ProtoChunk) -> Option<()> {
        for tag in root.get_list("sections")? {
            let NbtTag::Compound(section) = tag else {
                return None;
            };
            let base = i32::from(section.get_byte("Y")?) * 16;
            let Some(bs) = section.get_compound("block_states") else {
                continue; // no blocks in this section
            };

            let mut states = Vec::new();
            for e in bs.get_list("palette")? {
                let NbtTag::Compound(entry) = e else {
                    return None;
                };
                let bname = entry.get_string("Name")?;
                // mojang renames since the maps data version
                let bname = match bname.strip_prefix("minecraft:").unwrap_or(bname) {
                    "chain" => "iron_chain",
                    n       => n,
                };
                let Some(block) = Block::from_name(bname) else {
                    eprintln!("[mcgym] unknown block {bname}");
                    return None;
                };
                let state = match entry.get_compound("Properties") {
                    None        => block.default_state.id,
                    Some(props) => {
                        let kv: Vec<(&str, &str)> = props
                            .child_tags
                            .iter()
                            .filter_map(|(k, v)| match v {
                                NbtTag::String(s) => Some((&**k, &**s)),
                                _                 => None,
                            })
                            .collect();
                        block.from_properties(&kv).to_state_id(block)
                    }
                };
                states.push(state);
            }

            match bs.get_long_array("data") {
                None => {
                    // single entry palette fills the whole section
                    let state = BlockState::from_id(states[0]);
                    if state.is_air() {
                        continue;
                    }
                    for y in 0..16 {
                        for z in 0..16 {
                            for x in 0..16 {
                                proto.set_block_state(cx * 16 + x, base + y, cz * 16 + z, state);
                            }
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
                        let state = BlockState::from_id(states[pi]);
                        if state.is_air() {
                            continue;
                        }
                        let (x, z, y) = ((cell & 15) as i32, ((cell >> 4) & 15) as i32, (cell >> 8) as i32);
                        proto.set_block_state(cx * 16 + x, base + y, cz * 16 + z, state);
                    }
                }
            }
        }

        Some(())
    }

    pub fn bottom_y(&self) -> i32 {
        self.bottom_y
    }
    pub fn height(&self) -> i32 {
        self.height
    }
    pub fn top_y(&self) -> i32 {
        self.bottom_y + self.height
    }

    // surface only, no trees, for read/throughput benchmarking
    pub fn ensure_terrain_chunk(&mut self, cx: i32, cz: i32) {
        if self.fixed || self.chunks.contains_key(&(cx, cz)) {
            return;
        }

        let mut c = ProtoChunk::new(cx, cz, &self.generator);
        c.step_to_biomes(&self.generator);
        c.step_to_noise(&self.generator);
        c.step_to_surface(&self.generator);

        self.chunks.insert((cx, cz), c);
    }

    // full vanilla generation through features
    pub fn ensure_chunk(&mut self, cx: i32, cz: i32) {
        if self.fixed || self.chunks.contains_key(&(cx, cz)) {
            return;
        }

        let chunk = generate_single_chunk(
            &self.generator.dimension,
            self.generator.biome_mixer_seed,
            &self.generator,
            &self.hooks,
            cx,
            cz,
            StagedChunkEnum::Features,
        );
        let proto = match chunk {
            Chunk::Proto(boxed) => *boxed,
            Chunk::Level(_)     => unreachable!("features stage stays a proto chunk"),
        };

        self.chunks.insert((cx, cz), proto);
    }

    #[inline]
    pub fn block_state_raw(&self, x: i32, y: i32, z: i32) -> Option<u16> {
        let local_y = y - self.bottom_y;
        if local_y < 0 || local_y >= self.height {
            return None;
        }

        let chunk = self.chunks.get(&(x >> 4, z >> 4))?;
        Some(chunk.get_block_state_raw(x & 15, local_y, z & 15))
    }

    // set block to air, returns the old state id
    pub fn break_block(&mut self, x: i32, y: i32, z: i32) -> Option<u16> {
        let local_y = y - self.bottom_y;
        if local_y < 0 || local_y >= self.height {
            return None;
        }

        let chunk = self.chunks.get_mut(&(x >> 4, z >> 4))?;
        let old = chunk.get_block_state_raw(x & 15, local_y, z & 15);
        chunk.set_block_state(x, y, z, Block::AIR.default_state);

        Some(old)
    }

    // chunk ptr cached across cells, loop is dy>dz>dx so neighbours share a chunk
    pub fn fill_voxels(
        &self,
        reg:    &Registry,
        cx:     i32,
        cy:     i32,
        cz:     i32,
        stride: i32,
        out:    &mut [i32]
    ) {
        let r    = VOXEL_RADIUS as i32;
        let edge = VOXEL_EDGE as i32;
        
        let mut cur:       Option<(i32, i32)>  = None;
        let mut cur_chunk: Option<&ProtoChunk> = None;
        for dy in -r..=r {
            let wy      = cy + dy * stride;
            let local_y = wy - self.bottom_y;
            let y_ok    = local_y >= 0 && local_y < self.height;
            for dz in -r..=r {
                let wz = cz + dz * stride;
                for dx in -r..=r {
                    let wx  = cx + dx * stride;
                    let idx = (((dy + r) * edge + (dz + r)) * edge + (dx + r)) as usize;
                    if !y_ok {
                        out[idx] = 0;
                        continue;
                    }

                    let ck = (wx >> 4, wz >> 4);
                    if cur != Some(ck) {
                        cur       = Some(ck);
                        cur_chunk = self.chunks.get(&ck);
                    }

                    let sid  = cur_chunk.map_or(0, |c| c.get_block_state_raw(wx & 15, local_y, wz & 15));
                    out[idx] = reg.block(sid);
                }
            }
        }
    }

    // per cell union collision aabb in 16ths, x0 y0 z0 x1 y1 z1, air all zero
    pub fn fill_bounds(&self, cx: i32, cy: i32, cz: i32, out: &mut [u8]) {
        let r    = VOXEL_RADIUS as i32;
        let edge = VOXEL_EDGE as i32;

        let mut cur:       Option<(i32, i32)>  = None;
        let mut cur_chunk: Option<&ProtoChunk> = None;
        for dy in -r..=r {
            let wy      = cy + dy;
            let local_y = wy - self.bottom_y;
            let y_ok    = local_y >= 0 && local_y < self.height;
            for dz in -r..=r {
                let wz = cz + dz;
                for dx in -r..=r {
                    let wx  = cx + dx;
                    let idx = ((((dy + r) * edge + (dz + r)) * edge + (dx + r)) as usize) * BOUNDS_DIMS;
                    let at  = &mut out[idx..idx + BOUNDS_DIMS];
                    at.fill(0);
                    if !y_ok {
                        continue;
                    }

                    let ck = (wx >> 4, wz >> 4);
                    if cur != Some(ck) {
                        cur       = Some(ck);
                        cur_chunk = self.chunks.get(&ck);
                    }
                    let Some(chunk) = cur_chunk else {
                        continue;
                    };

                    let sid = chunk.get_block_state_raw(wx & 15, local_y, wz & 15);
                    let mut lo = [f64::MAX; 3];
                    let mut hi = [f64::MIN; 3];
                    let mut any = false;
                    for shape in BlockState::from_id(sid).get_block_collision_shapes() {
                        any = true;
                        lo[0] = lo[0].min(shape.min.x); hi[0] = hi[0].max(shape.max.x);
                        lo[1] = lo[1].min(shape.min.y); hi[1] = hi[1].max(shape.max.y);
                        lo[2] = lo[2].min(shape.min.z); hi[2] = hi[2].max(shape.max.z);
                    }
                    if !any {
                        continue;
                    }

                    for ax in 0..3 {
                        at[ax]     = (lo[ax] * 16.0).clamp(0.0, 255.0) as u8;
                        at[ax + 3] = (hi[ax] * 16.0).clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    }

    // drop chunks not within radius of any center
    pub fn retain_chunks_near(&mut self, centers: &[(i32, i32)], radius: i32) {
        if self.fixed {
            return; // loaded maps stay resident
        }

        self.chunks.retain(|&(cx, cz), _| {
            centers
                .iter()
                .any(|&(ax, az)| (cx - ax).abs() <= radius && (cz - az).abs() <= radius)
        });
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}
