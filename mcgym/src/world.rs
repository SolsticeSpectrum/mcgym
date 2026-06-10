//! headless vanilla worldgen via pumpkin, chunks generated straight into an in memory cache
//! ensure_terrain_chunk is surface only (fast), ensure_chunk runs features so trees exist

use std::collections::HashMap;

use crate::registry::Registry;
use crate::schema::{VOXEL_EDGE, VOXEL_RADIUS};
use pumpkin_data::chunk::Biome;
use pumpkin_data::dimension::Dimension;
use pumpkin_data::{Block, BlockState};
use pumpkin_util::math::position::BlockPos;
use pumpkin_util::world_seed::Seed;
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
        }
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

    /// terrain only, no trees, for read/throughput benchmarking
    pub fn ensure_terrain_chunk(&mut self, cx: i32, cz: i32) {
        if self.chunks.contains_key(&(cx, cz)) {
            return;
        }
        let mut c = ProtoChunk::new(cx, cz, &self.generator);
        c.step_to_biomes(&self.generator);
        c.step_to_noise(&self.generator);
        c.step_to_surface(&self.generator);
        self.chunks.insert((cx, cz), c);
    }

    /// full vanilla generation through features (trees/logs present), cached
    pub fn ensure_chunk(&mut self, cx: i32, cz: i32) {
        if self.chunks.contains_key(&(cx, cz)) {
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

    /// raw block state id at world coords, None if ungenerated or y out of range
    #[inline]
    pub fn block_state_raw(&self, x: i32, y: i32, z: i32) -> Option<u16> {
        let local_y = y - self.bottom_y;
        if local_y < 0 || local_y >= self.height {
            return None;
        }
        let chunk = self.chunks.get(&(x >> 4, z >> 4))?;
        Some(chunk.get_block_state_raw(x & 15, local_y, z & 15))
    }

    /// set block to air, returns the old state id
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

    /// fill voxel grid centred at (cx,cy,cz) with stride, mapped to mcai ints
    /// caches the chunk ptr across cells, loop is dy>dz>dx so neighbours share a chunk
    pub fn fill_voxels(&self, reg: &Registry, cx: i32, cy: i32, cz: i32, stride: i32, out: &mut [i32]) {
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

    /// drop chunks not within radius of any center, bounds memory on long roaming runs
    pub fn retain_chunks_near(&mut self, centers: &[(i32, i32)], radius: i32) {
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
