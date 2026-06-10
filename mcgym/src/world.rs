//! Headless vanilla world generation via Pumpkin's `pumpkin-world`.
//!
//! No disk, no networking, no async chunk-system: chunks are generated directly into an
//! in-memory cache. Two paths:
//!   * `ensure_terrain_chunk` — biomes->noise->surface only (fast, no trees); for read/throughput.
//!   * `ensure_chunk` — full vanilla generation through the Features stage, so chunks contain
//!     trees/logs (the wood task). Uses Pumpkin's `generate_single_chunk` with a noop block
//!     registry (mobs off, matching the gym requirement).

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

/// Block-behaviour hooks needed during generation. The gym wants vanilla terrain + features but
/// no mob spawning, so placement is always allowed and mob spawning is a noop.
struct GymBlockRegistry;

impl WorldPortalExt for GymBlockRegistry {
    fn can_place_at(
        &self,
        _block: &Block,
        _state: &BlockState,
        _accessor: &dyn BlockAccessor,
        _pos: &BlockPos,
    ) -> bool {
        true
    }

    fn spawn_mobs_for_chunk_generation(
        &self,
        _cache: &mut dyn GenerationCache,
        _biome: &'static Biome,
        _chunk_x: i32,
        _chunk_z: i32,
    ) {
    }
}

/// A generated world: the vanilla generator plus an in-memory chunk cache.
pub struct World {
    generator: VanillaGenerator,
    registry: GymBlockRegistry,
    chunks: HashMap<(i32, i32), ProtoChunk>,
    bottom_y: i32,
    height: i32,
}

impl World {
    pub fn new(seed: i64) -> Self {
        let generator = VanillaGenerator::new(Seed(seed as u64), Dimension::OVERWORLD);
        let probe = ProtoChunk::new(0, 0, &generator);
        let bottom_y = i32::from(probe.bottom_y());
        let height = i32::from(probe.height());
        Self {
            generator,
            registry: GymBlockRegistry,
            chunks: HashMap::new(),
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

    /// Fast terrain-only generation (no trees). For read/throughput benchmarking.
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

    /// Full vanilla generation through Features (trees/logs present). Generates the 3x3
    /// neighbourhood internally; the result is cached so each chunk is produced once.
    pub fn ensure_chunk(&mut self, cx: i32, cz: i32) {
        if self.chunks.contains_key(&(cx, cz)) {
            return;
        }
        let chunk = generate_single_chunk(
            &self.generator.dimension,
            self.generator.biome_mixer_seed,
            &self.generator,
            &self.registry,
            cx,
            cz,
            StagedChunkEnum::Features,
        );
        let proto = match chunk {
            Chunk::Proto(boxed) => *boxed,
            Chunk::Level(_) => unreachable!("Features stage stays a proto-chunk"),
        };
        self.chunks.insert((cx, cz), proto);
    }

    /// Raw block-state id at world coords, or `None` if the chunk isn't generated or y is out
    /// of range. The obs hot path.
    #[inline]
    pub fn block_state_raw(&self, x: i32, y: i32, z: i32) -> Option<u16> {
        let local_y = y - self.bottom_y;
        if local_y < 0 || local_y >= self.height {
            return None;
        }
        let chunk = self.chunks.get(&(x >> 4, z >> 4))?;
        Some(chunk.get_block_state_raw(x & 15, local_y, z & 15))
    }

    /// Break the block at world coords (set it to air), returning the old state id. The agent
    /// then receives the drop directly (item-entity-on-ground step is a later refinement).
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

    /// Fill a voxel grid centred at block (cx,cy,cz) with the given stride, mapped to MCAI
    /// registry ints. Caches the chunk pointer across cells (the loop is dy>dz>dx, X fastest, so
    /// consecutive cells usually share a chunk) — turning ~4913 HashMap lookups per call into a
    /// handful. Byte-identical to per-cell `block_state_raw` + `reg.block`. Same index convention.
    pub fn fill_voxels(&self, reg: &Registry, cx: i32, cy: i32, cz: i32, stride: i32, out: &mut [i32]) {
        let r = VOXEL_RADIUS as i32;
        let edge = VOXEL_EDGE as i32;
        let mut cur: Option<(i32, i32)> = None;
        let mut cur_chunk: Option<&ProtoChunk> = None;
        for dy in -r..=r {
            let wy = cy + dy * stride;
            let local_y = wy - self.bottom_y;
            let y_ok = local_y >= 0 && local_y < self.height;
            for dz in -r..=r {
                let wz = cz + dz * stride;
                for dx in -r..=r {
                    let wx = cx + dx * stride;
                    let index = (((dy + r) * edge + (dz + r)) * edge + (dx + r)) as usize;
                    if !y_ok {
                        out[index] = 0;
                        continue;
                    }
                    let ck = (wx >> 4, wz >> 4);
                    if cur != Some(ck) {
                        cur = Some(ck);
                        cur_chunk = self.chunks.get(&ck);
                    }
                    let sid = cur_chunk.map_or(0, |c| c.get_block_state_raw(wx & 15, local_y, wz & 15));
                    out[index] = reg.block(sid);
                }
            }
        }
    }

    /// Drop generated chunks that are not within `radius` chunks of any center (agent). Bounds
    /// memory on long roaming runs; an agent re-entering an evicted area regenerates it
    /// deterministically. Cheap O(chunks * centers) sweep, called infrequently.
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
