//! obs/action byte layout, the cross language contract (schema/mcgym_schema.yaml v2)

pub const SCHEMA_VERSION:      i32 = 2;

pub const VOXEL_RADIUS:      usize = 8;
pub const VOXEL_EDGE:        usize = 2 * VOXEL_RADIUS + 1; // 17
pub const VOXEL_CELLS:       usize = VOXEL_EDGE * VOXEL_EDGE * VOXEL_EDGE; // 4913
pub const VOXEL_FAR_STRIDE:    i32 = 4;
pub const MAX_ENTITIES:      usize = 16;
pub const INVENTORY_SLOTS:   usize = 41;
pub const HIDDEN_BLOCK_ID:     i32 = -1;
pub const BOUNDS_DIMS:       usize = 6; // per cell aabb x0 y0 z0 x1 y1 z1 in 16ths

pub const OBS_NBYTES:        usize = 69647;
pub const ACTION_NBYTES:     usize = 27;

#[derive(Clone, Debug, PartialEq)]
pub struct Obs {
    pub schema_version:  i32,
    pub tick:            i64,
    pub agent_id:        i32,
    pub pos:             [f32; 3],
    pub vel:             [f32; 3],
    pub yaw:             f32,
    pub pitch:           f32,
    pub on_ground:       u8,
    pub health:          f32,
    pub food:            f32,
    pub selected_slot:   u8,
    pub voxel_blocks:    Vec<i32>, // len VOXEL_CELLS
    pub voxel_far:       Vec<i32>, // len VOXEL_CELLS
    pub voxel_bounds:    Vec<u8>,  // len VOXEL_CELLS * BOUNDS_DIMS, near grid collision aabbs
    pub target_block:    i32,
    pub target_face:     u8,
    pub target_distance: f32,
    pub target_in_range: u8,
    pub entity_type_id:  [i32; MAX_ENTITIES],
    pub entity_rel_pos:  [[f32; 3]; MAX_ENTITIES],
    pub entity_vel:      [[f32; 3]; MAX_ENTITIES],
    pub entity_yaw:      [f32; MAX_ENTITIES],
    pub entity_health:   [f32; MAX_ENTITIES],
    pub entity_flags:    [u8; MAX_ENTITIES],
    pub inv_item_id:     [i32; INVENTORY_SLOTS],
    pub inv_count:       [u8; INVENTORY_SLOTS],
}

impl Default for Obs {
    fn default() -> Self {
        Self {
            schema_version:  SCHEMA_VERSION,
            tick:            0,
            agent_id:        0,
            pos:             [0.0; 3],
            vel:             [0.0; 3],
            yaw:             0.0,
            pitch:           0.0,
            on_ground:       0,
            health:          0.0,
            food:            0.0,
            selected_slot:   0,
            voxel_blocks:    vec![0; VOXEL_CELLS],
            voxel_far:       vec![0; VOXEL_CELLS],
            voxel_bounds:    vec![0; VOXEL_CELLS * BOUNDS_DIMS],
            target_block:    0,
            target_face:     0,
            target_distance: 0.0,
            target_in_range: 0,
            entity_type_id:  [0; MAX_ENTITIES],
            entity_rel_pos:  [[0.0; 3]; MAX_ENTITIES],
            entity_vel:      [[0.0; 3]; MAX_ENTITIES],
            entity_yaw:      [0.0; MAX_ENTITIES],
            entity_health:   [0.0; MAX_ENTITIES],
            entity_flags:    [0; MAX_ENTITIES],
            inv_item_id:     [0; INVENTORY_SLOTS],
            inv_count:       [0; INVENTORY_SLOTS],
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Action {
    pub forward:       f32,
    pub strafe:        f32,
    pub jump:          u8,
    pub sneak:         u8,
    pub sprint:        u8,
    pub yaw_delta:     f32,
    pub pitch_delta:   f32,
    pub attack:        u8,
    pub use_:          u8,
    pub selected_slot: u8,
    pub inv_op_type:   u8,
    pub inv_slot_a:    i16,
    pub inv_slot_b:    i16,
}

// sequential little endian writer, panics on overrun (layout drifted from OBS_NBYTES)
struct LeWriter<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> LeWriter<'a> {
    fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    #[inline]
    fn bytes(&mut self, b: &[u8]) {
        self.buf[self.pos..self.pos + b.len()].copy_from_slice(b);
        self.pos += b.len();
    }

    #[inline] fn  u8(&mut self, v:  u8) { self.bytes(&[v]); }
    #[inline] fn i16(&mut self, v: i16) { self.bytes(&v.to_le_bytes()); }
    #[inline] fn i32(&mut self, v: i32) { self.bytes(&v.to_le_bytes()); }
    #[inline] fn i64(&mut self, v: i64) { self.bytes(&v.to_le_bytes()); }
    #[inline] fn f32(&mut self, v: f32) { self.bytes(&v.to_le_bytes()); }
}

// sequential little endian reader
struct LeReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> LeReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    #[inline]
    fn take(&mut self, n: usize) -> &[u8] {
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        s
    }

    #[inline] fn  u8(&mut self)  -> u8 { self.take(1)[0] }
    #[inline] fn i16(&mut self) -> i16 { i16::from_le_bytes(self.take(2).try_into().unwrap()) }
    #[inline] fn i32(&mut self) -> i32 { i32::from_le_bytes(self.take(4).try_into().unwrap()) }
    #[inline] fn i64(&mut self) -> i64 { i64::from_le_bytes(self.take(8).try_into().unwrap()) }
    #[inline] fn f32(&mut self) -> f32 { f32::from_le_bytes(self.take(4).try_into().unwrap()) }
}

impl Obs {
    pub fn encode_into(&self, dst: &mut [u8]) {
        assert_eq!(dst.len(), OBS_NBYTES, "obs dst wrong size");
        assert_eq!(self.voxel_blocks.len(), VOXEL_CELLS, "voxel_blocks len");
        assert_eq!(self.voxel_far.len(), VOXEL_CELLS, "voxel_far len");
        assert_eq!(self.voxel_bounds.len(), VOXEL_CELLS * BOUNDS_DIMS, "voxel_bounds len");

        let mut w = LeWriter::new(dst);
        w.i32(self.schema_version);
        w.i64(self.tick);
        w.i32(self.agent_id);
        for v in self.pos { w.f32(v); }
        for v in self.vel { w.f32(v); }

        w.f32(self.yaw);
        w.f32(self.pitch);
        w.u8(self.on_ground);
        w.f32(self.health);
        w.f32(self.food);
        w.u8(self.selected_slot);
        for &v in &self.voxel_blocks { w.i32(v); }
        for &v in &self.voxel_far    { w.i32(v); }
        w.bytes(&self.voxel_bounds);

        w.i32(self.target_block);
        w.u8(self.target_face);
        w.f32(self.target_distance);
        w.u8(self.target_in_range);
        for v in self.entity_type_id   { w.i32(v); }
        for row in self.entity_rel_pos { for v in row { w.f32(v); } }
        for row in self.entity_vel     { for v in row { w.f32(v); } }
        for v in self.entity_yaw       { w.f32(v); }
        for v in self.entity_health    { w.f32(v); }
        for v in self.entity_flags     { w.u8(v);  }
        for v in self.inv_item_id      { w.i32(v); }
        for v in self.inv_count        { w.u8(v);  }

        debug_assert_eq!(w.pos, OBS_NBYTES);
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut buf = vec![0u8; OBS_NBYTES];
        self.encode_into(&mut buf);
        buf
    }

    pub fn decode(src: &[u8]) -> Self {
        assert_eq!(src.len(), OBS_NBYTES, "obs src wrong size");

        let mut r = LeReader::new(src);
        let mut o = Self::default();
        o.schema_version  = r.i32();
        o.tick            = r.i64();
        o.agent_id        = r.i32();
        o.pos             = [r.f32(), r.f32(), r.f32()];
        o.vel             = [r.f32(), r.f32(), r.f32()];
        o.yaw             = r.f32();
        o.pitch           = r.f32();
        o.on_ground       = r.u8();
        o.health          = r.f32();
        o.food            = r.f32();
        o.selected_slot   = r.u8();
        for c in o.voxel_blocks.iter_mut() { *c = r.i32(); }
        for c in o.voxel_far.iter_mut()    { *c = r.i32(); }
        o.voxel_bounds.copy_from_slice(r.take(VOXEL_CELLS * BOUNDS_DIMS));

        o.target_block    = r.i32();
        o.target_face     = r.u8();
        o.target_distance = r.f32();
        o.target_in_range = r.u8();
        for c in o.entity_type_id.iter_mut()   { *c = r.i32(); }
        for row in o.entity_rel_pos.iter_mut() { *row = [r.f32(), r.f32(), r.f32()]; }
        for row in o.entity_vel.iter_mut()     { *row = [r.f32(), r.f32(), r.f32()]; }
        for c in o.entity_yaw.iter_mut()       { *c = r.f32(); }
        for c in o.entity_health.iter_mut()    { *c = r.f32(); }
        for c in o.entity_flags.iter_mut()     { *c = r.u8();  }
        for c in o.inv_item_id.iter_mut()      { *c = r.i32(); }
        for c in o.inv_count.iter_mut()        { *c = r.u8();  }

        debug_assert_eq!(r.pos, OBS_NBYTES);
        o
    }
}

impl Action {
    pub fn decode(src: &[u8]) -> Self {
        assert_eq!(src.len(), ACTION_NBYTES, "action src wrong size");
        let mut r = LeReader::new(src);
        Self {
            forward:       r.f32(),
            strafe:        r.f32(),
            jump:          r.u8(),
            sneak:         r.u8(),
            sprint:        r.u8(),
            yaw_delta:     r.f32(),
            pitch_delta:   r.f32(),
            attack:        r.u8(),
            use_:          r.u8(),
            selected_slot: r.u8(),
            inv_op_type:   r.u8(),
            inv_slot_a:    r.i16(),
            inv_slot_b:    r.i16(),
        }
    }

    pub fn encode_into(&self, dst: &mut [u8]) {
        assert_eq!(dst.len(), ACTION_NBYTES, "action dst wrong size");
        let mut w = LeWriter::new(dst);
        w.f32(self.forward);
        w.f32(self.strafe);
        w.u8(self.jump);
        w.u8(self.sneak);
        w.u8(self.sprint);
        w.f32(self.yaw_delta);
        w.f32(self.pitch_delta);
        w.u8(self.attack);
        w.u8(self.use_);
        w.u8(self.selected_slot);
        w.u8(self.inv_op_type);
        w.i16(self.inv_slot_a);
        w.i16(self.inv_slot_b);
        debug_assert_eq!(w.pos, ACTION_NBYTES);
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut buf = vec![0u8; ACTION_NBYTES];
        self.encode_into(&mut buf);
        buf
    }
}
