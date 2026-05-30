package com.mcai.agent;

import net.minecraft.block.Block;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.player.PlayerInventory;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Direction;
import net.minecraft.util.math.Vec3d;

/**
 * Builds the four ONNX policy inputs from the live client player + world.
 *
 * This is a 1:1 port of the trainer's obs_to_tensors (mcai_train.models.policy)
 * combined with the canonical server-side packing (McaiGymRuntime / McaiObsCodec).
 * Field order, voxel indexing, the scalar layout, and inventory slot order are
 * replicated exactly so the client observation is byte-compatible with training.
 */
public final class ObservationBuilder {
    public static final int VOXEL_RADIUS = 8;
    public static final int VOXEL_EDGE = 17;            // 2*8+1
    public static final int VOXEL_COUNT = 4913;         // 17^3
    public static final int INV_SLOTS = 41;             // 36 main + 4 armor + 1 offhand
    public static final double DEFAULT_BLOCK_REACH = 4.5;

    public final long[] voxel = new long[VOXEL_COUNT];
    public final float[] scalars = new float[OnnxPolicy.SCALARS];
    public final long[] invItemId = new long[INV_SLOTS];
    public final float[] invCount = new float[INV_SLOTS];

    /** Block id of the raycast target this build (0 = none/air), for telemetry. */
    public int targetBlockId;

    private final SchemaRegistry registry;

    public ObservationBuilder(SchemaRegistry registry) {
        this.registry = registry;
    }

    /** Populate the four input arrays from the current client state. */
    public void build(MinecraftClient mc) {
        ClientPlayerEntity player = mc.player;
        ClientWorld world = mc.world;
        if (player == null || world == null) {
            return;
        }

        // --- 1. Voxel grid (int64[4913]) -----------------------------------
        // center = floor(playerPos), index = ((dy+8)*17 + (dz+8))*17 + (dx+8),
        // loop order dy -> dz -> dx (matches McaiGymRuntime.fillVoxels exactly).
        BlockPos center = BlockPos.ofFloored(player.getX(), player.getY(), player.getZ());
        BlockPos.Mutable cursor = new BlockPos.Mutable();
        for (int dy = -VOXEL_RADIUS; dy <= VOXEL_RADIUS; dy++) {
            for (int dz = -VOXEL_RADIUS; dz <= VOXEL_RADIUS; dz++) {
                for (int dx = -VOXEL_RADIUS; dx <= VOXEL_RADIUS; dx++) {
                    int index = ((dy + VOXEL_RADIUS) * VOXEL_EDGE + (dz + VOXEL_RADIUS)) * VOXEL_EDGE + (dx + VOXEL_RADIUS);
                    cursor.set(center.getX() + dx, center.getY() + dy, center.getZ() + dz);
                    Block block = world.getBlockState(cursor).getBlock();
                    voxel[index] = registry.blockId(Registries.BLOCK.getId(block).toString());
                }
            }
        }

        // --- 2. Target raycast (block-only, no fluids) ---------------------
        // Mirrors the gym's player.pick(blockInteractionRange(), 1.0F, false):
        // same reach (the block-interaction-range attribute, ~4.5) and same
        // fallback so target_* is byte-compatible with training.
        double reach = player.getBlockInteractionRange();
        if (reach <= 0.0) {
            reach = DEFAULT_BLOCK_REACH;
        }
        boolean targetInRange = false;
        float targetDistance = 0.0f;
        int targetFace = 255; // sentinel: no face
        targetBlockId = 0;

        HitResult hit = player.raycast(reach, 1.0f, false);
        if (hit != null && hit.getType() == HitResult.Type.BLOCK && hit instanceof BlockHitResult bhr) {
            BlockPos hitPos = bhr.getBlockPos();
            if (!world.getBlockState(hitPos).isAir()) {
                Direction face = bhr.getSide();
                targetFace = face.ordinal();
                targetDistance = (float) player.getEyePos().distanceTo(bhr.getPos());
                targetInRange = true;
                targetBlockId = registry.blockId(Registries.BLOCK.getId(world.getBlockState(hitPos).getBlock()).toString());
            }
        }

        // --- 3. Scalars (float32[18]) --------------------------------------
        // Order: velX,velY,velZ, sin(yaw),cos(yaw), sin(pitch),cos(pitch),
        //        onGround, health/20, food/20, targetDistance/8, targetInRange,
        //        faceOneHot[6].
        Vec3d vel = player.getVelocity();
        double yawRad = Math.toRadians(player.getYaw());
        double pitchRad = Math.toRadians(player.getPitch());
        float health = player.getHealth();
        int food = player.getHungerManager().getFoodLevel();

        int s = 0;
        scalars[s++] = (float) vel.x;
        scalars[s++] = (float) vel.y;
        scalars[s++] = (float) vel.z;
        scalars[s++] = (float) Math.sin(yawRad);
        scalars[s++] = (float) Math.cos(yawRad);
        scalars[s++] = (float) Math.sin(pitchRad);
        scalars[s++] = (float) Math.cos(pitchRad);
        scalars[s++] = player.isOnGround() ? 1.0f : 0.0f;
        scalars[s++] = health / 20.0f;
        scalars[s++] = food / 20.0f;
        scalars[s++] = targetDistance / 8.0f;
        scalars[s++] = targetInRange ? 1.0f : 0.0f;
        // faceOneHot[6]: valid faces 0..5, sentinel (255) -> all zero.
        for (int f = 0; f < 6; f++) {
            scalars[s++] = (targetFace == f) ? 1.0f : 0.0f;
        }

        // --- 4. Inventory (int64 ids[41], float32 counts[41]) --------------
        // Slot order: main 0..35, armor 36..39 (feet,legs,chest,head), offhand 40.
        // PlayerInventory.getStack uses this combined container indexing.
        PlayerInventory inv = player.getInventory();
        for (int slot = 0; slot < INV_SLOTS; slot++) {
            ItemStack stack = inv.getStack(slot);
            if (stack.isEmpty()) {
                invItemId[slot] = 0;
                invCount[slot] = 0.0f;
            } else {
                invItemId[slot] = registry.itemId(Registries.ITEM.getId(stack.getItem()).toString());
                invCount[slot] = stack.getCount();
            }
        }
    }

    /** Whether the most recent build saw a block target in range. */
    public boolean hasTarget() {
        // targetInRange is stored at scalar index 11.
        return scalars[11] != 0.0f;
    }
}
