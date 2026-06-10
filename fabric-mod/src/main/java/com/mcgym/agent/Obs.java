package com.mcgym.agent;

import com.mcgym.policy.Policy;
import com.mcgym.schema.Registry;

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

public final class Obs {

    public static final int    RADIUS = 8;
    public static final int    EDGE   = 17;
    public static final int    COUNT  = 4913; // 17^3
    public static final int    STRIDE = 4;    // far shell samples every 4th block
    public static final int    SLOTS  = 41;   // 36 main + 4 armor + 1 offhand
    public static final double REACH  = 4.5;

    public final int[]   voxel    = new int[COUNT];
    public final int[]   voxelFar = new int[COUNT];
    public final float[] scalars  = new float[Policy.SCALARS];
    public final int[]   invId    = new int[SLOTS];
    public final float[] invCount = new float[SLOTS];

    public int target;

    private final Registry registry;

    public Obs(Registry registry) {
        this.registry = registry;
    }

    public void build(MinecraftClient mc) {
        ClientPlayerEntity player = mc.player;
        ClientWorld world = mc.world;
        if (player == null || world == null) return;

        // voxel grids, index = ((dy+8)*17 + (dz+8))*17 + (dx+8), loop dy dz dx
        // unloaded chunks read as air like the gym
        BlockPos center = BlockPos.ofFloored(player.getX(), player.getY(), player.getZ());
        BlockPos.Mutable cursor = new BlockPos.Mutable();
        for (int dy = -RADIUS; dy <= RADIUS; dy++) {
            for (int dz = -RADIUS; dz <= RADIUS; dz++) {
                for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                    int idx = ((dy + RADIUS) * EDGE + (dz + RADIUS)) * EDGE + (dx + RADIUS);

                    cursor.set(center.getX() + dx, center.getY() + dy, center.getZ() + dz);
                    Block near = world.getBlockState(cursor).getBlock();
                    voxel[idx] = registry.blockId(Registries.BLOCK.getId(near).toString());

                    cursor.set(center.getX() + dx * STRIDE, center.getY() + dy * STRIDE, center.getZ() + dz * STRIDE);
                    Block far = world.getBlockState(cursor).getBlock();
                    voxelFar[idx] = registry.blockId(Registries.BLOCK.getId(far).toString());
                }
            }
        }

        // target raycast, block only no fluids
        double reach = player.getBlockInteractionRange();
        if (reach <= 0.0) reach = REACH;

        boolean aimed = false;
        float    dist = 0.0f;
        int      face = 255; // sentinel, no face
        target        = 0;

        HitResult hit = player.raycast(reach, 1.0f, false);
        if (hit instanceof BlockHitResult bhr && hit.getType() == HitResult.Type.BLOCK) {
            BlockPos pos = bhr.getBlockPos();
            if (!world.getBlockState(pos).isAir()) {
                Direction side = bhr.getSide();
                face = side.ordinal();
                dist = (float) player.getEyePos().distanceTo(bhr.getPos());
                aimed = true;
                target = registry.blockId(Registries.BLOCK.getId(world.getBlockState(pos).getBlock()).toString());
            }
        }

        // scalars, velX velY velZ, sin cos yaw, sin cos pitch, onGround,
        // health/20, food/20, dist/8, aimed, face onehot[6]
        Vec3d vel = player.getVelocity();
        double yaw = Math.toRadians(player.getYaw());
        double pitch = Math.toRadians(player.getPitch());

        int s = 0;
        scalars[s++] = (float) vel.x;
        scalars[s++] = (float) vel.y;
        scalars[s++] = (float) vel.z;
        scalars[s++] = (float) Math.sin(yaw);
        scalars[s++] = (float) Math.cos(yaw);
        scalars[s++] = (float) Math.sin(pitch);
        scalars[s++] = (float) Math.cos(pitch);
        scalars[s++] = player.isOnGround() ? 1.0f : 0.0f;
        scalars[s++] = player.getHealth() / 20.0f;
        scalars[s++] = player.getHungerManager().getFoodLevel() / 20.0f;
        scalars[s++] = dist / 8.0f;
        scalars[s++] = aimed ? 1.0f : 0.0f;
        for (int f = 0; f < 6; f++)
            scalars[s++] = (face == f) ? 1.0f : 0.0f;

        // inventory, main 0..35, armor 36..39, offhand 40
        PlayerInventory inv = player.getInventory();
        for (int slot = 0; slot < SLOTS; slot++) {
            ItemStack stack = inv.getStack(slot);
            if (stack.isEmpty()) {
                invId[slot] = 0;
                invCount[slot] = 0.0f;
            } else {
                invId[slot] = registry.itemId(Registries.ITEM.getId(stack.getItem()).toString());
                invCount[slot] = stack.getCount();
            }
        }
    }
}
