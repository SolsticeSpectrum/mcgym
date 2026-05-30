package com.mcai.agent;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.network.ClientPlayerInteractionManager;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Direction;
import net.minecraft.util.math.MathHelper;

/**
 * Owns the per-tick agent loop: build obs -> run policy -> apply a VALID action.
 *
 * Movement is applied by pressing the player's movement KeyBindings (the same
 * input path a human keyboard drives), so vanilla movement code produces valid
 * move packets. Mining goes through ClientPlayerInteractionManager so the client
 * emits correctly-timed START/STOP_DESTROY_BLOCK packets.
 */
public final class AgentRunner {
    private final OnnxPolicy policy;
    private final ObservationBuilder obs;

    public AgentRunner(OnnxPolicy policy, SchemaRegistry registry) {
        this.policy = policy;
        this.obs = new ObservationBuilder(registry);
    }

    /** One agent step. Returns a short status string for diagnostics, or null. */
    public void tick(MinecraftClient mc) throws Exception {
        ClientPlayerEntity player = mc.player;
        if (player == null || mc.world == null || mc.interactionManager == null) {
            return;
        }

        obs.build(mc);
        float[] logits = policy.run(obs.voxel, obs.scalars, obs.invItemId, obs.invCount);
        ActionSpace action = ActionSpace.decode(logits);

        applyLook(player, action);
        applyMovement(mc, action);
        applyMining(mc, action);
    }

    private void applyLook(ClientPlayerEntity player, ActionSpace action) {
        player.setYaw(player.getYaw() + action.yawDelta);
        player.setPitch(MathHelper.clamp(player.getPitch() + action.pitchDelta, -90.0f, 90.0f));
    }

    private void applyMovement(MinecraftClient mc, ActionSpace action) {
        // forward: +1 -> forward key, -1 -> back key; strafe: +1 -> right(?) consistent
        // with the gym's left/right convention: strafe -1 = left, +1 = right.
        setKey(mc.options.forwardKey, action.forward > 0.5f);
        setKey(mc.options.backKey, action.forward < -0.5f);
        setKey(mc.options.leftKey, action.strafe < -0.5f);
        setKey(mc.options.rightKey, action.strafe > 0.5f);
        setKey(mc.options.jumpKey, action.jump);
        setKey(mc.options.sprintKey, action.sprint);
    }

    private void applyMining(MinecraftClient mc, ActionSpace action) {
        ClientPlayerInteractionManager im = mc.interactionManager;
        if (!action.attack) {
            im.cancelBlockBreaking();
            return;
        }

        // Re-raycast the crosshair (block-only) to find the target.
        HitResult hit = mc.player.raycast(4.5, 1.0f, false);
        if (hit == null || hit.getType() != HitResult.Type.BLOCK || !(hit instanceof BlockHitResult bhr)) {
            im.cancelBlockBreaking();
            return;
        }

        BlockPos pos = bhr.getBlockPos();
        Direction side = bhr.getSide();
        if (mc.world.getBlockState(pos).isAir()) {
            im.cancelBlockBreaking();
            return;
        }

        // attackBlock starts (or restarts on a new block) the dig; per-tick
        // updateBlockBreakingProgress advances it; both emit the proper packets
        // with vanilla timing. interactionManager tracks the "isBreakingBlock"
        // state internally and ignores redundant starts on the same block.
        if (!im.isBreakingBlock()) {
            im.attackBlock(pos, side);
            mc.player.swingHand(mc.player.getActiveHand());
        } else {
            im.updateBlockBreakingProgress(pos, side);
            mc.player.swingHand(mc.player.getActiveHand());
        }
    }

    private static void setKey(KeyBinding key, boolean down) {
        key.setPressed(down);
    }

    /** Release all driven inputs and abort any in-progress mining. */
    public void release(MinecraftClient mc) {
        if (mc.options != null) {
            setKey(mc.options.forwardKey, false);
            setKey(mc.options.backKey, false);
            setKey(mc.options.leftKey, false);
            setKey(mc.options.rightKey, false);
            setKey(mc.options.jumpKey, false);
            setKey(mc.options.sprintKey, false);
        }
        if (mc.interactionManager != null) {
            mc.interactionManager.cancelBlockBreaking();
        }
    }

    public void close() {
        policy.close();
    }
}
