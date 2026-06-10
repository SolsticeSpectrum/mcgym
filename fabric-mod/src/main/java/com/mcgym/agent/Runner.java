package com.mcgym.agent;

import com.mcgym.policy.Actions;
import com.mcgym.policy.Policy;
import com.mcgym.schema.Registry;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.network.ClientPlayerInteractionManager;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.MathHelper;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Per tick agent loop, build obs, run policy, apply the action.
 *
 * Movement works by replacing the player input object with ours, the meteor and
 * baritone pattern, keybindings alone never move the player in 1.21. Mining goes
 * through the interaction manager so the client emits correctly timed packets.
 */
public final class Runner {

    private static final Logger LOG = LoggerFactory.getLogger("mcgym");
    private static final int LOG_EVERY = 10;

    private final Policy policy;
    private final Obs obs;
    private final Input input = new Input();
    private final Registry registry;

    // restored on stop, non null only while our input is installed
    private net.minecraft.client.input.Input prev;
    private long ticks;

    public Runner(Policy policy, Registry registry) {
        this.policy = policy;
        this.registry = registry;
        this.obs = new Obs(registry);
    }

    public void tick(MinecraftClient mc) throws Exception {
        ClientPlayerEntity player = mc.player;
        if (player == null || mc.world == null || mc.interactionManager == null) return;

        // install our input once the player exists
        if (prev == null) {
            prev = player.input;
            player.input = input;
        }

        obs.build(mc);
        float[] logits = policy.run(obs.voxel, obs.voxelFar, obs.target, obs.scalars, obs.invId, obs.invCount);
        Actions act = Actions.decode(logits);

        look(player, act);
        input.set(act);
        mine(mc, act);

        telemetry(player, act);
    }

    private void look(ClientPlayerEntity player, Actions act) {
        // exactly the policys look delta, no assist
        player.setYaw(MathHelper.wrapDegrees(player.getYaw() + act.yaw));
        player.setPitch(MathHelper.clamp(player.getPitch() + act.pitch, -90.0f, 90.0f));
    }

    private void mine(MinecraftClient mc, Actions act) {
        ClientPlayerInteractionManager im = mc.interactionManager;
        if (!act.attack) {
            im.cancelBlockBreaking();
            return;
        }

        // re raycast the crosshair like the gym does
        double reach = mc.player.getBlockInteractionRange();
        if (reach <= 0.0) reach = Obs.REACH;

        HitResult hit = mc.player.raycast(reach, 1.0f, false);
        if (!(hit instanceof BlockHitResult bhr) || hit.getType() != HitResult.Type.BLOCK
                || mc.world.getBlockState(bhr.getBlockPos()).isAir()) {
            im.cancelBlockBreaking();
            return;
        }

        // attackBlock starts the dig, updateBlockBreakingProgress advances it,
        // both emit proper packets with vanilla timing
        if (!im.isBreakingBlock()) im.attackBlock(bhr.getBlockPos(), bhr.getSide());
        else im.updateBlockBreakingProgress(bhr.getBlockPos(), bhr.getSide());
        mc.player.swingHand(mc.player.getActiveHand());
    }

    private void telemetry(ClientPlayerEntity player, Actions act) {
        if (ticks++ % LOG_EVERY != 0) return;

        int logs = 0;
        for (int id : obs.voxel)
            if (registry.isLogBlock(id)) logs++;

        int wood = 0;
        for (int i = 0; i < obs.invId.length; i++)
            if (registry.isLogItem(obs.invId[i])) wood += (int) obs.invCount[i];

        // target distance and in range live in the scalars
        LOG.info("[mcgym] tick={} x={} z={} yaw={} pitch={} hp={} logs={} aimed={} target={} dist={} wood={} | fwd={} strafe={} jump={} sprint={} yawD={} pitchD={} attack={}",
            ticks,
            String.format("%.1f", player.getX()),
            String.format("%.1f", player.getZ()),
            String.format("%.1f", player.getYaw()),
            String.format("%.1f", player.getPitch()),
            String.format("%.1f", player.getHealth()),
            logs,
            obs.scalars[11] != 0.0f,
            obs.target,
            String.format("%.2f", obs.scalars[10] * 8.0f),
            wood,
            act.forward, act.strafe, act.jump, act.sprint, act.yaw, act.pitch, act.attack);
    }

    /** Restore the players input and abort any mining. */
    public void release(MinecraftClient mc) {
        input.clear();
        if (mc.player != null && prev != null) mc.player.input = prev;
        prev = null;
        if (mc.interactionManager != null) mc.interactionManager.cancelBlockBreaking();
    }

    public void close() {
        policy.close();
    }
}
