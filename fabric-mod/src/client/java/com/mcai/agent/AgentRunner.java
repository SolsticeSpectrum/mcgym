package com.mcai.agent;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.input.Input;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.network.ClientPlayerInteractionManager;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Direction;
import net.minecraft.util.math.MathHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Owns the per-tick agent loop: build obs -> run policy -> apply a VALID action.
 *
 * Movement is actuated by REPLACING the player's input object
 * ({@code ClientPlayerEntity.input}) with a {@link McaiInput} whose tick() sets
 * the player's {@code playerInput}/{@code movementVector} from the policy action.
 * This is the meteor/baritone pattern: in 1.21.11 the move vector and the
 * ServerboundPlayerInputPacket both come from that object, so KeyBindings alone
 * never move the player. The original input is restored on stop.
 *
 * Look is applied via setYaw/setPitch. Mining goes through
 * ClientPlayerInteractionManager so the client emits correctly-timed
 * START/STOP_DESTROY_BLOCK packets.
 */
public final class AgentRunner {
    private static final Logger LOG = LoggerFactory.getLogger("mcai-agent");
    private static final int LOG_EVERY_TICKS = 10;
    private static final int OAK_LOG_BLOCK = 49; // gym id for minecraft:oak_log

    private final OnnxPolicy policy;
    private final ObservationBuilder obs;
    private final McaiInput mcaiInput = new McaiInput();
    private final SchemaRegistry registry;

    // Saved player input restored on stop. Non-null only while our input is installed.
    private Input prevInput;
    private long tickCounter;

    public AgentRunner(OnnxPolicy policy, SchemaRegistry registry) {
        this.policy = policy;
        this.registry = registry;
        this.obs = new ObservationBuilder(registry);
    }

    /** One agent step. */
    public void tick(MinecraftClient mc) throws Exception {
        ClientPlayerEntity player = mc.player;
        if (player == null || mc.world == null || mc.interactionManager == null) {
            return;
        }

        // Install our input object once the player exists; restored in stop().
        if (prevInput == null) {
            prevInput = player.input;
            player.input = mcaiInput;
        }

        obs.build(mc);
        float[] logits = policy.run(obs.voxel, obs.scalars, obs.invItemId, obs.invCount);
        ActionSpace action = ActionSpace.decode(logits);

        applyLook(player, action);
        mcaiInput.setDesired(action);
        applyMining(mc, action);

        logTelemetry(mc, player, action);
    }

    private void applyLook(ClientPlayerEntity player, ActionSpace action) {
        player.setYaw(player.getYaw() + action.yawDelta);
        player.setPitch(MathHelper.clamp(player.getPitch() + action.pitchDelta, -90.0f, 90.0f));
    }

    private void applyMining(MinecraftClient mc, ActionSpace action) {
        ClientPlayerInteractionManager im = mc.interactionManager;
        if (!action.attack) {
            im.cancelBlockBreaking();
            return;
        }

        // Re-raycast the crosshair (block-only) at the player's interaction range to
        // find the target, mirroring the gym's pick(blockInteractionRange, ...).
        double reach = mc.player.getBlockInteractionRange();
        if (reach <= 0.0) {
            reach = ObservationBuilder.DEFAULT_BLOCK_REACH;
        }
        HitResult hit = mc.player.raycast(reach, 1.0f, false);
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

    // In-world telemetry: log one line every ~10 ticks describing what the agent
    // perceives and what it decided, so the runClient console shows the agent's
    // behaviour without a debugger.
    private void logTelemetry(MinecraftClient mc, ClientPlayerEntity player, ActionSpace action) {
        if (tickCounter++ % LOG_EVERY_TICKS != 0) {
            return;
        }

        int logVoxels = 0;
        for (long id : obs.voxel) {
            if (id == OAK_LOG_BLOCK) {
                logVoxels++;
            }
        }

        // target_* live in the scalars: distance/8 at [10], in_range at [11].
        boolean targetInRange = obs.scalars[11] != 0.0f;
        float targetDistance = obs.scalars[10] * 8.0f;
        int targetBlock = obs.targetBlockId;

        // Wood count = sum of inventory counts whose item id is a known log.
        int woodCount = 0;
        for (int i = 0; i < obs.invItemId.length; i++) {
            if (registry.isLogItem((int) obs.invItemId[i])) {
                woodCount += (int) obs.invCount[i];
            }
        }

        LOG.info(
            "[mcai] tick={} x={} z={} yaw={} pitch={} onGround={} hp={} logVoxels={} targetInRange={} targetBlock={} targetDist={} wood={} | act fwd={} strafe={} jump={} sprint={} yawD={} pitchD={} attack={}",
            tickCounter,
            String.format("%.2f", player.getX()),
            String.format("%.2f", player.getZ()),
            String.format("%.1f", player.getYaw()),
            String.format("%.1f", player.getPitch()),
            player.isOnGround(),
            String.format("%.1f", player.getHealth()),
            logVoxels,
            targetInRange,
            targetBlock,
            String.format("%.2f", targetDistance),
            woodCount,
            action.forward,
            action.strafe,
            action.jump,
            action.sprint,
            action.yawDelta,
            action.pitchDelta,
            action.attack
        );
    }

    /** Restore the player's input and abort any in-progress mining. */
    public void release(MinecraftClient mc) {
        mcaiInput.clear();
        if (mc.player != null && prevInput != null) {
            mc.player.input = prevInput;
        }
        prevInput = null;
        if (mc.interactionManager != null) {
            mc.interactionManager.cancelBlockBreaking();
        }
    }

    public void close() {
        policy.close();
    }
}
