package com.mcai.agent;

import net.minecraft.client.input.Input;
import net.minecraft.util.PlayerInput;
import net.minecraft.util.math.Vec2f;

/**
 * Client input driven by the policy instead of the keyboard.
 *
 * In 1.21.11 the local player's movement is computed from
 * {@code ClientPlayerEntity.input} (a {@link net.minecraft.client.input.Input},
 * Mojmap {@code ClientInput}): each tick its {@code playerInput}
 * ({@link PlayerInput}, Mojmap {@code Input} record) drives both
 * {@code getMovementInput()} (the move vector vanilla physics consumes) and the
 * {@code ServerboundPlayerInputPacket} sent to the server. KeyBindings only feed
 * the vanilla {@code KeyboardInput.tick()}; replacing the whole input object is
 * the meteor/baritone-proven way to actuate movement (HighwayBuilder swaps
 * {@code mc.player.input}; baritone's PlayerMovementInput overrides tick()).
 *
 * This subclass overrides {@code tick()} to set {@code playerInput} and
 * {@code movementVector} from the policy's desired action, ignoring the keyboard,
 * so both vanilla movement and the input packet use OUR values.
 */
public final class McaiInput extends Input {
    /** The action the agent wants applied on the next tick. */
    private volatile PlayerInput desired = PlayerInput.DEFAULT;

    /** Set the desired input from a decoded policy action (called each agent tick). */
    public void setDesired(ActionSpace action) {
        boolean forward = action.forward > 0.5f;
        boolean backward = action.forward < -0.5f;
        // Gym/strafe convention: strafe -1 = left, +1 = right.
        boolean left = action.strafe < -0.5f;
        boolean right = action.strafe > 0.5f;
        // PlayerInput order: forward, backward, left, right, jump, sneak, sprint.
        this.desired = new PlayerInput(forward, backward, left, right, action.jump, false, action.sprint);
    }

    /** Release all movement (used when stopping). */
    public void clear() {
        this.desired = PlayerInput.DEFAULT;
    }

    @Override
    public void tick() {
        // Mirror KeyboardInput.tick(): set playerInput (keyPresses) AND movementVector
        // (the move vector getMovementInput() returns), but from OUR action, not the keyboard.
        this.playerInput = this.desired;
        float forwardImpulse = impulse(this.desired.forward(), this.desired.backward());
        float leftImpulse = impulse(this.desired.left(), this.desired.right());
        this.movementVector = new Vec2f(leftImpulse, forwardImpulse).normalize();
    }

    private static float impulse(boolean positive, boolean negative) {
        if (positive == negative) {
            return 0.0f;
        }
        return positive ? 1.0f : -1.0f;
    }
}
