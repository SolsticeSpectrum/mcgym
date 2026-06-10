package com.mcgym.agent;

import com.mcgym.policy.Actions;

import net.minecraft.util.PlayerInput;
import net.minecraft.util.math.Vec2f;

public final class Input extends net.minecraft.client.input.Input {

    private volatile PlayerInput desired = PlayerInput.DEFAULT;

    public void set(Actions act) {
        boolean forward  = act.forward > 0.5f;
        boolean backward = act.forward < -0.5f;

        // strafe -1 is left, +1 is right
        boolean left  = act.strafe < -0.5f;
        boolean right = act.strafe > 0.5f;
        this.desired  = new PlayerInput(forward, backward, left, right, act.jump, false, act.sprint);
    }

    public void clear() {
        this.desired = PlayerInput.DEFAULT;
    }

    @Override
    public void tick() {
        // mirror KeyboardInput.tick() but from our action not the keyboard
        this.playerInput    = this.desired;
        float forward       = impulse(this.desired.forward(), this.desired.backward());
        float left          = impulse(this.desired.left(), this.desired.right());
        
        this.movementVector = new Vec2f(left, forward).normalize();
    }

    private static float impulse(boolean pos, boolean neg) {
        if (pos == neg) return 0.0f;
        return pos ? 1.0f : -1.0f;
    }
}
