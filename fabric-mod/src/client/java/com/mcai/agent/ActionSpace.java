package com.mcai.agent;

/**
 * Port of the trainer's multi-discrete action space (mcai_train.models.action_space).
 *
 * The 22 policy logits split into 7 categorical heads by BINS = {3,3,2,2,5,5,2}.
 * Argmax per head yields indices [forward, strafe, jump, sprint, yaw, pitch, attack],
 * which map to gym action values via the per-head value tables below.
 */
public final class ActionSpace {
    public static final int[] BINS = {3, 3, 2, 2, 5, 5, 2};

    private static final float[] FORWARD_VALUES = {-1.0f, 0.0f, 1.0f};
    private static final float[] STRAFE_VALUES = {-1.0f, 0.0f, 1.0f};
    private static final int[] JUMP_VALUES = {0, 1};
    private static final int[] SPRINT_VALUES = {0, 1};
    private static final float[] YAW_VALUES = {-10.0f, -3.0f, 0.0f, 3.0f, 10.0f};
    private static final float[] PITCH_VALUES = {-10.0f, -3.0f, 0.0f, 3.0f, 10.0f};
    private static final int[] ATTACK_VALUES = {0, 1};

    public float forward;
    public float strafe;
    public boolean jump;
    public boolean sprint;
    public float yawDelta;
    public float pitchDelta;
    public boolean attack;

    /** Decode the 22 raw logits into a concrete action via per-head argmax. */
    public static ActionSpace decode(float[] logits) {
        int[] idx = new int[BINS.length];
        int offset = 0;
        for (int head = 0; head < BINS.length; head++) {
            idx[head] = argmax(logits, offset, BINS[head]);
            offset += BINS[head];
        }
        ActionSpace a = new ActionSpace();
        a.forward = FORWARD_VALUES[idx[0]];
        a.strafe = STRAFE_VALUES[idx[1]];
        a.jump = JUMP_VALUES[idx[2]] != 0;
        a.sprint = SPRINT_VALUES[idx[3]] != 0;
        a.yawDelta = YAW_VALUES[idx[4]];
        a.pitchDelta = PITCH_VALUES[idx[5]];
        a.attack = ATTACK_VALUES[idx[6]] != 0;
        return a;
    }

    private static int argmax(float[] logits, int offset, int len) {
        int best = 0;
        float bestVal = logits[offset];
        for (int i = 1; i < len; i++) {
            float v = logits[offset + i];
            if (v > bestVal) {
                bestVal = v;
                best = i;
            }
        }
        return best;
    }
}
