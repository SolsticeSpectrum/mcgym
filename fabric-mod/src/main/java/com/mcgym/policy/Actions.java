package com.mcgym.policy;

// the trainers multi discrete action space, 22 logits split into 7 heads by BINS,
// argmax per head gives [forward, strafe, jump, sprint, yaw, pitch, attack]
public final class Actions {

    public static final int[] BINS = {3, 3, 2, 2, 5, 5, 2};

    private static final float[] FORWARD = {-1.0f, 0.0f, 1.0f};
    private static final float[] STRAFE  = {-1.0f, 0.0f, 1.0f};
    private static final float[] CAMERA  = {-10.0f, -3.0f, 0.0f, 3.0f, 10.0f};

    public float   forward;
    public float   strafe;
    public boolean jump;
    public boolean sprint;
    public float   yaw;
    public float   pitch;
    public boolean attack;

    public static Actions decode(float[] logits) {
        int[] idx = new int[BINS.length];
        int off = 0;
        for (int head = 0; head < BINS.length; head++) {
            idx[head] = argmax(logits, off, BINS[head]);
            off += BINS[head];
        }

        Actions a = new Actions();
        a.forward = FORWARD[idx[0]];
        a.strafe  = STRAFE[idx[1]];
        a.jump    = idx[2] != 0;
        a.sprint  = idx[3] != 0;
        a.yaw     = CAMERA[idx[4]];
        a.pitch   = CAMERA[idx[5]];
        a.attack  = idx[6] != 0;

        return a;
    }

    private static int argmax(float[] logits, int off, int len) {
        int best = 0;
        float val = logits[off];
        for (int i = 1; i < len; i++) {
            if (logits[off + i] > val) {
                val = logits[off + i];
                best = i;
            }
        }
        
        return best;
    }
}
