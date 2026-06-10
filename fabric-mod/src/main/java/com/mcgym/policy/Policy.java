package com.mcgym.policy;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * The trained policy onnx graph.
 *
 * Inputs are int32 like the schema, voxel[1,4913] voxel_far[1,4913] target_block[1]
 * scalars f32[1,18] inv_item_id[1,41] inv_count f32[1,41], outputs logits f32[1,22] value f32[1].
 */
public final class Policy implements AutoCloseable {

    public static final int VOXEL   = 4913;
    public static final int SCALARS = 18;
    public static final int INV     = 41;

    private final OrtEnvironment env;
    private final OrtSession session;

    public Policy(Path onnx) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(onnx.toString(), new OrtSession.SessionOptions());
    }

    /** Run one observation, returns the 22 raw logits. */
    public float[] run(int[] voxel, int[] voxelFar, int target, float[] scalars, int[] invId, float[] invCount) throws OrtException {
        OnnxTensor tVoxel    = OnnxTensor.createTensor(env, IntBuffer.wrap(voxel),             new long[]{1, VOXEL});
        OnnxTensor tFar      = OnnxTensor.createTensor(env, IntBuffer.wrap(voxelFar),          new long[]{1, VOXEL});
        OnnxTensor tTarget   = OnnxTensor.createTensor(env, IntBuffer.wrap(new int[]{target}), new long[]{1});
        OnnxTensor tScalars  = OnnxTensor.createTensor(env, FloatBuffer.wrap(scalars),         new long[]{1, SCALARS});
        OnnxTensor tInvId    = OnnxTensor.createTensor(env, IntBuffer.wrap(invId),             new long[]{1, INV});
        OnnxTensor tInvCount = OnnxTensor.createTensor(env, FloatBuffer.wrap(invCount),        new long[]{1, INV});

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("voxel",        tVoxel);
        inputs.put("voxel_far",    tFar);
        inputs.put("target_block", tTarget);
        inputs.put("scalars",      tScalars);
        inputs.put("inv_item_id",  tInvId);
        inputs.put("inv_count",    tInvCount);

        try (OrtSession.Result result = session.run(inputs)) {
            float[][] logits = (float[][]) result.get(0).getValue();
            return logits[0];
        } finally {
            tVoxel.close();
            tFar.close();
            tTarget.close();
            tScalars.close();
            tInvId.close();
            tInvCount.close();
        }
    }

    @Override
    public void close() {
        try {
            session.close();
        } catch (OrtException ignored) {}
    }
}
