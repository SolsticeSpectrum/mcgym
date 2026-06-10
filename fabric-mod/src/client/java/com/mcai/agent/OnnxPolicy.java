package com.mcai.agent;

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
 * Wraps the trained policy ONNX graph.
 *
 * Inputs:  voxel int32[1,4913], voxel_far int32[1,4913], target_block int32[1],
 *          scalars float32[1,18], inv_item_id int32[1,41], inv_count float32[1,41]
 *          (id grids are int32 in the schema; the trainer keeps them int32 end-to-end)
 * Outputs: logits float32[1,22], value float32[1]
 */
public final class OnnxPolicy implements AutoCloseable {
    public static final int VOXEL = 4913;
    public static final int SCALARS = 18;
    public static final int INV = 41;
    public static final int LOGITS = 22;

    private final OrtEnvironment env;
    private final OrtSession session;

    public OnnxPolicy(Path onnxFile) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        this.session = env.createSession(onnxFile.toString(), opts);
    }

    /** Run a single observation, returning the 22 raw logits. */
    public float[] run(int[] voxel, int[] voxelFar, int targetBlock, float[] scalars, int[] invItemId, float[] invCount) throws OrtException {
        OnnxTensor tVoxel = OnnxTensor.createTensor(env, IntBuffer.wrap(voxel), new long[]{1, VOXEL});
        OnnxTensor tVoxelFar = OnnxTensor.createTensor(env, IntBuffer.wrap(voxelFar), new long[]{1, VOXEL});
        OnnxTensor tTarget = OnnxTensor.createTensor(env, IntBuffer.wrap(new int[]{targetBlock}), new long[]{1});
        OnnxTensor tScalars = OnnxTensor.createTensor(env, FloatBuffer.wrap(scalars), new long[]{1, SCALARS});
        OnnxTensor tInvId = OnnxTensor.createTensor(env, IntBuffer.wrap(invItemId), new long[]{1, INV});
        OnnxTensor tInvCount = OnnxTensor.createTensor(env, FloatBuffer.wrap(invCount), new long[]{1, INV});

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("voxel", tVoxel);
        inputs.put("voxel_far", tVoxelFar);
        inputs.put("target_block", tTarget);
        inputs.put("scalars", tScalars);
        inputs.put("inv_item_id", tInvId);
        inputs.put("inv_count", tInvCount);

        try (OrtSession.Result result = session.run(inputs)) {
            float[][] logits = (float[][]) result.get(0).getValue();
            return logits[0];
        } finally {
            tVoxel.close();
            tVoxelFar.close();
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
        } catch (OrtException ignored) {
        }
    }
}
