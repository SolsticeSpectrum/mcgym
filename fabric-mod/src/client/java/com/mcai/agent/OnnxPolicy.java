package com.mcai.agent;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Wraps the trained policy ONNX graph.
 *
 * Inputs:  voxel int64[1,4913], scalars float32[1,18],
 *          inv_item_id int64[1,41], inv_count float32[1,41]
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
    public float[] run(long[] voxel, float[] scalars, long[] invItemId, float[] invCount) throws OrtException {
        OnnxTensor tVoxel = OnnxTensor.createTensor(env, LongBuffer.wrap(voxel), new long[]{1, VOXEL});
        OnnxTensor tScalars = OnnxTensor.createTensor(env, FloatBuffer.wrap(scalars), new long[]{1, SCALARS});
        OnnxTensor tInvId = OnnxTensor.createTensor(env, LongBuffer.wrap(invItemId), new long[]{1, INV});
        OnnxTensor tInvCount = OnnxTensor.createTensor(env, FloatBuffer.wrap(invCount), new long[]{1, INV});

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("voxel", tVoxel);
        inputs.put("scalars", tScalars);
        inputs.put("inv_item_id", tInvId);
        inputs.put("inv_count", tInvCount);

        try (OrtSession.Result result = session.run(inputs)) {
            float[][] logits = (float[][]) result.get(0).getValue();
            return logits[0];
        } finally {
            tVoxel.close();
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
