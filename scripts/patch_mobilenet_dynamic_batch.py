#!/usr/bin/env python3
"""The published ONNX Model Zoo MobileNetV2 export declares a *static*
batch size of 1 on its "data" input/output, even though every op in the
graph (convolutions, batchnorm, global average pool, and a final Reshape
that uses ONNX's "0 = copy this dim from the input" convention) is
actually batch-size agnostic. ONNX Runtime enforces declared static
dimensions strictly, so as downloaded the model rejects any batch size
other than 1 — which would make it impossible to demonstrate Loomcore's
cross-job dynamic batching on it. This script relaxes the batch dimension
on the input/output to a symbolic dim so real batching works, without
touching any weights or computation.
"""
import sys

import onnx
import onnxruntime as ort


def make_batch_dynamic(path: str) -> None:
    model = onnx.load(path)
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    onnx.checker.check_model(model)
    onnx.save(model, path)


def verify(path: str) -> None:
    import numpy as np

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for batch in (1, 3):
        x = np.random.rand(batch, 3, 224, 224).astype(np.float32)
        out = sess.run(None, {"data": x})
        assert out[0].shape == (batch, 1000), out[0].shape
    print(f"OK: {path} now accepts dynamic batch sizes")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "models/mobilenetv2.onnx"
    make_batch_dynamic(path)
    verify(path)
