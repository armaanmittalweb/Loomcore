#!/usr/bin/env python3
"""Static (QDQ) INT8 quantization of MobileNetV2 — the "quantized inference
path" milestone. A CNN's cost is dominated by convolutions, which is
exactly what static quantization (calibrate real activation ranges, then
fold them into QuantizeLinear/DequantizeLinear nodes around every weight)
targets; dynamic quantization (used instead for bert_tiny, see
quantize_bert_tiny.py) mainly helps MatMul-heavy transformer workloads and
would do much less for a conv-heavy graph like this one.

Calibration uses synthetic random activations rather than a real image
dataset: the point of this project is the orchestration layer, not model
accuracy, and Loomcore's own benchmark only measures latency, never
predictions — see benchmarks/latency_bench.cpp and docs/BENCHMARKS.md for
that caveat stated plainly again next to the numbers it affects.
"""
import os

import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT, "models")


class RandomCalibrationDataReader(CalibrationDataReader):
    """Feeds `n` random [batch, 3, 224, 224] float32 tensors, matching
    MobileNetV2's real input distribution closely enough (roughly [0, 1]
    after normalization) to exercise every activation's quantization
    range without requiring a real calibration image set."""

    def __init__(self, input_name: str, shape, n: int = 32):
        self.input_name = input_name
        self.shape = shape
        self.n = n
        self.i = 0

    def get_next(self):
        if self.i >= self.n:
            return None
        self.i += 1
        data = np.random.normal(loc=0.0, scale=0.25, size=self.shape).astype(np.float32)
        return {self.input_name: data}


def main() -> None:
    src = os.path.join(MODELS_DIR, "mobilenetv2.onnx")
    preprocessed = os.path.join(MODELS_DIR, "mobilenetv2.preprocessed.onnx")
    quant_pre_process(src, preprocessed)

    dst = os.path.join(MODELS_DIR, "mobilenetv2.int8.onnx")
    reader = RandomCalibrationDataReader("data", (1, 3, 224, 224), n=32)
    quantize_static(
        preprocessed,
        dst,
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
