#!/usr/bin/env python3
"""Dynamic INT8 quantization of bert_tiny's MatMul/Gemm weights — the
standard technique for transformer encoders, where activation ranges
depend heavily on the actual input text and are quantized on the fly per
inference rather than pre-calibrated (contrast with MobileNetV2's static
quantization in quantize_mobilenet.py).
"""
import os

from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT, "models")


def main() -> None:
    src = os.path.join(MODELS_DIR, "bert_tiny.onnx")
    dst = os.path.join(MODELS_DIR, "bert_tiny.int8.onnx")
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
