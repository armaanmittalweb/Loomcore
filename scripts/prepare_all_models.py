#!/usr/bin/env python3
"""One-shot pipeline that produces everything under models/ and the model-
derived files under assets/: downloads MobileNetV2 + ImageNet labels,
relaxes MobileNetV2 to a dynamic batch axis, exports bert_tiny to ONNX
with its vocab, quantizes both to INT8, and regenerates the tiny fixture
models the unit tests use.

Requires: pip install -r scripts/requirements.txt (onnx, onnxruntime,
onnxscript, torch, transformers, huggingface_hub — already satisfied in
any environment that ran `pip install -r requirements.txt`).
"""
import subprocess
import sys
from pathlib import Path

STEPS = [
    ["fetch_mobilenet.py"],
    ["patch_mobilenet_dynamic_batch.py"],
    ["export_bert_tiny.py"],
    ["quantize_mobilenet.py"],
    ["quantize_bert_tiny.py"],
    ["gen_test_fixtures.py"],
    ["make_sample_jpeg.py"],
]

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    for step in STEPS:
        script = script_dir / step[0]
        print(f"\n=== running {step[0]} ===")
        subprocess.run([sys.executable, str(script)] + step[1:], check=True, cwd=script_dir.parent)
    print("\nAll models prepared under models/ and assets/.")
