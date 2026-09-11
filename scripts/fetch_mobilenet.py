#!/usr/bin/env python3
"""Downloads the ONNX Model Zoo's MobileNetV2 (opset 7, FP32) — one of the
2-3 small public ONNX models the project spec calls for — plus the
1000-class ImageNet label list used to turn its output into text for the
downstream BERT-tiny node.
"""
import json
import os
import urllib.request

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT, "models")
ASSETS_DIR = os.path.join(ROOT, "assets")

MOBILENET_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/classification/"
    "mobilenet/model/mobilenetv2-7.onnx"
)
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"


def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"already have {dest}")
        return
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    download(MOBILENET_URL, os.path.join(MODELS_DIR, "mobilenetv2.onnx"))

    labels_json = os.path.join(ASSETS_DIR, "imagenet_labels.json")
    download(LABELS_URL, labels_json)

    # The C++ example reads a plain newline-delimited list (line N == class
    # index N) rather than parsing JSON, to avoid pulling a JSON dependency
    # into a plain example program.
    labels_txt = os.path.join(ASSETS_DIR, "imagenet_labels.txt")
    with open(labels_json, "r", encoding="utf-8") as f:
        labels = json.load(f)
    with open(labels_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(labels) + "\n")
    print(f"wrote {len(labels)} labels to {labels_txt}")
