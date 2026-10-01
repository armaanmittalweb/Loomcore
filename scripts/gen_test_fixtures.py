#!/usr/bin/env python3
"""Generates tiny, hand-built ONNX graphs used only by the C++ unit tests
(tests/test_scheduler.cpp). They are NOT the reference MobileNetV2 /
BERT-tiny models used by the example pipeline and benchmark — see
scripts/fetch_mobilenet.py and scripts/export_bert_tiny.py for those.

Kept deliberately trivial (a couple of ops over a fixed [batch, 4] float32
tensor) and checked into assets/ so the unit test suite is hermetic: it
never needs network access or a multi-hundred-MB model download to run.
"""
import os

import onnx
from onnx import TensorProto, helper

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")


def make_identity_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 4])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "loomcore_test_identity", [x], [y])
    model = helper.make_model(graph, producer_name="loomcore-test-fixtures", opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_double_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 4])
    node = helper.make_node("Add", ["x", "x"], ["y"])
    graph = helper.make_graph([node], "loomcore_test_double", [x], [y])
    model = helper.make_model(graph, producer_name="loomcore-test-fixtures", opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_variable_width_identity_model(path: str) -> None:
    # Like test_identity.onnx, but the *non-batch* dimension is symbolic
    # too (["batch", "width"] rather than ["batch", 4]) — used only by
    # test_scheduler.cpp's PendingKey-shape test, which needs a real ONNX
    # model that legitimately accepts two different non-batch shapes so it
    # can prove the scheduler never offers them to concatBatch as
    # batch-mates (see docs/ARCHITECTURE.md "Scheduler": the
    # (node, precision, shape) batching key).
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", "width"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", "width"])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "loomcore_test_variable_width_identity", [x], [y])
    model = helper.make_model(graph, producer_name="loomcore-test-fixtures", opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    make_identity_model(os.path.join(OUT_DIR, "test_identity.onnx"))
    make_double_model(os.path.join(OUT_DIR, "test_double.onnx"))
    make_variable_width_identity_model(os.path.join(OUT_DIR, "test_variable_width_identity.onnx"))
    print(f"Wrote test fixtures to {os.path.abspath(OUT_DIR)}")
