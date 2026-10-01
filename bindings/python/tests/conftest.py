"""Shared fixtures for the Python binding tests.

These are hermetic: they only use the tiny fixture graphs committed under
assets/ (test_identity.onnx, test_double.onnx), never the downloaded
reference models, so they run anywhere the C++ build ran. If the compiled
_loomcore extension isn't built (or can't be found), every test here is
skipped with the import error as the reason rather than failing.
"""
import json
import locale
import os
import sys

import pytest

if sys.platform == "win32":
    # A native Windows build made with llvm-mingw (libc++) caches the C
    # runtime's ctype table when std::locale::classic() is first used, and
    # that pointer can dangle once the CRT locale Python set at startup is
    # replaced, which breaks istream whitespace splitting inside the C++
    # tokenizer. Pinning the C locale avoids it; MSVC and Linux builds are
    # unaffected either way.
    locale.setlocale(locale.LC_CTYPE, "C")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ASSETS = os.path.join(REPO_ROOT, "assets")
sys.path.insert(0, os.path.join(REPO_ROOT, "bindings", "python"))

try:
    import loomcore  # noqa: E402

    IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on the build
    loomcore = None
    IMPORT_ERROR = exc


def pytest_collection_modifyitems(config, items):
    if IMPORT_ERROR is None:
        return
    skip = pytest.mark.skip(reason=f"compiled _loomcore extension not importable: {IMPORT_ERROR}")
    for item in items:
        item.add_marker(skip)


def asset(name: str) -> str:
    return os.path.join(ASSETS, name).replace("\\", "/")


@pytest.fixture
def write_config(tmp_path):
    """Writes a graph config and returns its path. `nodes` uses the same
    schema as examples/graph_config.json."""

    def _write(nodes, name="graph.json"):
        path = tmp_path / name
        path.write_text(json.dumps({"nodes": nodes}), encoding="utf-8")
        return str(path)

    return _write


def identity_node(node_id="a", **extra):
    node = {
        "id": node_id,
        "backend": "CPU",
        "max_batch_size": 1,
        "depends_on": [],
        "variants": [
            {"precision": "FP32", "model_path": asset("test_identity.onnx")},
            {"precision": "INT8", "model_path": asset("test_identity.onnx")},
        ],
    }
    node.update(extra)
    return node


def double_node(node_id="b", depends_on=("a",), **extra):
    node = {
        "id": node_id,
        "backend": "GPU_SIM",
        "max_batch_size": 1,
        "depends_on": list(depends_on),
        "variants": [
            {"precision": "FP32", "model_path": asset("test_double.onnx")},
            {"precision": "INT8", "model_path": asset("test_double.onnx")},
        ],
    }
    node.update(extra)
    return node


def quiet_options(**scheduler_flags):
    """RuntimeOptions that keep the JSON-lines stream off stdout (tests read
    Runtime.recent_logs instead) with the given SchedulerConfig flags set."""
    options = loomcore.RuntimeOptions()
    options.log_to_stdout = False
    for key, value in scheduler_flags.items():
        setattr(options.scheduler, key, value)
    return options
