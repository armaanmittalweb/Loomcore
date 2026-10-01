"""Tests for the Python surface added for the live Space (space/server.py):
hot-swap, scheduler flags, typed errors, the remaining router policies, graph
introspection, the C++ tokenizer and the Perfetto export. Each drives the
real C++ scheduler over the committed fixture graphs.
"""
import gc
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from conftest import REPO_ROOT, double_node, identity_node, loomcore, quiet_options

X = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)


def pass_through(name):
    def binder(graph_inputs, upstream):
        if name in graph_inputs:
            return [("x", graph_inputs[name])]
        return [("x", next(iter(upstream.values()))[0])]

    return binder


def test_scheduler_config_and_runtime_options_round_trip():
    cfg = loomcore.SchedulerConfig()
    assert cfg.enable_admission_control is False  # every deadline feature ships off by default
    assert cfg.enable_deadline_cancellation is False
    assert cfg.enable_precision_planning is False
    assert cfg.use_edf_scoring is False
    cfg.enable_admission_control = True
    cfg.deadline_reaper_poll_ms = 2.5
    assert cfg.enable_admission_control is True
    assert cfg.deadline_reaper_poll_ms == 2.5

    options = loomcore.RuntimeOptions()
    options.scheduler.enable_deadline_cancellation = True
    options.log_to_stdout = False
    options.intra_op_threads_per_variant = 1
    assert options.scheduler.enable_deadline_cancellation is True
    assert options.log_file == ""


def test_error_hierarchy():
    assert issubclass(loomcore.LoomcoreError, RuntimeError)
    assert issubclass(loomcore.JobRejectedError, loomcore.LoomcoreError)
    assert issubclass(loomcore.DeadlineExceededError, loomcore.LoomcoreError)


def test_graph_introspection(write_config):
    path = write_config([identity_node("a", priority=5), double_node("b", quality_weight=2.0)])
    rt = loomcore.Runtime()
    rt.load_graph(path, binders={"a": pass_through("x"), "b": pass_through("x")}, options=quiet_options())
    g = rt.graph()
    assert g["topo_order"] == ["a", "b"]
    assert g["sinks"] == ["b"]
    by_id = {n["id"]: n for n in g["nodes"]}
    assert by_id["a"]["backend"] == "CPU" and by_id["a"]["priority"] == 5
    assert by_id["b"]["backend"] == "GPU_SIM" and by_id["b"]["depends_on"] == ["a"]
    assert by_id["b"]["quality_weight"] == 2.0
    assert [v["precision"] for v in by_id["a"]["variants"]] == ["FP32", "INT8"]
    assert by_id["a"]["confidence_source"] is None


def test_reload_graph_swaps_the_graph(write_config):
    identity = write_config([identity_node("a")], "identity.json")
    doubled = write_config([{**double_node("a", depends_on=()), "backend": "CPU"}], "double.json")
    rt = loomcore.Runtime()
    binders = {"a": pass_through("x")}
    rt.load_graph(identity, binders=binders, options=quiet_options())
    assert rt.run({"x": X})["a"][0].tolist() == X.tolist()
    rt.reload_graph(doubled, binders=binders, options=quiet_options())
    assert rt.run({"x": X})["a"][0].tolist() == (X * 2).tolist()


def test_reload_graph_requires_load_graph_first(write_config):
    rt = loomcore.Runtime()
    with pytest.raises(loomcore.LoomcoreError):
        rt.reload_graph(write_config([identity_node("a")]), binders={"a": pass_through("x")})


def test_reload_graph_under_concurrent_load_loses_no_job(write_config):
    """The Python twin of tests/test_runtime_reload.cpp: producer threads keep
    calling run() (the GIL is released while C++ runs) while the main thread
    hot-swaps between two graphs. Every job must succeed, and each result
    must be wholly one graph's or the other's, never a mix. Retired snapshots
    (and the Python binders they hold) are destroyed on the runtime's own
    teardown thread, which never holds the GIL; this also exercises that
    path many times over."""
    identity = write_config([identity_node("a")], "identity.json")
    doubled = write_config([{**double_node("a", depends_on=()), "backend": "CPU"}], "double.json")
    rt = loomcore.Runtime()
    rt.load_graph(identity, binders={"a": pass_through("x")}, options=quiet_options())

    stop = threading.Event()
    counts = {"submitted": 0, "ok": 0, "bad": 0}
    lock = threading.Lock()

    def producer():
        while not stop.is_set():
            with lock:
                counts["submitted"] += 1
            try:
                out = rt.run({"x": X})["a"][0].tolist()
                good = out in (X.tolist(), (X * 2).tolist())
            except Exception:  # noqa: BLE001 - any failure is a lost job
                good = False
            with lock:
                counts["ok" if good else "bad"] += 1

    threads = [threading.Thread(target=producer) for _ in range(4)]
    for t in threads:
        t.start()
    for i in range(12):
        time.sleep(0.02)
        # A fresh binder closure each time, so every retired snapshot drops
        # the last reference to a Python callable when it is torn down.
        rt.reload_graph(doubled if i % 2 == 0 else identity, binders={"a": pass_through("x")},
                        options=quiet_options())
    time.sleep(0.05)
    stop.set()
    for t in threads:
        t.join()
    gc.collect()

    assert counts["bad"] == 0
    assert counts["ok"] == counts["submitted"] > 0


def test_deadline_cancellation_raises_deadline_exceeded(write_config):
    path = write_config([identity_node("a")])

    def slow_binder(graph_inputs, _upstream):
        time.sleep(0.08)  # the job's whole 5 ms budget passes while this binder runs
        return [("x", graph_inputs["x"])]

    rt = loomcore.Runtime()
    rt.load_graph(path, binders={"a": slow_binder},
                  options=quiet_options(enable_deadline_cancellation=True, deadline_reaper_poll_ms=1.0))
    with pytest.raises(loomcore.DeadlineExceededError):
        rt.run({"x": X}, time_budget_ms=5.0)
    # Unbudgeted jobs are never reaped.
    assert rt.run({"x": X})["a"][0].tolist() == X.tolist()


def test_admission_control_rejects_an_undeliverable_budget(write_config):
    path = write_config([identity_node("a")])
    router = loomcore.CompositeRouter()
    router.add(loomcore.LatencyBudgetPolicy(1e9))  # any budgeted job runs INT8, so both precisions get measured
    rt = loomcore.Runtime()
    rt.load_graph(path, binders={"a": pass_through("x")}, router=router,
                  options=quiet_options(enable_admission_control=True))
    rt.run({"x": X}, time_budget_ms=1000.0)  # cold registry: admitted (nothing to judge by), runs INT8
    rt.run({"x": X})  # unbudgeted: FP32
    with pytest.raises(loomcore.JobRejectedError):
        rt.run({"x": X}, time_budget_ms=1e-4)
    events = [json.loads(line)["event"] for line in rt.recent_logs(20)]
    assert "job_rejected" in events


def test_bulkhead_policy_sheds(write_config):
    path = write_config([identity_node("a")])
    router = loomcore.CompositeRouter()
    router.add(loomcore.BulkheadPolicy(0))  # a cap of zero sheds every request
    rt = loomcore.Runtime()
    rt.load_graph(path, binders={"a": pass_through("x")}, router=router, options=quiet_options())
    result = rt.run({"x": X})
    assert result.get("a", []) == []
    messages = [json.loads(line).get("message", "") for line in rt.recent_logs(20)]
    assert any("BulkheadPolicy" in m and "shedding" in m for m in messages)


def test_confidence_gate_policy_skips_the_downstream_node(write_config):
    path = write_config([identity_node("a"), double_node("b", confidence_source="a")])
    router = loomcore.CompositeRouter()
    router.add(loomcore.ConfidenceGatePolicy(0.5))
    rt = loomcore.Runtime()
    seen = []

    def extractor(upstream):
        seen.append(upstream[0].tolist())
        return 0.9

    rt.load_graph(path, binders={"a": pass_through("x"), "b": pass_through("x")},
                  confidence_extractors={"b": extractor}, router=router, options=quiet_options())
    result = rt.run({"x": X})
    assert result.get("b", []) == []  # skipped: 0.9 >= 0.5
    assert seen == [X.tolist()]  # the extractor saw node a's real output
    messages = [json.loads(line).get("message", "") for line in rt.recent_logs(20)]
    assert any("ConfidenceGatePolicy" in m for m in messages)


def test_planned_precision_and_circuit_breaker_policies_chain(write_config):
    path = write_config([identity_node("a"), double_node("b")])
    router = loomcore.CompositeRouter()
    router.add(loomcore.CircuitBreakerPolicy(0.5, 4, 1000.0))
    router.add(loomcore.PlannedPrecisionPolicy())
    rt = loomcore.Runtime()
    rt.load_graph(path, binders={"a": pass_through("x"), "b": pass_through("x")}, router=router,
                  options=quiet_options(enable_precision_planning=True))
    for _ in range(3):  # healthy runs: the breaker has no opinion, the plan fits
        assert rt.run({"x": X}, time_budget_ms=1000.0)["b"][0].tolist() == (X * 2).tolist()


def test_wordpiece_tokenizer(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "photo", "of", "golden", "retriever"]) + "\n",
                     encoding="utf-8")
    tok = loomcore.WordPieceTokenizer(str(vocab), 12)
    assert tok.vocab_size == 9
    enc = tok.encode("a photo of a golden retriever")
    assert set(enc) == {"input_ids", "attention_mask", "token_type_ids"}
    assert enc["input_ids"].dtype == np.int64 and enc["input_ids"].shape == (1, 12)
    assert enc["input_ids"][0, :8].tolist() == [2, 4, 5, 6, 4, 7, 8, 3]
    assert enc["attention_mask"][0].tolist() == [1] * 8 + [0] * 4


def test_export_perfetto_trace_from_a_real_run(write_config, tmp_path):
    path = write_config([identity_node("a"), double_node("b")])
    rt = loomcore.Runtime()
    rt.load_graph(path, binders={"a": pass_through("x"), "b": pass_through("x")}, options=quiet_options())
    start = time.time()
    rt.run({"x": X})
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(start)) + ".%06dZ" % int((start % 1) * 1e6)
    lines = [line for line in rt.recent_logs(200) if json.loads(line)["ts"] >= since]
    jsonl = tmp_path / "run.jsonl"
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "trace.json"
    written = loomcore.export_perfetto_trace(str(jsonl), str(out))
    assert written > 0
    events = json.loads(out.read_text(encoding="utf-8"))["traceEvents"]
    bars = {e["name"]: e for e in events if e["ph"] == "X"}
    assert set(bars) >= {"a", "b"}
    assert bars["a"]["tid"] == 1 and bars["b"]["tid"] == 2  # CPU lane, GPU_SIM lane
    assert bars["a"]["ts"] + bars["a"]["dur"] <= bars["b"]["ts"]  # b ran after a finished
    flows = [e for e in events if e.get("cat") == "dag_edge"]
    assert [f["ph"] for f in flows] == ["s", "f"]
    assert flows[0]["tid"] == 1 and flows[1]["tid"] == 2


def test_export_perfetto_trace_missing_file_raises(tmp_path):
    with pytest.raises(loomcore.LoomcoreError):
        loomcore.export_perfetto_trace(str(tmp_path / "missing.jsonl"), str(tmp_path / "out.json"))


def test_finds_the_default_build_directory_without_env():
    """loomcore/__init__.py documents that a build/ directory in the repo
    root is found with no LOOMCORE_BUILD_DIR set. It used to compute the repo
    root one level short (bindings/), so only the env var ever worked."""
    if not os.path.isdir(os.path.join(REPO_ROOT, "build", "bin")):
        pytest.skip("no in-tree build/ directory (this build used a different directory)")
    env = {k: v for k, v in os.environ.items() if k != "LOOMCORE_BUILD_DIR"}
    code = "import sys; sys.path.insert(0, sys.argv[1]); import loomcore; print(loomcore.Runtime.__name__)"
    proc = subprocess.run([sys.executable, "-c", code, os.path.join(REPO_ROOT, "bindings", "python")],
                          cwd=os.path.expanduser("~"), env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "Runtime"
