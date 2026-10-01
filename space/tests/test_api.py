"""The API end to end over the real C++ runtime and the real reference models.

Skipped, with the reason, when the compiled bindings or the prepared models
aren't present (e.g. a checkout without a build). In the Docker image's build
stage both are, and these run before the image is assembled.
"""
import io
import time

import pytest
from PIL import Image

import server

try:
    import sys

    sys.path.insert(0, str(server.BINDINGS_DIR))
    import loomcore  # noqa: F401

    MISSING = None
except ImportError as exc:  # pragma: no cover - depends on the build
    MISSING = f"compiled _loomcore extension not importable: {exc}"
if MISSING is None and not (server.MODELS_DIR / "mobilenetv2.int8.onnx").exists():
    MISSING = f"reference models not prepared in {server.MODELS_DIR} (python scripts/prepare_all_models.py)"

pytestmark = pytest.mark.skipif(MISSING is not None, reason=MISSING or "")


@pytest.fixture(scope="module")
def api():
    from fastapi.testclient import TestClient

    live = server.LiveRuntime()
    live.start()  # synchronous here: load, warm both precisions, apply the default chain
    with TestClient(server.create_app(live, start=False)) as client:
        yield live, client


def test_health_and_graph(api):
    _, client = api
    h = client.get("/health").json()
    assert h["ok"] and h["ready"]
    assert h["models"] == {"mobilenet": ["FP32", "INT8"], "bert_tiny": ["FP32", "INT8"]}
    g = client.get("/graph").json()
    assert g["topoOrder"] == ["mobilenet", "bert_tiny"]
    assert g["edges"] == [{"from": "mobilenet", "to": "bert_tiny"}]
    assert [p["name"] for p in g["policies"]] == server.POLICY_NAMES
    assert g["activePolicies"] == server.DEFAULT_POLICIES
    assert g["scheduler"]["admissionControl"] and g["scheduler"]["deadlineCancellation"]
    assert len(g["samples"]) == 6


def test_run_a_sample(api):
    _, client = api
    r = client.post("/run", json={"sample": "lighthouse"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["label"] == "lighthouse" and body["confidence"] > 0.85
    assert len(body["top5"]) == 5
    # Confident enough: the confidence gate skips bert_tiny, and says why.
    assert body["skipped"] == ["bert_tiny"] and body["embedding"] is None
    assert any(p["policy"] == "ConfidenceGatePolicy" for d in body["decisions"] for p in d["policies"])
    bars = [e for e in body["trace"] if e["ph"] == "X"]
    assert [b["name"] for b in bars] == ["mobilenet"]
    assert bars[0]["tid"] in (1, 2) and bars[0]["dur"] > 0


def test_run_through_both_nodes_returns_an_embedding(api):
    _, client = api
    r = client.post("/run", json={"sample": "tabby", "policies": ["load-aware"]})
    body = r.json()
    assert body["status"] == "ok" and body["graphSwapMs"] is not None  # a new chain means a hot-swap
    assert len(body["embedding"]) == 16 and body["text"].startswith("a photo of a ")
    assert [n["id"] for n in body["nodes"]] == ["mobilenet", "bert_tiny"]
    bars = sorted((e for e in body["trace"] if e["ph"] == "X"), key=lambda e: e["ts"])
    assert bars[0]["ts"] + bars[0]["dur"] <= bars[1]["ts"]  # bert_tiny ran after mobilenet finished
    flows = [e for e in body["trace"] if e.get("cat") == "dag_edge"]
    assert {f["ph"] for f in flows} == {"s", "f"}
    # Same chain again: no swap.
    assert client.post("/run", json={"sample": "tabby", "policies": ["load-aware"]}).json()["graphSwapMs"] is None


def test_an_impossible_budget_is_rejected_by_admission_control(api):
    _, client = api
    body = client.post("/run", json={"sample": "fox", "timeBudgetMs": 1}).json()
    assert body["status"] == "rejected"
    assert "admission control" in body["error"]
    assert body["nodes"] == []
    assert any(e["name"] == "job_rejected" for e in body["trace"] if e["ph"] == "i")


def test_upload_png(api):
    _, client = api
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), (230, 230, 40)).save(buf, "PNG")
    r = client.post("/run", files={"image": ("y.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200 and r.json()["source"] == "upload" and r.json()["label"]


def test_load(api):
    _, client = api
    body = client.post("/load", json={"jobs": 12, "concurrency": 4}).json()
    assert body["submitted"] == 12
    assert body["completed"] + body["rejected"] + body["cancelled"] + body["shed"] + body["failed"] == 12
    assert body["failed"] == 0
    assert {n["id"] for n in body["nodes"]} >= {"mobilenet"}
    assert sum(b["count"] for b in body["batches"]) > 0
    assert sum(1 for e in body["trace"] if e["ph"] == "X") == sum(b["count"] for b in body["batches"])


def test_reload_under_load_loses_nothing(api):
    live, client = api
    body = client.post("/reload", json={"jobs": 16, "swaps": 2}).json()
    assert body["submitted"] == 16 and body["completed"] == 16 and body["lost"] == 0
    assert [s["index"] for s in body["swapTimings"]] == [1, 2]
    assert all(s["buildMs"] > 0 and s["atMs"] >= 0 for s in body["swapTimings"])


def test_one_heavy_run_at_a_time(api):
    live, client = api
    live.heavy.acquire()
    live.heavy_started = time.monotonic()
    try:
        r = client.post("/load", json={"jobs": 4, "concurrency": 2})
        assert r.status_code == 429 and r.json()["error"] == "busy" and r.json()["retryAfter"] >= 1
        assert client.post("/reload", json={"jobs": 8, "swaps": 1}).status_code == 429
    finally:
        live.heavy.release()


def test_bench_binary_runs_and_parses(api):
    exe = server.find_bench()
    if not exe:
        pytest.skip("loomcore_bench not built")
    live, _ = api
    result = server.run_bench(live.cpu)
    assert result["status"] == "ready"
    assert {(r["model"], r["precision"]) for r in result["rows"]} == {
        ("mobilenetv2", "FP32"), ("mobilenetv2", "INT8"), ("bert_tiny", "FP32"), ("bert_tiny", "INT8")}
