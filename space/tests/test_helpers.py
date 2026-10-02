"""The server's pure helpers and its HTTP layer (guards, CORS, errors), with
no C++ build needed: the HTTP tests drive create_app() over a fake runtime."""
import io
import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import server


def png_bytes(size=(32, 24), color=(200, 30, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def jpeg_bytes(size=(40, 40)):
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 120, 200)).save(buf, "JPEG")
    return buf.getvalue()


# -- input validation -------------------------------------------------------


def test_parse_policies_defaults_and_chain_order():
    assert server.parse_policies(None) == server.DEFAULT_POLICIES
    assert "bulkhead" not in server.DEFAULT_POLICIES
    # Always returned in chain order, whatever order the client sent.
    assert server.parse_policies(["load-aware", "circuit-breaker"]) == ["circuit-breaker", "load-aware"]
    assert server.parse_policies("latency-budget, confidence-gate") == ["confidence-gate", "latency-budget"]
    assert server.parse_policies([]) == []
    with pytest.raises(server.ApiError) as exc:
        server.parse_policies(["teleport"])
    assert exc.value.status == 400 and exc.value.extra["known"] == server.POLICY_NAMES
    with pytest.raises(server.ApiError):
        server.parse_policies([1, 2])


def test_parse_budget_and_int_bounds():
    assert server.parse_budget(None) is None
    assert server.parse_budget("") is None
    assert server.parse_budget("40") == 40.0
    for bad in (0, -1, 10_001, "abc", float("nan"), float("inf")):
        with pytest.raises(server.ApiError):
            server.parse_budget(bad)
    assert server.parse_int(None, "jobs", 1, 64, default=16) == 16
    assert server.parse_int("8", "jobs", 1, 64) == 8
    for bad in (0, 65, 2.5, "x", True, None, [3]):
        with pytest.raises(server.ApiError):
            server.parse_int(bad, "jobs", 1, 64)


def test_image_validation_and_preprocessing():
    assert server.sniff_image(png_bytes()) == "png"
    assert server.sniff_image(jpeg_bytes()) == "jpeg"
    with pytest.raises(server.ApiError) as exc:
        server.sniff_image(b"GIF89a....")
    assert exc.value.status == 415
    with pytest.raises(server.ApiError) as exc:
        server.image_to_tensor(b"\x89PNG\r\n\x1a\n" + b"\0" * 64)  # right magic, broken body
    assert exc.value.status == 415
    with pytest.raises(server.ApiError) as exc:
        server.image_to_tensor(b"\xff\xd8\xff" + b"\0" * (server.MAX_UPLOAD_BYTES + 1))
    assert exc.value.status == 413

    t = server.image_to_tensor(png_bytes(color=(255, 255, 255)))
    assert t.shape == (1, 3, 224, 224) and t.dtype == np.float32 and t.flags["C_CONTIGUOUS"]
    # White, normalised with ImageNet mean/std, per channel.
    expected = (1.0 - server.IMAGENET_MEAN) / server.IMAGENET_STD
    assert np.allclose(t[0, :, 100, 100], expected, atol=1e-5)


# -- log reading ---------------------------------------------------------------


def log(event, ts, **fields):
    return json.dumps({"event": event, "ts": ts, **fields}, sort_keys=True)


LOG = [
    log("job_submitted", "2026-10-01T10:00:00.000000Z", job_id="job-1"),
    log("routing_decision", "2026-10-01T10:00:00.000100Z", job_id="job-1", node_id="mobilenet",
        message="CompositeRouter: LatencyBudgetPolicy: remaining budget 20ms < threshold 30ms; downgrading to INT8 "
                "for node 'mobilenet' | LoadAwareBackendPolicy: cpu_queue=2 gpu_sim_queue=0; routing to the "
                "shallower lane"),
    log("batch_flushed", "2026-10-01T10:00:00.012000Z", job_id="(2 jobs)", node_id="mobilenet", backend="GPU_SIM",
        precision="INT8", batch_size=2, latency_ms=10.0),
    log("node_completed", "2026-10-01T10:00:00.012010Z", job_id="job-1", node_id="mobilenet", backend="GPU_SIM",
        precision="INT8", latency_ms=10.0),
    log("node_completed", "2026-10-01T10:00:00.012020Z", job_id="job-2", node_id="mobilenet", backend="GPU_SIM",
        precision="INT8", latency_ms=10.0),
    log("routing_decision", "2026-10-01T10:00:00.012100Z", job_id="job-2", node_id="bert_tiny",
        message="CompositeRouter -> ConfidenceGatePolicy: upstream confidence 0.95 >= threshold 0.85; skipping "
                "node 'bert_tiny'"),
    log("node_skipped", "2026-10-01T10:00:00.012110Z", job_id="job-2", node_id="bert_tiny"),
    log("job_rejected", "2026-10-01T10:00:00.013000Z", message="PrecisionPlanner: critical path ..."),
    log("error", "2026-10-01T10:00:00.014000Z", job_id="job-3", message="job 'job-3' exceeded its time budget of 5ms"),
    log("error", "2026-10-01T10:00:00.015000Z", job_id="job-4", message="UpstreamShed: " + server.SHED_MESSAGE),
    "not json at all",
]


def test_parse_decision_splits_policies():
    merged = server.parse_decision("CompositeRouter: A: one reason | B: two: with a colon")
    assert merged == [{"policy": "A", "reason": "one reason"}, {"policy": "B", "reason": "two: with a colon"}]
    skip = server.parse_decision("CompositeRouter -> BulkheadPolicy: shedding")
    assert skip == [{"policy": "BulkheadPolicy", "reason": "shedding"}]


def test_summarise_counts_everything():
    s = server.summarise(LOG)
    mobilenet = s["nodes"][0]
    assert mobilenet["id"] == "mobilenet" and mobilenet["count"] == 2 and mobilenet["p50"] == 10.0
    assert mobilenet["precisions"] == {"INT8": 2} and mobilenet["lanes"] == {"GPU_SIM": 2}
    assert s["batches"] == [{"node": "mobilenet", "count": 1, "sizes": {"2": 1}, "meanSize": 2.0}]
    counts = {c["policy"]: c["count"] for c in s["policyCounts"]}
    assert counts == {"LatencyBudgetPolicy": 1, "LoadAwareBackendPolicy": 1, "ConfidenceGatePolicy": 1}
    assert s["skipped"] == {"bert_tiny": 1}
    assert (s["rejected"], s["cancelled"], s["shed"], s["errors"]) == (1, 1, 1, 0)


def test_trace_base_counts_batches_from_their_start():
    # The batch ended at 12.000 ms after 10 ms, so it started at 2.000 ms, which
    # is after job_submitted at 0: the base is job_submitted.
    assert server.trace_base_us(LOG) == server.iso_to_us("2026-10-01T10:00:00.000000Z")
    early = [log("batch_flushed", "2026-10-01T10:00:00.005000Z", latency_ms=8.0)] + LOG
    assert server.trace_base_us(early) == server.iso_to_us("2026-10-01T10:00:00.000000Z") - 3000


def test_iso_round_trip():
    assert server.iso_to_us("1970-01-01T00:00:01.000002Z") == 1_000_002
    now = server.iso_now()
    assert len(now) == 27 and now.endswith("Z")


# -- bench and CPU -------------------------------------------------------------

BENCH_STDOUT = """Loomcore FP32 vs INT8 latency benchmark  (warmup=20, iters=100)

model         prec    mean_ms     p50_ms      p95_ms      n
--------------------------------------------------------------------
mobilenetv2   FP32    8.193       8.172       8.419       100
mobilenetv2   INT8    8.734       8.779       9.118       100
  -> INT8 p50 speedup vs FP32: 0.93x
bert_tiny     FP32    0.281       0.278       0.295       100
bert_tiny     INT8    0.250       0.247       0.261       100
  -> INT8 p50 speedup vs FP32: 1.13x

Wrote benchmarks/results/latency_1.csv
"""


def test_parse_bench_output_reads_the_table():
    rows = server.parse_bench_output(BENCH_STDOUT)
    assert [(r["model"], r["precision"]) for r in rows] == [
        ("mobilenetv2", "FP32"), ("mobilenetv2", "INT8"), ("bert_tiny", "FP32"), ("bert_tiny", "INT8")]
    assert rows[1] == {"model": "mobilenetv2", "precision": "INT8", "mean": 8.734, "p50": 8.779, "p95": 9.118,
                       "n": 100}


def test_parse_cpuinfo_reads_model_and_int8_flags():
    text = ("processor\t: 0\nmodel name\t: Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz\n"
            "flags\t\t: fpu sse2 avx avx2 avx512f avx512_vnni\n\n"
            "processor\t: 1\nmodel name\t: Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz\nflags\t\t: fpu\n")
    info = server.parse_cpuinfo(text)
    assert info["arch"] == "x86_64"
    assert info["model"].startswith("Intel(R) Xeon(R) Platinum 8375C")
    assert info["logicalCpus"] == 2
    assert info["flags"] == {"avx2": True, "avx512f": True, "avx512_vnni": True, "avx_vnni": False,
                             "amx_int8": False}
    zen3 = server.parse_cpuinfo("model name : AMD Ryzen 9 6900HX\nflags : avx avx2 fma\n")
    assert zen3["flags"]["avx2"] and not zen3["flags"]["avx512f"] and not zen3["flags"]["avx512_vnni"]


ORACLE_A1_CPUINFO = """processor	: 0
BogoMIPS	: 50.00
Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp ssbs
CPU implementer	: 0x41
CPU architecture: 8
CPU variant	: 0x3
CPU part	: 0xd0c
CPU revision	: 1

processor	: 1
BogoMIPS	: 50.00
Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp ssbs
CPU implementer	: 0x41
CPU architecture: 8
CPU variant	: 0x3
CPU part	: 0xd0c
CPU revision	: 1
"""


def test_parse_cpuinfo_reads_an_aarch64_neoverse_n1():
    info = server.parse_cpuinfo(ORACLE_A1_CPUINFO)
    assert info["arch"] == "aarch64"
    assert info["model"] == "Arm Neoverse N1"
    assert info["logicalCpus"] == 2
    # N1 has the int8 dot product (SDOT/UDOT) but not i8mm, bf16 or SVE.
    assert info["flags"] == {"asimd": True, "asimddp": True, "i8mm": False, "bf16": False, "sve": False}
    named = server.parse_cpuinfo(ORACLE_A1_CPUINFO, "Ampere Altra")
    assert named["model"] == "Ampere Altra (Neoverse N1)"
    v2 = server.parse_cpuinfo("processor : 0\nFeatures : fp asimd asimddp i8mm bf16 sve sve2\nCPU implementer : 0x41\nCPU part : 0xd4f\n")
    assert v2["model"] == "Arm Neoverse V2" and v2["flags"]["i8mm"] and v2["flags"]["sve"]
    unknown = server.parse_cpuinfo("processor : 0\nFeatures : fp asimd\nCPU implementer : 0x51\nCPU part : 0x001\n")
    assert unknown["model"] == "Arm CPU part 0x001" and not unknown["flags"]["asimddp"]


def test_available_cpus_follows_the_allotment(monkeypatch):
    monkeypatch.setenv("LOOMCORE_CPUS", "4")
    assert server.available_cpus() == 4 and server.cpu_lane_threads() == 2
    monkeypatch.setenv("LOOMCORE_CPUS", "nope")
    assert server.available_cpus() == (os.cpu_count() or 2)


def test_run_bench_reports_unavailable_without_the_binary(monkeypatch):
    monkeypatch.setenv("LOOMCORE_BENCH", "/definitely/not/here")
    result = server.run_bench({"model": "x"})
    assert result["status"] == "unavailable"


# -- rate limiter ---------------------------------------------------------------


def test_rate_limiter_sliding_window():
    rl = server.RateLimiter()
    for i in range(3):
        assert rl.check("a", "post", 3, now=100.0 + i) == 0.0
    wait = rl.check("a", "post", 3, now=103.0)
    assert 56.9 < wait <= 57.0  # the oldest hit (t=100) leaves the window at t=160
    assert rl.check("b", "post", 3, now=103.0) == 0.0  # per client
    assert rl.check("a", "get", 3, now=103.0) == 0.0  # per bucket
    assert rl.check("a", "post", 3, now=160.5) == 0.0  # window slid


# -- HTTP layer over a fake runtime ---------------------------------------------


class FakeLive:
    ready = True

    def __init__(self):
        self.calls = []

    def health(self):
        return {"ok": True, "ready": True}

    def inputs_for(self, sample, upload):
        if upload is not None:
            return server.image_to_tensor(upload), "upload"
        if sample not in (None, "tabby"):
            raise server.ApiError(400, "bad_sample", f"unknown sample '{sample}'")
        return np.zeros((1, 3, 224, 224), np.float32), sample or "tabby"

    def run(self, tensor, source, budget, policies):
        self.calls.append(("run", source, budget, policies, tensor.shape))
        return {"status": "ok", "source": source}

    def load(self, *args):
        raise server.Busy(4.2)

    def reload(self, jobs, swaps):
        return {"jobs": jobs, "swaps": swaps, "lost": 0}


@pytest.fixture
def fake():
    live = FakeLive()
    return live, TestClient(server.create_app(live, start=False))


def test_http_run_json_and_multipart(fake):
    live, client = fake
    r = client.post("/run", json={"sample": "tabby", "timeBudgetMs": 40, "policies": ["load-aware"]})
    assert r.status_code == 200 and r.json()["source"] == "tabby"
    assert live.calls[-1][2:4] == (40.0, ["load-aware"])
    r = client.post("/run", files={"image": ("x.png", png_bytes(), "image/png")},
                    data={"policies": '["confidence-gate"]', "timeBudgetMs": "60"})
    assert r.status_code == 200 and r.json()["source"] == "upload"
    assert live.calls[-1][2:5] == (60.0, ["confidence-gate"], (1, 3, 224, 224))


def test_http_errors_are_json(fake):
    _, client = fake
    r = client.post("/run", json={"sample": "nope"})
    assert r.status_code == 400 and r.json()["error"] == "bad_sample"
    r = client.post("/run", files={"image": ("x.gif", b"GIF89a" + b"\0" * 20, "image/gif")})
    assert r.status_code == 415 and r.json()["error"] == "unsupported_image"
    r = client.post("/run", content=b"{nope", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"] == "bad_json"
    r = client.post("/load", json={"jobs": 65})
    assert r.status_code == 400 and r.json()["error"] == "bad_jobs"
    r = client.post("/reload", json={"jobs": 4})
    assert r.status_code == 400 and r.json()["error"] == "bad_jobs"


def test_http_busy_is_429_with_retry_after(fake):
    _, client = fake
    r = client.post("/load", json={"jobs": 8, "concurrency": 2})
    assert r.status_code == 429
    assert r.json() == {"error": "busy", "message": "the runtime is busy with another run; try again shortly",
                        "retryAfter": 4.2}
    assert r.headers["retry-after"] == "5"


def test_http_rate_limit(fake, monkeypatch):
    _, client = fake
    monkeypatch.setattr(server, "POSTS_PER_MINUTE", 3)
    codes = [client.post("/reload", json={"jobs": 8, "swaps": 1}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    assert client.get("/health").status_code == 200  # GETs are counted separately


def test_http_rate_limit_trusts_the_proxy_only_with_its_key(monkeypatch):
    monkeypatch.setenv("LOOMCORE_PROXY_KEY", "k")
    monkeypatch.setattr(server, "POSTS_PER_MINUTE", 1)
    client = TestClient(server.create_app(FakeLive(), start=False))
    body = {"jobs": 8, "swaps": 1}
    proxied = lambda ip, key="k": {"x-proxy-key": key, "x-client-ip": ip}  # noqa: E731
    assert client.post("/reload", json=body, headers=proxied("1.1.1.1")).status_code == 200
    assert client.post("/reload", json=body, headers=proxied("2.2.2.2")).status_code == 200  # another visitor
    assert client.post("/reload", json=body, headers=proxied("1.1.1.1")).status_code == 429
    # A wrong key is not believed: the request counts against its real address.
    assert client.post("/reload", json=body, headers=proxied("3.3.3.3", "x")).status_code == 200
    assert client.post("/reload", json=body, headers=proxied("4.4.4.4", "x")).status_code == 429


def test_http_upload_size_guard(fake):
    _, client = fake
    big = b"\xff\xd8\xff" + b"\0" * (server.MAX_UPLOAD_BYTES + 200_000)
    r = client.post("/run", files={"image": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413


def test_http_cors_allows_only_the_console(fake):
    _, client = fake
    ok = client.get("/health", headers={"Origin": "https://loomcore.amittal.dev"})
    assert ok.headers.get("access-control-allow-origin") == "https://loomcore.amittal.dev"
    dev = client.options("/run", headers={"Origin": "http://localhost:5177", "Access-Control-Request-Method": "POST",
                                          "Access-Control-Request-Headers": "content-type"})
    assert dev.status_code == 200 and dev.headers.get("access-control-allow-origin") == "http://localhost:5177"
    other = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_http_landing_page(fake):
    _, client = fake
    r = client.get("/")
    assert r.status_code == 200 and "loomcore.amittal.dev" in r.text
