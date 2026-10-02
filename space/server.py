"""Loomcore live: a small HTTP server over the real C++ runtime.

It loads the reference graph (MobileNetV2 -> confidence gate -> bert_tiny)
through the Python bindings and exposes it to the console at
loomcore.amittal.dev:

  GET  /health   liveness, version, models, CPU
  GET  /graph    the loaded graph, the router policy chain, scheduler flags
  POST /run      one job on a sample or an uploaded image -> RunResult
  POST /load     N jobs at concurrency C -> LoadResult
  POST /reload   hot-swap the graph under continuous load -> ReloadResult
  GET  /bench    loomcore_bench's FP32 vs INT8 numbers on this machine

Every result carries `trace`: the Chrome Trace Event JSON that the runtime's
own exporter (loomcore.export_perfetto_trace) produced from the JSON-lines
log of exactly that request's jobs.

Guards: one /load or /reload at a time (429 with retryAfter), one job-running
request in the runtime at a time, 20 POSTs a minute per IP, request timeouts,
uploads of at most 2 MB (JPEG/PNG, checked by content), and nothing written
outside a temporary directory.

Run locally: python space/server.py  (needs the C++ build and the prepared
models; see docs/BUILD.md and space/README.md).
"""
from __future__ import annotations

import asyncio
import calendar
import hmac
import io
import json
import locale
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


MODELS_DIR = _env_path("LOOMCORE_MODELS_DIR", REPO / "models")
ASSETS_DIR = _env_path("LOOMCORE_ASSETS_DIR", REPO / "assets")
SAMPLES_DIR = _env_path("LOOMCORE_SAMPLES_DIR", HERE / "samples")
BINDINGS_DIR = _env_path("LOOMCORE_PY_DIR", REPO / "bindings" / "python")
def _version() -> str:
    explicit = os.environ.get("LOOMCORE_VERSION")
    if explicit:
        return explicit
    sha_file = REPO / ".git-sha"  # written by the Dockerfile from the commit it built
    sha = sha_file.read_text(encoding="utf-8").strip() if sha_file.exists() else ""
    return f"0.1.0+{sha}" if sha else "0.1.0"


VERSION = _version()

ALLOWED_ORIGINS = ["https://loomcore.amittal.dev", "http://localhost:5177"] + [
    o for o in os.environ.get("LOOMCORE_EXTRA_ORIGINS", "").split(",") if o
]

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
POSTS_PER_MINUTE = 20
GETS_PER_MINUTE = 240
RUN_TIMEOUT_S = 25.0
HEAVY_TIMEOUT_S = 90.0
LOG_RING = 2048  # Logger's in-memory ring (loomcore/logger.h); a request's slice must fit in it

# --------------------------------------------------------------------------
# The router policy chain a visitor can toggle. Order is the chain order:
# sheds first (a skip is terminal in CompositeRouter), then the gate, then
# precision, then the lane.
# --------------------------------------------------------------------------
POLICIES: list[dict[str, str]] = [
    {
        "name": "circuit-breaker",
        "cpp": "CircuitBreakerPolicy(0.5, 4, 2000)",
        "reads": "the node's recent success and failure outcomes",
        "decides": "skips a node whose error rate reached 50% over at least 4 runs; lets one probe through after 2 s",
    },
    {
        "name": "bulkhead",
        "cpp": "BulkheadPolicy(6)",
        "reads": "how many requests for the node are in flight",
        "decides": "sheds (skips) a request once 6 are already in flight for that node",
    },
    {
        "name": "confidence-gate",
        "cpp": "ConfidenceGatePolicy(0.85)",
        "reads": "mobilenet's softmax confidence for this image",
        "decides": "skips bert_tiny once mobilenet is at least 85% sure; there is nothing left to describe",
    },
    {
        "name": "precision-planner",
        "cpp": "PlannedPrecisionPolicy()",
        "reads": "measured FP32 and INT8 p95 per node, and the job's budget",
        "decides": "solves a 0/1 knapsack over the critical path for which nodes to run INT8 so the job fits",
    },
    {
        "name": "latency-budget",
        "cpp": "LatencyBudgetPolicy(30)",
        "reads": "the job's remaining time budget",
        "decides": "drops a node to INT8 once less than 30 ms of the budget is left",
    },
    {
        "name": "load-aware",
        "cpp": "LoadAwareBackendPolicy()",
        "reads": "the CPU and GPU_SIM queue depths",
        "decides": "sends the node to whichever lane has the shallower queue",
    },
]
POLICY_NAMES = [p["name"] for p in POLICIES]
# Bulkhead starts off: with it on, a load test above 6 concurrent jobs is
# mostly sheds, which is worth seeing on purpose rather than by default.
DEFAULT_POLICIES = [name for name in POLICY_NAMES if name != "bulkhead"]
SHED_MESSAGE = "mobilenet was shed upstream, so bert_tiny has no label to embed"

SCHEDULER_FLAGS = {
    "enable_admission_control": True,
    "enable_precision_planning": True,
    "enable_deadline_cancellation": True,
    "deadline_reaper_poll_ms": 2.0,
    "use_edf_scoring": True,
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, **extra: Any):
        super().__init__(message)
        self.status, self.code, self.message, self.extra = status, code, message, extra


# --------------------------------------------------------------------------
# Pure helpers (no bindings needed; unit-tested in space/tests)
# --------------------------------------------------------------------------


def parse_policies(value: Any) -> list[str]:
    """None -> the default chain. A list (or comma-separated string) of names
    -> those policies, in chain order. Unknown names are a 400."""
    if value is None:
        return list(DEFAULT_POLICIES)
    if isinstance(value, str):
        value = [v for v in (s.strip() for s in value.split(",")) if v]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ApiError(400, "bad_policies", "policies must be a list of policy names")
    unknown = sorted(set(value) - set(POLICY_NAMES))
    if unknown:
        raise ApiError(400, "bad_policies", f"unknown policies: {', '.join(unknown)}", known=POLICY_NAMES)
    return [name for name in POLICY_NAMES if name in value]


def parse_budget(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        budget = float(value)
    except (TypeError, ValueError):
        raise ApiError(400, "bad_budget", "timeBudgetMs must be a number") from None
    if not math.isfinite(budget) or budget < 1 or budget > 10_000:
        raise ApiError(400, "bad_budget", "timeBudgetMs must be between 1 and 10000")
    return budget


def parse_int(value: Any, name: str, lo: int, hi: int, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ApiError(400, f"bad_{name}", f"{name} must be an integer from {lo} to {hi}")
    try:
        number = int(value)
    except ValueError:
        raise ApiError(400, f"bad_{name}", f"{name} must be an integer from {lo} to {hi}") from None
    if number != float(value) or not lo <= number <= hi:
        raise ApiError(400, f"bad_{name}", f"{name} must be an integer from {lo} to {hi}")
    return number


def sniff_image(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    raise ApiError(415, "unsupported_image", "the image must be a JPEG or PNG")


def image_to_tensor(data: bytes) -> np.ndarray:
    """Decode -> RGB -> 224x224 (bilinear) -> [0,1] -> ImageNet mean/std ->
    NCHW float32, the same preprocessing examples/run_example.cpp does."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ApiError(413, "too_large", "the image must be 2 MB or smaller")
    sniff_image(data)
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.width * img.height > 40_000_000:
                raise ApiError(413, "too_large", "the image has too many pixels")
            rgb = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    except ApiError:
        raise
    except Exception:  # noqa: BLE001 - any decoder failure is the client's file
        raise ApiError(415, "unreadable_image", "the image could not be decoded") from None
    arr = (np.asarray(rgb, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    e = np.exp(shifted)
    return e / np.sum(e)


def iso_now() -> str:
    """The Logger's own timestamp format, so log lines compare as strings."""
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + ".%06dZ" % int((t % 1) * 1e6)


def iso_to_us(ts: str) -> int:
    seconds = calendar.timegm(time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S"))
    return seconds * 1_000_000 + int(ts[20:26])


def parse_decision(message: str) -> list[dict[str, str]]:
    """'CompositeRouter: A: why | B: why' or 'CompositeRouter -> A: why' ->
    [{policy, reason}, ...]."""
    body = re.sub(r"^CompositeRouter(?::\s*| -> )", "", message)
    out = []
    for part in body.split(" | "):
        policy, _, reason = part.partition(": ")
        out.append({"policy": policy.strip(), "reason": reason.strip()})
    return out


def summarise(lines: list[str]) -> dict[str, Any]:
    """Reads one request's JSON-lines log slice into the numbers the console
    shows: per-node latency, batch sizes, router decisions, skips, rejects and
    cancels."""
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    node_ms: dict[str, list[float]] = defaultdict(list)
    node_prec: dict[str, Counter] = defaultdict(Counter)
    node_lane: dict[str, Counter] = defaultdict(Counter)
    batch_sizes: dict[str, Counter] = defaultdict(Counter)
    decisions: list[dict[str, Any]] = []
    policy_counts: Counter = Counter()
    skipped: Counter = Counter()
    completed_nodes = []
    rejected = cancelled = shed = errors = 0
    for ev in events:
        kind = ev.get("event")
        if kind == "node_completed":
            node_ms[ev["node_id"]].append(float(ev.get("latency_ms", 0.0)))
            node_prec[ev["node_id"]][ev.get("precision", "FP32")] += 1
            node_lane[ev["node_id"]][ev.get("backend", "CPU")] += 1
            completed_nodes.append(ev)
        elif kind == "batch_flushed":
            batch_sizes[ev.get("node_id", "")][int(ev.get("batch_size", 1))] += 1
        elif kind == "routing_decision":
            parsed = parse_decision(ev.get("message", ""))
            for p in parsed:
                policy_counts[p["policy"]] += 1
            decisions.append({"job": ev.get("job_id"), "node": ev.get("node_id"), "message": ev.get("message", ""),
                              "policies": parsed, "ts": ev.get("ts")})
        elif kind == "node_skipped":
            skipped[ev.get("node_id", "")] += 1
        elif kind == "job_rejected":
            rejected += 1
        elif kind == "error":
            message = ev.get("message", "")
            if "exceeded its time budget" in message:
                cancelled += 1
            elif SHED_MESSAGE in message:
                shed += 1
            else:
                errors += 1

    def pct(values: list[float], q: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else 0.0

    nodes = []
    for node_id, values in node_ms.items():
        nodes.append({
            "id": node_id, "count": len(values), "p50": round(pct(values, 0.50), 3),
            "p95": round(pct(values, 0.95), 3), "mean": round(sum(values) / len(values), 3),
            "precisions": dict(node_prec[node_id]), "lanes": dict(node_lane[node_id]),
        })
    batches = []
    for node_id, sizes in batch_sizes.items():
        total = sum(sizes.values())
        batches.append({"node": node_id, "count": total, "sizes": {str(k): v for k, v in sorted(sizes.items())},
                        "meanSize": round(sum(k * v for k, v in sizes.items()) / total, 2)})
    return {
        "nodes": sorted(nodes, key=lambda n: n["id"] != "mobilenet"),
        "batches": sorted(batches, key=lambda b: b["node"] != "mobilenet"),
        "decisions": decisions,
        "policyCounts": [{"policy": k, "count": v} for k, v in policy_counts.most_common()],
        "skipped": dict(skipped),
        "rejected": rejected,
        "shed": shed,
        "cancelled": cancelled,
        "errors": errors,
        "completedNodes": completed_nodes,
    }


def trace_base_us(lines: list[str]) -> int | None:
    """The zero point loomcore.export_perfetto_trace uses for a log: the
    earliest event, counting each batch from its start (batch_flushed is
    logged when the batch ends)."""
    base = None
    for line in lines:
        try:
            ev = json.loads(line)
            ts = iso_to_us(ev["ts"])
        except (ValueError, KeyError):
            continue
        if ev.get("event") == "batch_flushed":
            ts -= int(float(ev.get("latency_ms", 0.0)) * 1000.0)
        base = ts if base is None else min(base, ts)
    return base


def parse_bench_output(text: str) -> list[dict[str, Any]]:
    rows = []
    for m in re.finditer(r"^(\S+)\s+(FP32|INT8)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", text, re.M):
        rows.append({"model": m.group(1), "precision": m.group(2), "mean": float(m.group(3)),
                     "p50": float(m.group(4)), "p95": float(m.group(5)), "n": int(m.group(6))})
    return rows


# Arm core names by (implementer, part) from /proc/cpuinfo; the ones a cloud
# VM or a recent laptop is likely to report.
ARM_PARTS = {
    ("0x41", "0xd03"): "Cortex-A53", ("0x41", "0xd08"): "Cortex-A72", ("0x41", "0xd0b"): "Cortex-A76",
    ("0x41", "0xd0c"): "Neoverse N1", ("0x41", "0xd40"): "Neoverse V1", ("0x41", "0xd49"): "Neoverse N2",
    ("0x41", "0xd4f"): "Neoverse V2", ("0x41", "0xd8e"): "Neoverse N3", ("0x41", "0xd84"): "Neoverse V3",
    ("0xc0", "0xac3"): "AmpereOne", ("0xc0", "0xac4"): "AmpereOne",
}


def parse_cpuinfo(text: str, name_override: str | None = None) -> dict[str, Any]:
    """x86 reports `model name` and `flags`; aarch64 reports `Features` and
    `CPU implementer`/`CPU part` instead. Each architecture gets the flags that
    explain its INT8 result: VNNI/AMX on x86, the int8 dot-product (asimddp,
    SDOT/UDOT) and matrix-multiply (i8mm) instructions on Arm."""
    cores = len(re.findall(r"^processor\s*:", text, re.M))
    features = re.search(r"^Features\s*:\s*(.+)$", text, re.M)
    if features:
        feats = set(features.group(1).split())
        implementer = re.search(r"^CPU implementer\s*:\s*(\S+)", text, re.M)
        part = re.search(r"^CPU part\s*:\s*(\S+)", text, re.M)
        key = (implementer.group(1).lower() if implementer else "", part.group(1).lower() if part else "")
        core = ARM_PARTS.get(key)
        core_name = core or (f"Arm CPU part {key[1]}" if key[1] else "Arm CPU")
        model = f"{name_override} ({core_name})" if name_override else (f"Arm {core}" if core else core_name)
        return {
            "arch": "aarch64",
            "model": model,
            "logicalCpus": cores or os.cpu_count(),
            "flags": {
                "asimd": "asimd" in feats,
                "asimddp": "asimddp" in feats,
                "i8mm": "i8mm" in feats,
                "bf16": "bf16" in feats,
                "sve": "sve" in feats,
            },
        }
    model = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
    flags_line = re.search(r"^flags\s*:\s*(.+)$", text, re.M)
    flags = set(flags_line.group(1).split()) if flags_line else set()
    return {
        "arch": "x86_64",
        "model": name_override or (model.group(1).strip() if model else platform.processor() or "unknown CPU"),
        "logicalCpus": cores or os.cpu_count(),
        "flags": {
            "avx2": "avx2" in flags,
            "avx512f": "avx512f" in flags,
            "avx512_vnni": "avx512_vnni" in flags,
            "avx_vnni": "avx_vnni" in flags,
            "amx_int8": "amx_int8" in flags,
        } if flags_line else None,
    }


def detect_cpu() -> dict[str, Any]:
    try:
        # LOOMCORE_CPU_NAME names the chip when /proc/cpuinfo can't (Arm reports
        # only the core): e.g. "Ampere Altra" on an Oracle A1 VM.
        return parse_cpuinfo(Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace"),
                             os.environ.get("LOOMCORE_CPU_NAME") or None)
    except OSError:
        pass
    info: dict[str, Any] = {"arch": platform.machine().lower() or "unknown", "model": platform.processor() or "unknown CPU",
                            "logicalCpus": os.cpu_count(), "flags": None}
    if sys.platform == "win32":
        try:
            import winreg  # noqa: PLC0415

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                info["model"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
            info["flags"] = _windows_cpu_flags()
        except Exception:  # noqa: BLE001 - best effort; the console says "not reported"
            pass
    return info


def _windows_cpu_flags() -> dict[str, bool]:
    """CPUID leaf 7 on Windows (there is no /proc/cpuinfo, and the OS feature
    query has no VNNI bit): a few bytes of x86-64 that run CPUID and return one
    register, placed in executable memory. Only used for a local run."""
    import ctypes  # noqa: PLC0415

    kernel32 = ctypes.windll.kernel32
    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong]
    prologue = b"\x53\x89\xc8\x89\xd1\x0f\xa2"  # push rbx; mov eax,ecx (leaf); mov ecx,edx (subleaf); cpuid
    epilogues = {"eax": b"\x5b\xc3", "ebx": b"\x89\xd8\x5b\xc3", "ecx": b"\x89\xc8\x5b\xc3", "edx": b"\x89\xd0\x5b\xc3"}

    def cpuid(leaf: int, subleaf: int, reg: str) -> int:
        code = prologue + epilogues[reg]
        mem = kernel32.VirtualAlloc(None, len(code), 0x3000, 0x40)  # MEM_COMMIT|MEM_RESERVE, PAGE_EXECUTE_READWRITE
        ctypes.memmove(mem, code, len(code))
        return ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32)(mem)(leaf, subleaf)

    ebx7, ecx7, edx7, eax71 = cpuid(7, 0, "ebx"), cpuid(7, 0, "ecx"), cpuid(7, 0, "edx"), cpuid(7, 1, "eax")
    return {
        "avx2": bool(ebx7 >> 5 & 1),
        "avx512f": bool(ebx7 >> 16 & 1),
        "avx512_vnni": bool(ecx7 >> 11 & 1),
        "avx_vnni": bool(eax71 >> 4 & 1),
        "amx_int8": bool(edx7 >> 25 & 1),
    }


class RateLimiter:
    """A sliding one-minute window per (client, bucket)."""

    def __init__(self) -> None:
        self.hits: dict[tuple[str, str], deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, client: str, bucket: str, limit: int, now: float | None = None) -> float:
        """Records a hit; returns 0 if allowed, else seconds until the next one is."""
        now = time.monotonic() if now is None else now
        with self.lock:
            q = self.hits[(client, bucket)]
            while q and now - q[0] >= 60.0:
                q.popleft()
            if len(q) >= limit:
                return max(0.1, 60.0 - (now - q[0]))
            q.append(now)
            if len(self.hits) > 10_000:  # forget idle clients
                for key in [k for k, v in self.hits.items() if not v or now - v[-1] > 60.0]:
                    del self.hits[key]
            return 0.0


# --------------------------------------------------------------------------
# The runtime
# --------------------------------------------------------------------------


class UpstreamShed(RuntimeError):
    pass


class Busy(Exception):
    def __init__(self, retry_after: float):
        super().__init__("busy")
        self.retry_after = retry_after


class LiveRuntime:
    """Owns the one loomcore.Runtime, serialises every job-running operation,
    and turns each operation's log slice into a result and a trace."""

    def __init__(self) -> None:
        sys.path.insert(0, str(BINDINGS_DIR))
        import loomcore  # noqa: PLC0415 - imported lazily so the helpers above work without the build

        self.lc = loomcore
        self.tmp = Path(tempfile.mkdtemp(prefix="loomcore-live-"))
        self.labels = [line.strip() for line in (ASSETS_DIR / "imagenet_labels.txt").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        self.tokenizer = loomcore.WordPieceTokenizer(str(MODELS_DIR / "bert_tiny_tokenizer" / "vocab.txt"), 16)
        self.samples = json.loads((SAMPLES_DIR / "samples.json").read_text(encoding="utf-8"))
        self.sample_tensors = {s["id"]: image_to_tensor((SAMPLES_DIR / s["file"]).read_bytes()) for s in self.samples}
        self.config_path = self._write_config()
        self.cpu = detect_cpu()

        self.lock = threading.Lock()  # one job-running operation in the runtime at a time
        self.heavy = threading.Lock()  # one /load or /reload at a time
        self.heavy_started = 0.0
        self.heavy_estimate_s = 10.0
        self.observed: dict[str, np.ndarray] = {}
        self.policies: list[str] | None = None
        self.reload_build_ms = 250.0
        self.ready = False
        self.started = time.time()
        self.bench: dict[str, Any] = {"status": "pending"}
        self.runtime = loomcore.Runtime()

    # -- graph -----------------------------------------------------------

    def _write_config(self) -> str:
        def node(node_id, backend, priority, window, deps, stem, **extra):
            return {
                "id": node_id, "backend": backend, "priority": priority, "max_batch_size": 4,
                "batch_window_ms": window, "depends_on": deps, **extra,
                "variants": [
                    {"precision": "FP32", "model_path": str(MODELS_DIR / f"{stem}.onnx").replace("\\", "/")},
                    {"precision": "INT8", "model_path": str(MODELS_DIR / f"{stem}.int8.onnx").replace("\\", "/")},
                ],
            }

        # examples/graph_config.json, with absolute model paths and no log file
        # (the server reads the Logger's in-memory ring instead).
        config = {
            "scheduler": {"cpu_threads": 0, "gpu_sim_threads": 2, "gpu_sim_fixed_overhead_ms": 1.5,
                          "gpu_sim_bytes_per_ms": 250000.0},
            "nodes": [
                node("mobilenet", "CPU", 5, 8, [], "mobilenetv2"),
                node("bert_tiny", "GPU_SIM", 3, 12, ["mobilenet"], "bert_tiny", confidence_source="mobilenet"),
            ],
        }
        path = self.tmp / "graph.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return str(path)

    def _options(self, flags: bool = True):
        options = self.lc.RuntimeOptions()
        options.log_to_stdout = False
        options.intra_op_threads_per_variant = 1
        if flags:
            for key, value in SCHEDULER_FLAGS.items():
                setattr(options.scheduler, key, value)
        return options

    def _binders(self):
        labels, tokenizer = self.labels, self.tokenizer

        def mobilenet(graph_inputs, _upstream):
            return [("data", graph_inputs["data"])]

        def bert_tiny(_graph_inputs, upstream):
            if not upstream.get("mobilenet"):
                # The router skipped mobilenet for this job (bulkhead or
                # circuit breaker), so there is no prediction to describe.
                # Failing the job says so in the log and the trace.
                raise UpstreamShed(SHED_MESSAGE)
            idx = int(np.argmax(upstream["mobilenet"][0]))
            label = labels[idx] if idx < len(labels) else "object"
            return list(tokenizer.encode(f"a photo of a {label}").items())

        def confidence(upstream):
            return float(np.max(softmax(upstream[0].reshape(-1))))

        return {"mobilenet": mobilenet, "bert_tiny": bert_tiny}, {"bert_tiny": confidence}

    def _router(self, policies: list[str]):
        lc, observed = self.lc, self.observed

        class Observer(lc.RoutingPolicy):
            """Never has an opinion; records mobilenet's logits per job as
            bert_tiny is routed (RoutingContext carries them because bert_tiny
            declares mobilenet as its confidence_source), so the result can
            show the label even when bert_tiny is skipped."""

            def name(self):
                return "Observer"

            def decide(self, ctx):
                if ctx.node_id == "bert_tiny" and ctx.upstream_confidence_source:
                    observed[ctx.job_id] = np.array(ctx.upstream_confidence_source[0]).reshape(-1)
                return None

        build = {
            "circuit-breaker": lambda: lc.CircuitBreakerPolicy(0.5, 4, 2000.0),
            "bulkhead": lambda: lc.BulkheadPolicy(6),
            "confidence-gate": lambda: lc.ConfidenceGatePolicy(0.85),
            "precision-planner": lambda: lc.PlannedPrecisionPolicy(),
            "latency-budget": lambda: lc.LatencyBudgetPolicy(30.0),
            "load-aware": lambda: lc.LoadAwareBackendPolicy(),
        }
        router = lc.CompositeRouter()
        router.add(Observer())
        for name in policies:
            router.add(build[name]())
        return router

    def _apply_policies(self, policies: list[str]) -> float | None:
        """Hot-swaps the graph with a router for `policies` if it differs from
        the loaded one. Returns how long the swap took, or None if none was
        needed. Caller holds self.lock."""
        if self.policies == policies:
            return None
        binders, extractors = self._binders()
        t0 = time.perf_counter()
        self.runtime.reload_graph(self.config_path, binders, extractors, self._router(policies), self._options())
        self.policies = list(policies)
        return round((time.perf_counter() - t0) * 1000.0, 1)

    def start(self) -> None:
        """Loads the graph, warms both precisions of both nodes so admission
        control and the precision planner have measurements to reason from,
        then runs the benchmark. Runs on a background thread at startup."""
        with self.lock:
            binders, extractors = self._binders()
            warm = self.lc.CompositeRouter()
            warm.add(self.lc.LatencyBudgetPolicy(1e9))  # any budgeted job runs INT8
            self.runtime.load_graph(self.config_path, binders, extractors, warm, self._options(flags=False))
            image = self.sample_tensors[self.samples[0]["id"]]
            for _ in range(4):
                self.runtime.run({"data": image}, 0, 1e6)  # INT8, both nodes
                self.runtime.run({"data": image})  # FP32, both nodes
            t0 = time.perf_counter()
            self._apply_policies(DEFAULT_POLICIES)
            self.reload_build_ms = (time.perf_counter() - t0) * 1000.0
            self.ready = True
            if os.environ.get("LOOMCORE_SKIP_BENCH"):
                self.bench = {"status": "skipped", "cpu": self.cpu}
                return
            self.bench = {"status": "running", "cpu": self.cpu}
            self.bench = run_bench(self.cpu)

    # -- log slices --------------------------------------------------------

    def _slice(self, since: str) -> list[str]:
        lines = [line for line in self.runtime.recent_logs(LOG_RING) if _ts(line) >= since]
        jobs = set()
        for line in lines:
            if '"job_submitted"' in line:
                jobs.add(json.loads(line).get("job_id"))
        kept = []
        for line in lines:
            ev = json.loads(line)
            job = ev.get("job_id")
            if job in jobs or ev.get("event") == "job_rejected" or (job or "").startswith("("):
                kept.append(line)
        return kept

    def _trace(self, lines: list[str]) -> list[dict[str, Any]]:
        with tempfile.TemporaryDirectory(dir=self.tmp) as d:
            src, dst = Path(d) / "run.jsonl", Path(d) / "trace.json"
            src.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.lc.export_perfetto_trace(str(src), str(dst))
            return json.loads(dst.read_text(encoding="utf-8"))["traceEvents"]

    def _acquire(self, timeout: float) -> None:
        if not self.ready:
            raise ApiError(503, "warming_up", "the runtime is still loading; try again in a few seconds",
                           retryAfter=5)
        if not self.lock.acquire(timeout=timeout):
            raise Busy(3.0)

    def _acquire_heavy(self) -> None:
        if not self.heavy.acquire(blocking=False):
            elapsed = time.monotonic() - self.heavy_started
            raise Busy(max(1.0, round(self.heavy_estimate_s - elapsed, 1)))
        try:
            self._acquire(timeout=10.0)
        except BaseException:
            self.heavy.release()
            raise
        self.heavy_started = time.monotonic()

    # -- operations ----------------------------------------------------------

    def inputs_for(self, sample: str | None, upload: bytes | None) -> tuple[np.ndarray, str]:
        if upload is not None:
            return image_to_tensor(upload), "upload"
        sample = sample or self.samples[0]["id"]
        if sample not in self.sample_tensors:
            raise ApiError(400, "bad_sample", f"unknown sample '{sample}'",
                           known=[s["id"] for s in self.samples])
        return self.sample_tensors[sample], sample

    def run(self, tensor: np.ndarray, source: str, budget: float | None, policies: list[str]) -> dict[str, Any]:
        self._acquire(timeout=15.0)
        try:
            swap_ms = self._apply_policies(policies)
            self.observed.clear()
            since = iso_now()
            status, error, result = "ok", None, {}
            t0 = time.perf_counter()
            try:
                result = self.runtime.run({"data": tensor}, 0, budget if budget is not None else -1.0)
            except self.lc.JobRejectedError as exc:
                status, error = "rejected", str(exc)
            except self.lc.DeadlineExceededError as exc:
                status, error = "cancelled", str(exc)
            except UpstreamShed as exc:
                status, error = "shed", str(exc)
            except Exception as exc:  # noqa: BLE001 - report it, never a 500
                status, error = "failed", str(exc)
            total_ms = (time.perf_counter() - t0) * 1000.0
            if status == "cancelled":
                time.sleep(0.06)  # let the cancelled ONNX Runtime call unwind and log its (cut-short) batch
            lines = self._slice(since)
        finally:
            self.lock.release()

        s = summarise(lines)
        logits = next(iter(self.observed.values()), None)
        label = confidence = None
        top5: list[dict[str, Any]] = []
        if logits is not None:
            probs = softmax(logits)
            order = np.argsort(probs)[::-1][:5]
            top5 = [{"label": self.labels[i], "p": round(float(probs[i]), 4)} for i in order]
            label, confidence = top5[0]["label"], top5[0]["p"]
        bert = result.get("bert_tiny") or []
        embedding = [round(float(v), 5) for v in np.asarray(bert[1]).reshape(-1)[:16]] if len(bert) > 1 else None
        job_ids = sorted({ev.get("job_id") for ev in s["completedNodes"]} - {None})
        return {
            "status": status, "error": error, "source": source, "timeBudgetMs": budget, "policies": policies,
            "graphSwapMs": swap_ms, "jobId": job_ids[0] if job_ids else None,
            "label": label, "confidence": confidence, "top5": top5,
            "text": f"a photo of a {label}" if embedding is not None and label else None,
            "embedding": embedding, "skipped": sorted(s["skipped"]),
            "decisions": [{"node": d["node"], "policies": d["policies"], "message": d["message"]}
                          for d in s["decisions"]],
            "nodes": [{"id": ev["node_id"], "precision": ev.get("precision"), "backend": ev.get("backend"),
                       "ms": round(float(ev.get("latency_ms", 0.0)), 3)} for ev in s["completedNodes"]],
            "totalMs": round(total_ms, 3),
            "trace": self._trace(lines),
        }

    def load(self, jobs: int, concurrency: int, budget: float | None, policies: list[str],
             tensor: np.ndarray, source: str) -> dict[str, Any]:
        self._acquire_heavy()
        self.heavy_estimate_s = 2.0 + jobs * 0.08
        try:
            swap_ms = self._apply_policies(policies)
            since = iso_now()
            outcomes: Counter = Counter()
            job_ms: list[float] = []

            def one(_i: int) -> None:
                t0 = time.perf_counter()
                try:
                    self.runtime.run({"data": tensor}, 0, budget if budget is not None else -1.0)
                    outcomes["completed"] += 1
                    job_ms.append((time.perf_counter() - t0) * 1000.0)
                except self.lc.JobRejectedError:
                    outcomes["rejected"] += 1
                except self.lc.DeadlineExceededError:
                    outcomes["cancelled"] += 1
                except UpstreamShed:
                    outcomes["shed"] += 1
                except Exception:  # noqa: BLE001
                    outcomes["failed"] += 1

            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(one, range(jobs)))
            wall_ms = (time.perf_counter() - t0) * 1000.0
            time.sleep(0.06)
            lines = self._slice(since)
        finally:
            self.lock.release()
            self.heavy.release()

        s = summarise(lines)
        ordered = sorted(job_ms)
        return {
            "jobs": jobs, "concurrency": concurrency, "timeBudgetMs": budget, "policies": policies,
            "source": source, "graphSwapMs": swap_ms,
            "submitted": jobs, "completed": outcomes["completed"], "rejected": outcomes["rejected"],
            "cancelled": outcomes["cancelled"], "shed": outcomes["shed"], "failed": outcomes["failed"],
            "wallMs": round(wall_ms, 1), "throughput": round(outcomes["completed"] / (wall_ms / 1000.0), 2),
            "jobP50": round(ordered[len(ordered) // 2], 2) if ordered else None,
            "jobP95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2) if ordered else None,
            "nodes": s["nodes"], "batches": s["batches"], "decisions": s["policyCounts"],
            "skipped": s["skipped"], "trace": self._trace(lines),
        }

    def reload(self, jobs: int, swaps: int) -> dict[str, Any]:
        """examples/reload_demo.cpp over HTTP: producer threads keep submitting
        while the graph is rebuilt and swapped `swaps` times; every job must
        complete. The swaps run back to back once load is established, and the
        producers stretch the jobs still to submit over the time the swaps still
        to do will take (re-estimated from each measured build), so every swap
        happens while jobs are being submitted and run."""
        self._acquire_heavy()
        policies = list(self.policies or DEFAULT_POLICIES)
        self.heavy_estimate_s = 2.0 + (swaps + 1) * max(0.05, self.reload_build_ms / 1000.0) * 1.5
        try:
            since = iso_now()
            tensor = self.sample_tensors[self.samples[0]["id"]]
            lock = threading.Lock()
            st = {"submitted": 0, "completed": 0, "lost": 0, "swapped": 0, "next_at": 0.0,
                  "build_s": max(0.05, self.reload_build_ms / 1000.0) * 1.3}
            start = time.perf_counter()

            def spacing_locked() -> float:
                left = jobs - st["submitted"]
                if st["swapped"] >= swaps or left <= 0:
                    return 0.0
                return (swaps - st["swapped"]) * st["build_s"] * 1.2 / left

            def producer() -> None:
                while True:
                    with lock:
                        if st["submitted"] >= jobs:
                            return
                        st["submitted"] += 1
                        now = time.perf_counter()
                        at = max(now, st["next_at"])
                        st["next_at"] = at + spacing_locked()
                    if at > now:
                        time.sleep(at - now)
                    try:
                        self.runtime.run({"data": tensor})
                        ok = True
                    except Exception:  # noqa: BLE001 - any failure is a lost job
                        ok = False
                    with lock:
                        st["completed" if ok else "lost"] += 1

            threads = [threading.Thread(target=producer, daemon=True) for _ in range(4)]
            for t in threads:
                t.start()
            swap_log = []
            binders, extractors = self._binders()
            warm = min(jobs - 1, max(4, jobs // (swaps + 1)))
            while True:  # let the load get going first
                with lock:
                    if st["submitted"] >= warm:
                        break
                time.sleep(0.002)
            for i in range(1, swaps + 1):
                with lock:
                    done_before = st["completed"]
                    in_flight = st["submitted"] - st["completed"] - st["lost"]
                wall_us = int(time.time() * 1e6)
                t0 = time.perf_counter()
                self.runtime.reload_graph(self.config_path, binders, extractors, self._router(policies),
                                          self._options())
                build_ms = (time.perf_counter() - t0) * 1e3
                with lock:
                    st["swapped"] += 1
                    st["build_s"] = max(st["build_s"] * 0.6, build_ms / 1000.0 * 1.15)
                self.reload_build_ms = 0.7 * self.reload_build_ms + 0.3 * build_ms
                swap_log.append({"index": i, "wallUs": wall_us, "buildMs": round(build_ms, 1),
                                 "completedBefore": done_before, "inFlight": in_flight})
            counts = st
            for t in threads:
                t.join(timeout=HEAVY_TIMEOUT_S)
            wall_ms = (time.perf_counter() - start) * 1000.0
            time.sleep(0.03)
            lines = self._slice(since)
        finally:
            self.lock.release()
            self.heavy.release()

        base = trace_base_us(lines) or (swap_log[0]["wallUs"] if swap_log else 0)
        for entry in swap_log:
            entry["atMs"] = round((entry.pop("wallUs") - base) / 1000.0, 3)
        s = summarise(lines)
        return {
            "jobs": jobs, "swaps": swaps, "policies": policies,
            "submitted": counts["submitted"], "completed": counts["completed"], "lost": counts["lost"],
            "wallMs": round(wall_ms, 1), "swapTimings": swap_log,
            "nodes": s["nodes"], "batches": s["batches"], "trace": self._trace(lines),
        }

    def graph(self) -> dict[str, Any]:
        g = self.runtime.graph()
        nodes = [{
            "id": n["id"], "backend": n["backend"], "priority": n["priority"], "maxBatchSize": n["max_batch_size"],
            "batchWindowMs": n["batch_window_ms"], "dependsOn": n["depends_on"],
            "variants": [v["precision"] for v in n["variants"]], "confidenceSource": n["confidence_source"],
            "qualityWeight": n["quality_weight"],
        } for n in g["nodes"]]
        return {
            "nodes": nodes,
            "edges": [{"from": d, "to": n["id"]} for n in nodes for d in n["dependsOn"]],
            "topoOrder": g["topo_order"], "sinks": g["sinks"],
            "policies": POLICIES, "activePolicies": self.policies or DEFAULT_POLICIES,
            "scheduler": {
                "admissionControl": SCHEDULER_FLAGS["enable_admission_control"],
                "precisionPlanning": SCHEDULER_FLAGS["enable_precision_planning"],
                "deadlineCancellation": SCHEDULER_FLAGS["enable_deadline_cancellation"],
                "edfScoring": SCHEDULER_FLAGS["use_edf_scoring"],
                "cpuThreads": max(1, (os.cpu_count() or 2) // 2), "gpuSimThreads": 2,
                "gpuSimOverheadMs": 1.5,
            },
            "samples": [{k: s[k] for k in ("id", "caption", "author", "licence", "source")} for s in self.samples],
        }

    def health(self) -> dict[str, Any]:
        models: dict[str, list[str]] = {}
        if self.ready:
            for n in self.runtime.graph()["nodes"]:
                models[n["id"]] = [v["precision"] for v in n["variants"]]
        return {
            "ok": True, "ready": self.ready, "version": VERSION, "models": models,
            "cpu": f"{self.cpu['model']} ({self.cpu['logicalCpus']} logical CPUs)",
            "uptimeS": round(time.time() - self.started), "busy": self.heavy.locked(),
            "bench": self.bench.get("status"),
        }


def _ts(line: str) -> str:
    m = re.search(r'"ts":"([^"]+)"', line)
    return m.group(1) if m else ""


def find_bench() -> str | None:
    explicit = os.environ.get("LOOMCORE_BENCH")
    if explicit:
        return explicit if Path(explicit).exists() else None
    build = _env_path("LOOMCORE_BUILD_DIR", REPO / "build")
    for config in ("", "Release", "RelWithDebInfo"):
        for name in ("loomcore_bench", "loomcore_bench.exe"):
            candidate = build / "bin" / config / name
            if candidate.exists():
                return str(candidate)
    return None


def run_bench(cpu: dict[str, Any]) -> dict[str, Any]:
    """Runs the repo's own benchmark binary once (benchmarks/latency_bench.cpp,
    the same settings as docs/BENCHMARKS.md) and caches the parsed table."""
    exe = find_bench()
    results = os.environ.get("LOOMCORE_BENCH_RESULTS")
    if results:
        # In the image, <project>/benchmarks/results is a symlink here (the
        # benchmark writes a CSV next to its compiled-in project root).
        os.makedirs(results, exist_ok=True)
    if not exe:
        return {"status": "unavailable", "reason": "loomcore_bench was not found", "cpu": cpu}
    t0 = time.time()
    try:
        proc = subprocess.run([exe, "--warmup", "20", "--iters", "100"], capture_output=True, text=True,
                              timeout=600, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "failed", "reason": str(exc), "cpu": cpu}
    rows = parse_bench_output(proc.stdout)
    if proc.returncode != 0 or not rows:
        return {"status": "failed", "reason": (proc.stderr or proc.stdout)[-400:], "cpu": cpu}
    return {"status": "ready", "ranAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "seconds": round(time.time() - t0, 1), "warmup": 20, "iters": 100, "rows": rows, "cpu": cpu,
            "onnxruntime": "1.30.0", "threadsPerSession": 1}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

LANDING = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>Loomcore runtime</title><style>
body{margin:0;font:15px/1.55 ui-sans-serif,system-ui,sans-serif;background:#f6f6f3;color:#1d2125}
main{max-width:640px;margin:0 auto;padding:48px 20px}h1{font-size:20px;margin:0 0 4px}
p{margin:0 0 14px;color:#4a5157}code{font:13px ui-monospace,Menlo,Consolas,monospace}
table{border-collapse:collapse;width:100%;margin:18px 0}td{padding:6px 0;border-top:1px solid #dcdcd6;
vertical-align:top}td:first-child{width:9.5em}a{color:#1d4e89}
@media (prefers-color-scheme:dark){body{background:#111315;color:#e4e6e8}p{color:#a7adb3}
td{border-color:#2b2f33}a{color:#8db4e8}}</style></head><body><main>
<h1>Loomcore runtime</h1>
<p>This Space runs the real Loomcore C++ runtime (MobileNetV2, a confidence gate and bert_tiny scheduled as a
graph) behind a small JSON API. The console that drives it and draws what it did is at
<a href="https://loomcore.amittal.dev">loomcore.amittal.dev</a>.</p>
<table>
<tr><td><code>GET /health</code></td><td>liveness, version, models, CPU</td></tr>
<tr><td><code>GET /graph</code></td><td>the graph, the router policy chain, scheduler flags</td></tr>
<tr><td><code>POST /run</code></td><td>one job on a sample or an uploaded image</td></tr>
<tr><td><code>POST /load</code></td><td>N concurrent jobs: latency, batches, rejects, cancels</td></tr>
<tr><td><code>POST /reload</code></td><td>hot-swap the graph under load; jobs lost must be 0</td></tr>
<tr><td><code>GET /bench</code></td><td>FP32 vs INT8 on this machine</td></tr>
</table>
<p>Source: <a href="https://github.com/armaanmittalweb/loomcore">github.com/armaanmittalweb/loomcore</a></p>
</main></body></html>"""


def create_app(live: LiveRuntime | None = None, start: bool = True):
    live = live or LiveRuntime()
    limiter = RateLimiter()
    proxy_key = os.environ.get("LOOMCORE_PROXY_KEY", "")
    app = FastAPI(title="Loomcore live", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.live = live

    if start and not live.ready:
        threading.Thread(target=live.start, name="loomcore-start", daemon=True).start()

    def client_ip(request: Request) -> str:
        # Behind the Cloudflare Tunnel, Cloudflare sets CF-Connecting-IP to the
        # visitor's address (and the server only listens on 127.0.0.1, so
        # nothing else can reach it to forge the header).
        # On Modal, the loomcore-api Worker forwards each request and passes the visitor's
        # address in X-Client-IP, vouched for by the key the two share.
        if proxy_key and hmac.compare_digest(request.headers.get("x-proxy-key", ""), proxy_key):
            relayed = request.headers.get("x-client-ip", "").strip()
            if relayed:
                return relayed
        cf = request.headers.get("cf-connecting-ip", "").strip()
        forwarded = request.headers.get("x-forwarded-for", "")
        return cf or forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")

    def error(status: int, code: str, message: str, headers: dict | None = None, **extra: Any):
        return JSONResponse({"error": code, "message": message, **extra}, status_code=status, headers=headers)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
        post = request.method == "POST"
        wait = limiter.check(client_ip(request), "post" if post else "get",
                             POSTS_PER_MINUTE if post else GETS_PER_MINUTE)
        if wait:
            return error(429, "rate_limited", f"at most {POSTS_PER_MINUTE} runs a minute; try again shortly",
                         headers={"Retry-After": str(math.ceil(wait))}, retryAfter=round(wait, 1))
        if post:
            length = request.headers.get("content-length")
            if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES + 64 * 1024:
                return error(413, "too_large", "the request must be 2 MB or smaller")
        return await call_next(request)

    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                       allow_headers=["content-type"], max_age=600)

    async def offload(fn, timeout: float):
        try:
            return await asyncio.wait_for(asyncio.to_thread(fn), timeout=timeout)
        except asyncio.TimeoutError:
            raise ApiError(504, "timeout", "the runtime took too long to answer") from None

    async def respond(fn, timeout: float):
        try:
            return JSONResponse(await offload(fn, timeout))
        except Busy as exc:
            return error(429, "busy", "the runtime is busy with another run; try again shortly",
                         headers={"Retry-After": str(math.ceil(exc.retry_after))}, retryAfter=exc.retry_after)
        except ApiError as exc:
            headers = {"Retry-After": str(exc.extra["retryAfter"])} if "retryAfter" in exc.extra else None
            return error(exc.status, exc.code, exc.message, headers=headers, **exc.extra)

    async def body_of(request: Request) -> tuple[dict[str, Any], bytes | None]:
        ctype = request.headers.get("content-type", "")
        if ctype.startswith("multipart/form-data"):
            form = await request.form(max_files=1, max_fields=8)
            fields: dict[str, Any] = {k: v for k, v in form.items() if isinstance(v, str)}
            upload = form.get("image")
            data = None
            if upload is not None and not isinstance(upload, str):
                data = await upload.read(MAX_UPLOAD_BYTES + 1)
                if len(data) > MAX_UPLOAD_BYTES:
                    raise ApiError(413, "too_large", "the image must be 2 MB or smaller")
            if "policies" in fields:
                raw = fields["policies"]
                fields["policies"] = json.loads(raw) if raw.startswith("[") else raw
            return fields, data
        raw = await request.body()
        if not raw:
            return {}, None
        try:
            body = json.loads(raw)
        except ValueError:
            raise ApiError(400, "bad_json", "the body must be JSON or multipart form data") from None
        if not isinstance(body, dict):
            raise ApiError(400, "bad_json", "the body must be a JSON object")
        return body, None

    @app.get("/", response_class=HTMLResponse)
    async def landing():
        return HTMLResponse(LANDING)

    @app.get("/health")
    async def health():
        return JSONResponse(live.health(), headers={"Cache-Control": "no-store"})

    @app.get("/graph")
    async def graph():
        if not live.ready:
            return error(503, "warming_up", "the runtime is still loading", retryAfter=5)
        return JSONResponse(live.graph())

    @app.get("/bench")
    async def bench():
        return JSONResponse(live.bench)

    @app.post("/run")
    async def run(request: Request):
        try:
            body, upload = await body_of(request)
            budget = parse_budget(body.get("timeBudgetMs"))
            policies = parse_policies(body.get("policies"))
            tensor, source = live.inputs_for(body.get("sample"), upload)
        except ApiError as exc:
            return error(exc.status, exc.code, exc.message, **exc.extra)
        return await respond(lambda: live.run(tensor, source, budget, policies), RUN_TIMEOUT_S)

    @app.post("/load")
    async def load(request: Request):
        try:
            body, _ = await body_of(request)
            jobs = parse_int(body.get("jobs"), "jobs", 1, 64, default=16)
            concurrency = parse_int(body.get("concurrency"), "concurrency", 1, 16, default=4)
            budget = parse_budget(body.get("timeBudgetMs"))
            policies = parse_policies(body.get("policies"))
            tensor, source = live.inputs_for(body.get("sample"), None)
        except ApiError as exc:
            return error(exc.status, exc.code, exc.message, **exc.extra)
        return await respond(lambda: live.load(jobs, concurrency, budget, policies, tensor, source), HEAVY_TIMEOUT_S)

    @app.post("/reload")
    async def reload(request: Request):
        try:
            body, _ = await body_of(request)
            jobs = parse_int(body.get("jobs"), "jobs", 8, 64, default=32)
            swaps = parse_int(body.get("swaps"), "swaps", 1, 6, default=3)
        except ApiError as exc:
            return error(exc.status, exc.code, exc.message, **exc.extra)
        return await respond(lambda: live.reload(jobs, swaps), HEAVY_TIMEOUT_S)

    return app


def main() -> None:
    if sys.platform == "win32":
        # See bindings/python/tests/conftest.py: a native llvm-mingw build's
        # libc++ needs the C locale to keep its ctype table valid.
        locale.setlocale(locale.LC_CTYPE, "C")
        # Windows sleeps in 15.6 ms ticks by default, which would turn
        # GPU_SIM's modelled 1.5 ms launch overhead into ~15 ms. Ask for 1 ms
        # timer resolution, as latency-sensitive Windows programs do; Linux
        # (the Space) needs nothing.
        import ctypes  # noqa: PLC0415

        ctypes.windll.winmm.timeBeginPeriod(1)
    import uvicorn  # noqa: PLC0415

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(create_app(), host=os.environ.get("HOST", "0.0.0.0"), port=port, log_level="info",
                timeout_keep_alive=10)


if __name__ == "__main__":
    main()
