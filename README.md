# Loomcore

**A C++ runtime that loads and schedules multiple ONNX models concurrently
as a dependency graph — not one model, one inference — with an agentic
routing layer deciding which model to invoke at runtime, deadlines it can
actually enforce, and a graph it can hot-swap under live traffic with
zero jobs dropped.**

Loomcore exists to demonstrate a multi-model execution *orchestrator*, not
a model. Two small, off-the-shelf, un-trained ONNX models
(MobileNetV2 and a tiny BERT variant) are wired into a real dependency
graph purely as a vehicle for the orchestration layer: a DAG scheduler
with priority + dynamic batching across simulated heterogeneous backends,
a pluggable runtime router that picks which model variant/precision/path
to invoke per job, a quantized inference path with a measured FP32
vs. INT8 latency comparison, admission control and real deadline
cancellation, a DAG-wide precision-downgrade planner (a genuine 0/1
knapsack, not a dressed-up greedy), a circuit breaker and bulkhead for
node-level resilience, and an RCU-style graph hot-swap proven under
continuous concurrent load. **[docs/CLAIMS.md](docs/CLAIMS.md) is the
short version of this README: every claim here, next to the exact
command that checks it yourself.**

```
Image ──▶ [mobilenet: FP32/INT8, CPU]──▶ argmax + label ──▶ [bert_tiny: FP32/INT8, GPU_SIM]──▶ 128-dim text embedding
              │                                                        ▲
              └── confidence ──────────────────────────────────────────┘
                  (skip bert_tiny once mobilenet is already confident — a live router decision, not a static graph)
```

## Live

**[loomcore.amittal.dev](https://loomcore.amittal.dev)** is a console for the real runtime: it opens
on a recorded run and draws it as a timeline from the runtime's own trace (one track per backend
lane, a bar per ONNX Runtime call coloured by precision, batches, router decisions with their
reasons, DAG edges, each job's life), then lets you run one job on a sample or your own image, run
a load test, or hot-swap the graph under load, with the router policies toggled as you like.

- **Runtime:** an Oracle Cloud Always Free Arm VM (Ampere A1, Neoverse N1, 4 OCPU / 24 GB, always
  on) behind a Cloudflare Tunnel at `https://loomcore-api.amittal.dev`. The VM builds
  `space/Dockerfile` natively (linux/arm64; it builds the same on x86_64): it clones this repo,
  prepares the models, builds and tests exactly as CI's Linux job does (the image fails to build
  unless `ctest` passes), runs the binding and server tests, and serves `space/server.py`, a small
  FastAPI layer over the Python bindings, on `127.0.0.1:7860` only. API:
  [space/README.md](space/README.md); VM steps: [deploy/oracle](deploy/oracle/README.md).
- **Console:** `web/` (Vite, TypeScript, Preact), deployed to Vercel with `web/` as the root.
- **The recorded run.** The console never opens on a spinner: it renders real results captured
  from the runtime (`web/src/recorded/*.json`, written by `npm run record` against a running
  server, labelled with the date and machine) while it connects to the live runtime, then unlocks
  the controls. If the runtime is down or restarting, the recorded run stays and the status bar
  says so. Every panel says whether it shows recorded or live data.

## What's actually in here

| Milestone | Where |
|---|---|
| 1. Single-model load + inference via the ONNX Runtime C++ API, CMake on both OSes | `include/loomcore/model_node.h`, `src/model_node.cpp`, `CMakeLists.txt` |
| 2. DAG scheduler, 2+ models, dependency ordering | `include/loomcore/{graph,scheduler}.h`, `src/{graph,scheduler}.cpp` |
| 3. Quantized path + FP32 vs INT8 benchmark | `scripts/quantize_*.py`, `benchmarks/latency_bench.cpp`, [docs/BENCHMARKS.md](docs/BENCHMARKS.md) |
| 4. Python bindings + basic router logic | `bindings/python/`, `examples/run_example.py` |

Plus, because the brief asked for the whole system, not just the four
milestones: a chain-of-responsibility **router** with seven independently
unit-tested policies — the original three (latency-budget precision
fallback, load-aware backend balancing, confidence-gated skip), plus a
DAG-wide precision planner, a circuit breaker, and a bulkhead (see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#router)); **structured,
asynchronous JSON-lines logging** of every scheduling decision and
per-node latency, off the scheduler's own lane threads; a from-scratch
**WordPiece tokenizer** (no NLP dependency) so the MobileNetV2 →
bert_tiny handoff is a real, meaningful pipeline instead of two unrelated
models bolted together; a from-scratch **JPEG/PNG decode → resize →
normalize** path (via stb_image) so the example runs on an actual photo,
not just synthetic tensors; a **C API** consumable from plain C, and a
**Perfetto trace exporter** that turns any run's log into a timeline you
can drop onto `ui.perfetto.dev`.

Beyond the original brief, four further capabilities were added
end-to-end — real feature, real tests, real docs, not a stub:

- **Deadlines that bind, not just advise**: admission control that
  rejects a job outright when it's judged undeliverable, real
  cancellation of an in-flight ONNX Runtime call once a budget expires,
  and a knapsack-based DAG-wide precision-downgrade plan (see
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#deadlines-and-resilience)).
- **Resilience policies** — a proper three-state circuit breaker and a
  per-node concurrency bulkhead — that needed zero scheduler changes to
  add, only two small `MetricsRegistry` extensions.
- **Zero-downtime graph hot-swap**: `Runtime::reloadGraph` swaps the
  entire graph/model/scheduler set under live traffic with no job ever
  dropped, delayed, or handed torn state (see
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#hot-reloading-a-graph) and
  `examples/reload_demo.cpp`).
- **A real installable package**: `cmake --install` plus an exported
  CMake config, verified by a from-scratch `find_package(Loomcore)`
  consumer in `examples/consumer/` that never touches this repo's source
  tree.

## Architecture, in one paragraph

`Runtime` loads a JSON graph config plus caller-supplied `NodeInputBinder`
callbacks (config describes topology; code describes data flow — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#design-principle-config-describes-topology-code-describes-data-flow)
for why) into a `Graph` of `NodeConfig`s and loaded `ModelVariant`s (one
loaded `Ort::Session` per node per precision, entirely encapsulated
behind `loomcore::Environment` — nothing in Loomcore's public headers
requires a consumer to include or link the ONNX Runtime C++ API
directly). `Scheduler` runs one **job** per `submit()` call as a proper
dataflow graph: nodes dispatch as their dependencies complete, each
dispatch consults the pluggable `IRoutingPolicy` for a
skip/precision/backend decision, and same-(node, precision) requests
across concurrent jobs get dynamically batched on one of two simulated
backend lanes (`CPU`, and `GPU_SIM` — a second worker pool with an
injected transfer-latency cost model, since there's no physical GPU here
to actually dispatch to; see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#simulated-backends)). Every
decision and every node's latency is logged as JSON-lines and fed into a
rolling `MetricsRegistry` the router itself reads. Full diagram and
design tradeoffs: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

```
pip install -r scripts/requirements.txt
python scripts/prepare_all_models.py        # downloads/exports/quantizes both models (~1-2 min)

cmake -S . -B build -G "Visual Studio 17 2022" -A x64      # or: -G Ninja on Linux/macOS
cmake --build build --config RelWithDebInfo --parallel

ctest --test-dir build -C RelWithDebInfo --output-on-failure   # 47 test cases / 444 assertions

./build/bin/RelWithDebInfo/loomcore_example examples/sample.jpg
./build/bin/RelWithDebInfo/loomcore_bench
./build/bin/RelWithDebInfo/loomcore_reload_demo      # hot-swaps the graph 6x under continuous load
./build/bin/RelWithDebInfo/loomcore_trace_export logs/loomcore.jsonl trace.json   # drop onto ui.perfetto.dev

export LOOMCORE_BUILD_DIR=$PWD/build
python examples/run_example.py
```

Full instructions, options, and troubleshooting:
[docs/BUILD.md](docs/BUILD.md). Every claim above, as a runnable command:
[docs/CLAIMS.md](docs/CLAIMS.md).

### What you'll see

`loomcore_example` runs the real pipeline end to end, prints the
predicted ImageNet class + confidence and the resulting text embedding,
then fires 8 concurrent jobs to exercise batching, and prints measured
per-node p50/p95 latency:

```
Loaded image: examples/sample.jpg
=== Single synchronous run ===
  [bert_tiny] embedding text: "a photo of a front curtain"
Predicted class : front curtain (confidence 0.350467)
Text embedding  : [-0.999991, 0.00937352, ...] (dim 128)

=== Concurrent load: 8 jobs submitted at once (exercises batching + priority) ===
mobilenet  p50=53.0022ms  p95=76.2338ms  n=9
bert_tiny  p50=14.9707ms  p95=15.9969ms  n=9
```

(`examples/sample.jpg` is a procedurally generated test pattern, not a
real photo — pass your own image path for a meaningful classification.
`n=9` because the concurrent batch of 8 plus the earlier synchronous run
share the same rolling metrics window.)

Every run also writes `logs/loomcore.jsonl`, one JSON object per
scheduling event:

```json
{"event":"routing_decision","job_id":"job-14","node_id":"a","message":"CompositeRouter: LatencyBudgetPolicy: remaining budget 0.969100ms < threshold 1000.000000ms; downgrading to INT8 for node 'a'", ...}
{"event":"batch_flushed","node_id":"mobilenet","backend":"CPU","precision":"FP32","batch_size":4,"latency_ms":9.87, ...}
```

## Repository layout

```
include/loomcore/     Public C++ API (Runtime, Graph, Scheduler, Router, ModelNode, Planner, Logger, Metrics, Tokenizer, ...)
                       plus c_api.h — a second, independent extern "C" surface (see docs/ARCHITECTURE.md "C API")
src/                   Implementation of the above — builds into loomcore_core (.dll / .so)
bindings/python/       pybind11 extension + thin Python package (see bindings/python/README.md)
bindings/c/            Plain-C smoke test proving c_api.h is a real, separately-consumable surface
examples/              graph_config.json, the C++ and Python reference-pipeline demos, the hot-swap
                       demo (reload_demo.cpp), and a from-scratch find_package(Loomcore) consumer
tools/                 loomcore_trace_export: JSONL log -> Perfetto/Chrome Trace Event Format
benchmarks/            FP32 vs INT8 latency benchmark
tests/                 doctest unit + scheduler-integration tests (hermetic: tiny fixture ONNX graphs, no downloads)
scripts/               Model download/export/quantization pipeline (Python, build-time only)
docs/                  ARCHITECTURE.md, BUILD.md, BENCHMARKS.md, CLAIMS.md, live/ (the live console's brief)
space/                 The Hugging Face Space: Dockerfile, FastAPI server over the bindings, tests, samples
web/                   The console at loomcore.amittal.dev
```

## Why these design choices

- **Two reference models, not trained, chosen for shape diversity.**
  MobileNetV2 (fixed-size image tensor, CNN) and bert_tiny
  (variable-length token sequence, transformer) exercise genuinely
  different tensor shapes, dtypes (float32 vs. int64), and quantization
  strategies (static QDQ vs. dynamic) through the same orchestration
  code path — proving the scheduler and router are generic, not
  special-cased for one model shape.
- **The DAG dependency is semantically real**, not contrived: bert_tiny
  embeds the *text description of mobilenet's own prediction*, so
  `NodeExecutionContext::upstreamFirst()` carries data that actually
  flows through the pipeline, and `ConfidenceGatePolicy` skipping
  bert_tiny is a meaningful decision (don't bother describing what
  you're already sure of), not a toy branch.
- **Everything simulated is labeled as simulated.** GPU_SIM never claims
  to be a real GPU; the INT8 calibration data is synthetic and the
  benchmark says so; the tokenizer's scope limitations are documented
  next to the code, not discovered by a user later.
- **The ORT dependency is fully encapsulated** behind
  `loomcore::Environment`/`ModelVariant`, so the benchmark and any future
  consumer never need to touch ONNX Runtime headers directly — a real
  library-boundary decision, not just "it compiles."
- **New scheduler behavior ships opt-in, not on by default.** Admission
  control, deadline cancellation, precision planning, and EDF lane
  scoring are all `SchedulerConfig` flags defaulting to `false`: layering
  genuinely new, more invasive machinery onto a scheduler that already
  worked should never silently change what an existing caller observes.
  See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#deadlines-and-resilience).
- **A concurrency bug found by testing under real load, not asserted
  away.** The first draft of the graph hot-swap passed a naive
  "reload-then-assert" test and crashed the moment a stress test kept
  submitting jobs *through* a reload — a lane worker thread ended up
  joining itself. The fix (a deferred-teardown "graveyard" thread) and
  the test that caught it are both described in
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#hot-reloading-a-graph) —
  left in as the honest record of what real concurrent testing is for,
  not smoothed over.

## License

MIT — see [LICENSE](LICENSE).
