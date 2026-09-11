# Loomcore Architecture

Loomcore is a C++ runtime that loads several ONNX models as nodes in a
dependency graph, schedules inference across them with priority and
dynamic batching, and uses a small pluggable "router" layer to decide —
at run time, per job — which model variant, precision, and simulated
backend to actually use. This document is the deep dive; `README.md` is
the tour.

```
   ┌──────────────┐        ┌─────────────────────────────────────────┐
   │ Runtime      │  load  │ Graph (topology)                        │
   │ (facade)     │───────▶│  NodeConfig[]: id, deps, variants, ...  │
   └──────┬───────┘        └─────────────────────────────────────────┘
          │ submit(inputs)
          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ Scheduler                                                     │
   │  per job: dispatch ready nodes → ask Router → pick variant   │
   │           → enqueue on a BackendLane → batch → ModelVariant  │
   │           ::run() → fan out to dependents                    │
   │                                                                │
   │   ┌────────────┐        ┌────────────┐                       │
   │   │ CPU lane   │        │ GPU_SIM lane│   (see "Simulated     │
   │   │ N workers  │        │ 2 workers   │    backends" below)   │
   │   └────────────┘        └────────────┘                       │
   └──────────────────────────────────────────────────────────────┘
          │                              ▲
          ▼                              │
   ┌──────────────┐   RoutingContext  ┌──┴─────────────┐
   │ MetricsRegistry│◀─────────────── │ IRoutingPolicy  │
   │ (p50/p95, queue│  RoutingDecision│ (Composite of   │
   │  depth)        │────────────────▶│  small policies)│
   └──────────────┘                   └────────────────┘
          ▲
          │ every event
   ┌──────┴───────┐
   │ Logger        │  JSON-lines: job/node lifecycle, routing
   │ (JSON lines)  │  decisions, batch sizes, latency
   └──────────────┘
```

## Design principle: config describes topology, code describes data flow

A DAG node's *shape* (id, dependencies, backend, batching, which ONNX
files back its FP32/INT8 variants) is declarative and lives in JSON
(`examples/graph_config.json`). What a node's inputs actually *mean* —
"take mobilenet's argmax class index, look up its label, tokenize
`"a photo of a {label}"`" — cannot be expressed in JSON without
reinventing a scripting language, so it lives in code as a
`NodeInputBinder`:

```cpp
using NodeInputBinder =
    std::function<std::vector<NamedTensor>(const NodeExecutionContext&)>;
```

`Runtime::loadGraph` takes the JSON path *and* a `map<node_id, NodeInputBinder>`
supplied by the caller (see `examples/run_example.cpp`, or
`bindings/python/loomcore_py.cpp`'s adapter for the Python-callable
equivalent). This split is why the graph config schema below has no
notion of "wiring" beyond `depends_on` and `confidence_source`.

## Graph config schema

```jsonc
{
  "log_file": "logs/loomcore.jsonl",     // optional; also settable via RuntimeOptions
  "scheduler": {                          // optional; all fields optional
    "cpu_threads": 0,                     // 0 = hardware_concurrency() / 2
    "gpu_sim_threads": 2,
    "gpu_sim_fixed_overhead_ms": 1.5,
    "gpu_sim_bytes_per_ms": 250000.0
  },
  "nodes": [
    {
      "id": "mobilenet",                  // unique
      "backend": "CPU",                   // "CPU" | "GPU_SIM"
      "priority": 5,                      // higher runs first, all else equal
      "max_batch_size": 4,                // 1 disables cross-job batching
      "batch_window_ms": 8,               // 0 = dispatch as soon as the lane is free
      "depends_on": [],                   // node ids this one needs outputs from
      "confidence_source": "mobilenet",   // optional: see ConfidenceGatePolicy
      "variants": [
        { "precision": "FP32", "model_path": "models/mobilenetv2.onnx" },
        { "precision": "INT8", "model_path": "models/mobilenetv2.int8.onnx" }
      ]
    }
  ]
}
```

A `Runtime::run()`/`submit()` call's `JobResult` contains only **sink**
node outputs (nodes with no dependents, i.e. `Graph::sinkNodes()`) — an
intermediate node like `mobilenet` above, once something depends on it,
never appears in the result even though it executed. Consumers that want
an intermediate value for logging/debugging capture it themselves inside
a binder or `ConfidenceExtractor` closure (see `run_example.cpp`'s
`DemoSideChannel`).

## Graph / topological execution

`Graph` (`include/loomcore/graph.h`) is a thin, validated adjacency
structure: `addNode`, then `validate()` checks every declared dependency
exists and (via Kahn's algorithm in `topoOrder()`) that the graph is
acyclic. A `Scheduler` doesn't actually need a single global topological
order to execute correctly — it runs a proper *dataflow* schedule instead
(see below) — but `topoOrder()`/`validate()` exist so `Runtime::loadGraph`
can fail fast with a clear error before any ONNX session is loaded.

## Scheduler

One **job** is a single end-to-end run of the graph for one set of
external inputs. Multiple jobs may be in flight concurrently. For each
job, `Scheduler` maintains:

- `remaining_deps[node]` — an atomic per-node countdown, initialized to
  that node's `depends_on.size()`.
- `node_outputs[node]` — a mutex-guarded map of completed outputs.
- `remaining_nodes` — an atomic countdown over *all* graph nodes (skipped
  nodes count too); the job's promise is fulfilled when it hits zero.

Nodes with no dependencies dispatch immediately; whenever a node finishes,
every dependent's counter is decremented, and any that reach zero dispatch
next. This is ordinary dataflow scheduling, not "compute a topo order and
walk it" — which matters once dynamic batching and skip decisions are in
play, since two jobs can be at completely different points in the graph
at the same instant.

**Dispatching a node** (`Scheduler::Impl::dispatchNode`) does, in order:

1. Build a `RoutingContext` (job id, the node's config, current CPU/GPU_SIM
   queue depths from `MetricsRegistry`, remaining time budget, and — if
   the node declares a `confidence_source` — that upstream node's raw
   output tensors).
2. Ask the configured `IRoutingPolicy` for a `RoutingDecision` (see
   "Router" below). Log it if it says anything.
3. If `skip`, record an empty output for the node and treat it as
   finished — its dependents still dispatch normally.
4. Otherwise resolve which `ModelVariant` to run (the decision's
   precision override, or FP32 by default) and which backend lane.
5. Snapshot the job's current inputs/upstream outputs under lock, call the
   node's `NodeInputBinder` *outside* the lock, and reorder its result to
   match the model's real input names when possible
   (`reorderToExpected` in `scheduler.cpp` — binders may name their
   output tensors after the model's real inputs for robustness, or just
   return them in the right positional order; either works).
6. Enqueue a `NodeTask` on the chosen `BackendLane`.

**Dynamic batching** happens inside `BackendLane` (an implementation
detail of `scheduler.cpp`, not a public type): each lane keeps a
`pending_by_(node_id, precision)` queue and a small "ready" list of keys
that have queued work. A worker thread picks the ready key with the
highest *aged* priority (`priority + waited_ms / aging_ms_per_priority_point`
— simple starvation avoidance), waits up to `batch_window_ms` for more
same-key requests to arrive if it isn't at `max_batch_size` yet, then
concatenates every queued request's inputs along axis 0 into one
`ModelVariant::run()` call and slices the batched output back apart for
each caller.

This is a deliberately simple design in a couple of specific places,
called out because a "no shortcuts, but no false sophistication either"
approach means being explicit about them:

- **Linear scan over ready keys**, not an indexed priority heap. At DAG
  sizes in the tens of distinct (node, precision) pairs this costs
  nothing measurable and is far easier to read/verify than a
  heap-with-live-comparator.
- **A busy-poll wait during the batch window** (release the lock, sleep a
  couple of milliseconds, re-check), rather than a per-key timer. Simple
  and correct; a production system serving many nodes would want
  per-key timers instead.
- **Batch latency is attributed to every item in the batch**, not divided
  by batch size — `MetricsRegistry` records the same wall-clock number for
  each of N batched callers. This is a conservative (slightly pessimistic
  for large batches) choice, stated explicitly here and in
  `docs/BENCHMARKS.md` rather than silently baked into the numbers.

## Simulated backends

There is no physical GPU anywhere in this project's reference
environment. `Backend::GPU_SIM` is a **second worker pool with a
different, deliberately-injected cost model** — not a fake label on the
same CPU execution path:

- Its lane defaults to far fewer worker threads (2 vs. `hardware_concurrency() / 2`),
  modeling a real discrete GPU's more limited number of concurrent
  execution streams compared to a many-core CPU.
- After the real `Ort::Session::Run()` call, `BackendLane::executeBatch`
  adds `gpu_sim_fixed_overhead_ms + total_bytes / gpu_sim_bytes_per_ms` of
  actual `sleep_for` — a launch/sync-overhead-plus-bandwidth model of
  host↔device transfer, so the batch's *measured* latency is representative
  of what routing to a transfer-bound accelerator would look like.

Nowhere does GPU_SIM claim to run *faster* than CPU or use any actual
GPU hardware or API. It exists so `LoadAwareBackendPolicy` and the
scheduler's per-lane queue-depth metrics have two backends with genuinely
different cost profiles to route across — which is what the project's
"simulate heterogeneous backends (CPU/GPU-style dispatch even if you only
have CPU to test on)" requirement asks for, made honest about exactly
what is and isn't simulated.

## Router

"Agentic" here means a small, inspectable decision layer that looks at
*runtime* state — recent latency, queue depth, remaining time budget,
upstream confidence — rather than a fixed static plan, to decide which
concrete model variant / backend / branch to invoke next. It echoes the
policy-router pattern (a chain of small, named, independently testable
policies, each free to decline and defer to the next) used for
model/path selection in the author's PRISM-Home project.

```cpp
class IRoutingPolicy {
public:
    virtual std::string name() const = 0;
    virtual std::optional<RoutingDecision> decide(const RoutingContext&) const = 0;
};
```

Returning `std::nullopt` means "no opinion" — `CompositeRouter` runs every
policy in its list and merges the first non-null `precision`/`backend`
each contributes; a `skip = true` decision is terminal (later policies
don't get a vote on whether to run a node that's already being skipped).
Every decision that says anything carries a `reason` string and gets
logged as a `routing_decision` event, so routing is auditable rather than
a black box.

Three built-in policies ship in `loomcore/router.h`:

| Policy | Reads | Decides |
|---|---|---|
| `LatencyBudgetPolicy(threshold_ms)` | job's remaining time budget, whether the node has an INT8 variant | fall back to INT8 once the deadline is close (wires the quantization milestone into live decision-making, not just a benchmark) |
| `LoadAwareBackendPolicy` | CPU vs GPU_SIM queue depth | route to whichever simulated lane is shallower |
| `ConfidenceGatePolicy(skip_above)` | a `ConfidenceExtractor` applied to a designated upstream node's output | skip a downstream node once upstream is already confident enough (used by the reference pipeline to skip `bert_tiny` once `mobilenet` doesn't need disambiguating) |

All three, plus `CompositeRouter` itself, are unit-tested in isolation
(`tests/test_router.cpp`) and exercised end-to-end against a real
scheduler + real ONNX sessions in `tests/test_scheduler.cpp`.

## ModelVariant / ModelNode

`ModelVariant` (`include/loomcore/model_node.h`) wraps exactly one loaded
`Ort::Session` for one (node, precision) pair: it introspects the model's
real input/output names at load time and exposes `run()` taking/returning
plain `loomcore::NamedTensor`s — nothing in this public header, or in
`Runtime`/`Graph`/`Scheduler`, requires a consumer to include or link the
ONNX Runtime C++ API directly. That dependency is fully encapsulated
behind `loomcore::Environment` (a process-wide `Ort::Env` handle exposed
only as an opaque pointer) and `model_node.cpp`/`environment.cpp`, which
are the only two translation units that `#include <onnxruntime_cxx_api.h>`.
`benchmarks/latency_bench.cpp` demonstrates this: it measures FP32 vs
INT8 latency using `loomcore::ModelVariant` directly, with no ONNX
Runtime include or link dependency of its own.

A `ModelNode` is the loaded counterpart of a `NodeConfig`: its `config`
plus a `map<Precision, shared_ptr<ModelVariant>>` — one entry per variant
declared in the JSON config.

## Quantization

MobileNetV2 (a CNN, cost dominated by convolutions) is **statically**
quantized: `scripts/quantize_mobilenet.py` runs `quant_pre_process` then
`quantize_static` with `QuantFormat.QDQ`, calibrated against synthetic
random activations (accuracy isn't this project's concern — see
`docs/BENCHMARKS.md`'s first paragraph). bert_tiny (a small transformer,
cost dominated by MatMul/Gemm) is **dynamically** quantized via
`quantize_dynamic`, the standard technique for transformer encoders,
where per-inference activation ranges make static calibration less
appropriate. Both choices are deliberate, not "quantize everything the
same way" — see `docs/BENCHMARKS.md` for what actually resulted from each.

MobileNetV2's public ONNX Model Zoo export also declares a **static**
batch size of 1 on its `data` input/output even though every op in the
graph is batch-agnostic (`scripts/patch_mobilenet_dynamic_batch.py`
relaxes this to a symbolic dimension so Loomcore's cross-job batching has
something real to batch); bert_tiny is exported directly with dynamic
batch and sequence-length axes (`scripts/export_bert_tiny.py`), so it
needed no such patch.

## Tokenizer scope

`loomcore::WordPieceTokenizer` (`src/tokenizer.cpp`) is a small,
from-scratch WordPiece implementation — no ICU, no Hugging Face
`tokenizers` dependency — covering the case the reference pipeline
actually needs: lowercase, mostly-ASCII English phrases like "a photo of
a golden retriever". It does **not** implement full Unicode
normalization, CJK character splitting, or accent stripping the way the
reference Python tokenizer does; anything outside that scope falls back
to `[UNK]` rather than silently misbehaving. This is stated here plainly
rather than left to be discovered: it is a legitimate simplification, not
an accidental limitation.

## Observability

Every scheduling decision and every node's latency is logged as one JSON
object per line (`loomcore::Logger`, JSON via a vendored nlohmann/json
single header) to an optional file and/or stdout, plus kept in a bounded
in-memory ring buffer (`Logger::recentLines`) so tests and the Python
bindings can inspect recent activity without re-parsing a log file. Event
types: `job_submitted`, `routing_decision`, `node_scheduled`,
`batch_flushed`, `node_completed`, `node_skipped`, `job_completed`,
`error`. `MetricsRegistry` is the separate, complementary piece: a
rolling per-node latency window (p50/p95/mean) and per-lane queue depth,
which is what `RoutingContext` and the router policies actually read —
Logger is an append-only record of *what happened*; MetricsRegistry is
the *current state* the router reasons about.

## Python bindings

`bindings/python/loomcore_py.cpp` is a pybind11 extension. Tensors cross
the boundary as numpy arrays (`tensorFromNumpy`/`tensorToNumpy`); node
binders and confidence extractors are plain Python callables, wrapped in
a `NodeInputBinder`/`ConfidenceExtractor` lambda that acquires the GIL for
the duration of each call — necessary because these run on whatever
`BackendLane` worker thread is dispatching that node, never the thread
that called `Runtime.run()` from Python.

A Python class can also subclass `loomcore.RoutingPolicy` (a pybind11
trampoline over `IRoutingPolicy`) and implement `decide(self, ctx)` —
Milestone 4's "basic router logic", made callable from either language
with the same C++ scheduler underneath (see `examples/run_example.py`'s
`SkipIfConfident`). One real, non-obvious bug surfaced building this and
is worth recording: a Python policy object passed as a bare temporary
into `CompositeRouter.add(...)` (`router.add(MyPolicy())`, no surviving
Python-side reference) had its Python wrapper garbage-collected as soon
as `add()` returned, even though the C++ `shared_ptr<IRoutingPolicy>`
kept the underlying object's memory alive — so `get_override()` later
found no Python object to call into, and `decide()` fell through to its
"you must implement this" error. The fix is `py::keep_alive<1, 2>()` on
`CompositeRouter::add` (and `py::keep_alive<1, 5>()` on
`Runtime::load_graph`'s `router` parameter for the same reason), tying
the policy's Python lifetime to its container's.
