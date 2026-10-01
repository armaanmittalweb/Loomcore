# Loomcore Architecture

Loomcore is a C++ runtime that loads several ONNX models as nodes in a
dependency graph, schedules inference across them with priority and
dynamic batching, and uses a small pluggable "router" layer to decide —
at run time, per job — which model variant, precision, and simulated
backend to actually use. Deadlines can be made load-bearing rather than
advisory (admission control, real cancellation, a DAG-wide precision
plan — see "Deadlines and resilience"), and the whole graph can be
hot-swapped under live traffic with zero jobs dropped (see
"Hot-reloading a graph"). This document is the deep dive; `README.md` is
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
      "quality_weight": 1.0,              // optional, default 1.0: see PrecisionPlanner
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
- `node_outputs[node]` — a mutex-guarded map of completed outputs. Values
  are `shared_ptr<const vector<NamedTensor>>`, not plain vectors: a node
  with several dependents is snapshotted once per dependent dispatch (see
  step 5 below), and sharing the pointer instead of deep-copying the
  tensor data turns that from an O(bytes produced so far) copy into an
  O(number of upstream nodes) pointer copy. `NodeExecutionContext::upstream_outputs`
  (`loomcore/model_node.h`) is typed to match, so binders never see the
  difference.
- `remaining_nodes` — an atomic countdown over *all* graph nodes (skipped
  nodes count too).
- `settled` — a single atomic flag, separate from `failed` below, that
  arbitrates who is allowed to call `promise.set_value()` /
  `promise.set_exception()`. A `std::promise` may only be settled once;
  a second attempt throws `std::future_error` with no caller frame above
  a lane worker thread to catch it, i.e. `std::terminate`. Two different
  code paths can each independently believe they're the one settling this
  job — `onNodeFinished`, when `remaining_nodes` hits zero, and `failJob`,
  called from a node's own failure *or* the deadline reaper below — and
  once `enable_deadline_cancellation` is on, these two
  genuinely can fire at the same instant (the reaper's periodic budget
  check has no knowledge of a node that happens to be completing
  successfully right then). Both paths `compare_exchange_strong` on
  `settled` before touching `promise`; the loser returns without touching
  it. `failed` remains a separate, best-effort, may-be-set-more-than-once
  flag — cheap to check at the top of `dispatchNode`/`onNodeFinished` to
  stop doing further work for a job that's already going to fail, but it
  is not itself the arbiter of anything.

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
`pending_by_(node_id, precision, shape)` queue and a small "ready" list of
keys that have queued work. A worker thread picks the ready key with the
highest *aged* priority (`priority + waited_ms / aging_ms_per_priority_point`
— simple starvation avoidance; see "Deadlines and resilience" below for the
earliest-deadline-first alternative), waits up to `batch_window_ms` for more
same-key requests to arrive if it isn't at `max_batch_size` yet, then
concatenates every queued request's inputs along axis 0 into one
`ModelVariant::run()` call and slices the batched output back apart for
each caller.

**The batching key includes shape, not just (node, precision).**
`NamedTensor::concatBatch` requires every non-batch dimension to match
across the parts it's concatenating; a node whose binder can legitimately
produce different non-batch shapes across jobs — variable sequence length
being the obvious case — would, if batching keyed on `(node, precision)`
alone, occasionally have two genuinely incompatible-shaped requests
offered to each other as batch-mates, and `concatBatch` throwing then
fails *every* request in that batch, not just the mismatched one. Folding
a canonical encoding of each input's non-batch dimensions
(`shapeSignature` in `scheduler.cpp`) into the key means such requests
simply land in separate pending queues instead, each flushed
independently; `tests/test_scheduler_deadlines.cpp` exercises this against
a real ONNX model whose non-batch dimension is symbolic
(`assets/test_variable_width_identity.onnx`) with concurrent width-4 and
width-8 jobs.

This is a deliberately simple design in a couple of specific places,
called out because a "no shortcuts, but no false sophistication either"
approach means being explicit about them:

- **Linear scan over ready keys**, not an indexed priority heap. At DAG
  sizes in the tens of distinct (node, precision, shape) triples this
  costs nothing measurable and is far easier to read/verify than a
  heap-with-live-comparator.
- **An exact `condition_variable::wait_until` for the batch window**, not
  a full timing wheel. A batch window that isn't flushable yet computes
  the earliest closing time among all such windows and waits precisely
  until then (or until `submit()`'s `notify_all` wakes it early because
  new work arrived); a worker in this state costs zero CPU. An earlier
  revision instead released the lock and slept a fixed 1-2ms before
  re-checking every ready key, which both burned CPU continuously while
  any window was open and quantized the window's real accuracy to that
  sleep granularity. This still isn't a full per-key timer-wheel data
  structure — at demo scale, waiting for the single nearest deadline
  among a handful of keys costs nothing measurable either, the same
  "deliberately simple, stated explicitly" tradeoff as the linear scan
  above; a production system managing hundreds of independently-timed
  keys would want one.
- **Batch latency is attributed to every item in the batch**, not divided
  by batch size — `MetricsRegistry` records the same wall-clock number for
  each of N batched callers. This is a conservative (slightly pessimistic
  for large batches) choice, stated explicitly here and in
  `docs/BENCHMARKS.md` rather than silently baked into the numbers.

**`shutdown()` actually stops new work.** Calling it sets an atomic flag;
`submitJob()` called afterwards returns an already-failed future
(`LoomcoreError`) instead of dispatching anything, while jobs already in
flight keep running to completion (each `BackendLane`'s worker loop
drains its queued work before noticing `stopping_` and exiting). An
earlier revision documented this contract but the method body was empty.

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

Seven built-in policies ship in `loomcore/router.h`:

| Policy | Reads | Decides |
|---|---|---|
| `LatencyBudgetPolicy(threshold_ms)` | job's remaining time budget, whether the node has an INT8 variant | fall back to INT8 once the deadline is close (wires the quantization milestone into live decision-making, not just a benchmark) |
| `LoadAwareBackendPolicy` | CPU vs GPU_SIM queue depth | route to whichever simulated lane is shallower |
| `ConfidenceGatePolicy(skip_above)` | a `ConfidenceExtractor` applied to a designated upstream node's output | skip a downstream node once upstream is already confident enough (used by the reference pipeline to skip `bert_tiny` once `mobilenet` doesn't need disambiguating) |
| `PlannedPrecisionPolicy` | `RoutingContext::precision_plan`, computed DAG-wide by `PrecisionPlanner` | downgrade exactly the nodes a knapsack over the critical path chose — see "Deadlines and resilience" below |
| `CircuitBreakerPolicy(error_rate, min_samples, cooldown_ms)` | `MetricsRegistry::outcomeStats` for the node | fail-fast (skip) a consistently-failing node, probing recovery after cooldown |
| `BulkheadPolicy(max_concurrent)` | `MetricsRegistry::inFlight` for the node | shed (skip) requests once a node's own concurrency cap is reached |

All seven, plus `CompositeRouter` itself, are unit-tested in isolation
(`tests/test_router.cpp`) and the first three are additionally exercised
end-to-end against a real scheduler + real ONNX sessions in
`tests/test_scheduler.cpp`; the deadline/planning policies are exercised
end-to-end in `tests/test_scheduler_deadlines.cpp`.

## Deadlines and resilience

`time_budget_ms` on `submit()`/`run()` long predates this section, but
until now it only ever fed `LatencyBudgetPolicy`'s per-node threshold
check — a budget that was missed simply resulted in a slow job. Four
features make a deadline something the scheduler actually enforces and
plans around, instead of merely a hint a policy might act on. All four
are **off by default** via `SchedulerConfig` — new, more invasive
machinery layered onto a scheduler that already worked, opted into
individually rather than silently changing behavior for every existing
caller:

**Admission control** (`SchedulerConfig::enable_admission_control`).
When a job is submitted with a time budget, `Scheduler::submitJob` calls
`PrecisionPlanner::planPrecisionForBudget` (see below) *before*
dispatching anything; if the plan judges the budget undeliverable even
after downgrading every eligible critical-path node, the call returns an
already-failed future (`JobRejectedError`) instead of a job that starts
and is doomed to miss its deadline. It never rejects on a cold
`MetricsRegistry` (see `PrecisionPlan::has_estimate`) — there is nothing
yet to judge feasibility from, so the honest behavior is to let the job
through and start learning.

**Deadline cancellation** (`SchedulerConfig::enable_deadline_cancellation`).
A per-job `CancellationToken` (`loomcore/model_node.h` — a thin,
header-clean wrapper around `Ort::RunOptions::SetTerminate`, keeping the
ONNX Runtime C++ API out of every public header the way
`loomcore::Environment` already does) is bound to whichever node happens
to be running when the budget expires. A dedicated "deadline reaper"
thread (started only when this flag is on) wakes every
`deadline_reaper_poll_ms`, and for each job whose `remainingBudgetMs()`
has gone non-positive, calls `requestCancel()` — aborting the in-flight
`ModelVariant::run()` at its next ONNX Runtime op boundary — and fails
the job with `DeadlineExceededError`. This is real cancellation, not a
routing hint: it's the difference between "route around a tight budget
ahead of time" (what `LatencyBudgetPolicy` already did) and "a budget
that's actually binding."

**Precision planning** (`SchedulerConfig::enable_precision_planning`,
implied by admission control). `include/loomcore/planner.h`'s
`planPrecisionForBudget` finds the DAG's critical path by measured FP32
p95 latency (`MetricsRegistry`, keyed per-precision via
`precisionMetricsKey("<node>", Precision::FP32/INT8)` — recorded
alongside the existing plain per-node key on every node completion), and
if that path's total cost exceeds the budget, solves a **0/1 knapsack**:
choose the subset of critical-path nodes to downgrade to INT8 that
minimizes total `NodeConfig::quality_weight` spent, subject to their
combined FP32-minus-INT8 savings covering the shortfall. This is a real
DP, not a dressed-up greedy: sorting candidates by savings and taking the
fewest that cover the gap is *provably optimal* when every node has the
same `quality_weight` (the default), but the moment weights differ — a
caller flagging one node as more precision-sensitive than another via a
higher `quality_weight` — greedy-by-savings can pick a strictly more
expensive set than the true minimum; `tests/test_planner.cpp` constructs
exactly such a case (a single node whose saving alone would satisfy a
naive greedy, at 2.5× the weight of two smaller nodes together) and
checks the DP finds the cheaper set. The resulting plan is computed once
per job, stored in `JobState`, and exposed read-only via
`RoutingContext::precision_plan` for `PlannedPrecisionPolicy` (or a
custom policy) to read — the DP itself never runs on a lane worker
thread or inside a routing decision.

**Earliest-deadline-first lane scoring** (`SchedulerConfig::use_edf_scoring`).
An alternative to `BackendLane`'s aged-priority scan: a
`(node, precision, shape)` key with at least one queued task carrying a
job deadline is scored by that deadline (soonest first) instead, and
always outranks a key with no deadline-bearing task. Selectable
independently of the aging formula, which keeps ordering non-deadline
keys among themselves either way.

Two more policies round out the resilience picture, both usable with no
scheduler changes at all — they only needed two small additions to
`MetricsRegistry` (`recordOutcome`/`outcomeStats` and
`incrementInFlight`/`decrementInFlight`/`inFlight`, called from the
scheduler's existing per-node completion/failure callbacks):

- **`CircuitBreakerPolicy`** is a proper three-state machine (Closed →
  Open → HalfOpen → Closed-or-Open), not a boolean "tripped" flag. A
  boolean is tempting and wrong: the request that flips it back to admit
  itself as the recovery probe makes every *other* concurrent request
  look healthy too, letting all of them through instead of just the one
  probe — `tests/test_router.cpp` fires 16 concurrent `decide()` calls the
  instant cooldown elapses and asserts exactly one is let through.
- **`BulkheadPolicy`** caps per-node in-flight concurrency, shedding
  (via `skip = true` — `RoutingDecision` has no "defer/retry" outcome, and
  shedding is preferable to an unbounded hidden queue anyway) once the cap
  is reached, so one overloaded node can't back up everything queued
  behind it on a shared lane.

## Hot-reloading a graph

`Runtime::reloadGraph` builds an entirely new `Graph` / loaded
`ModelNode` set / `Scheduler` and swaps it in atomically, with **zero
downtime**: every job already in flight keeps running against the exact
snapshot it started with, to completion, and a new job submitted after
the swap sees the new one — no job is ever dropped, delayed, or handed a
torn mix of old and new state.

This replaces what used to be a real bug: an earlier `Runtime::loadGraph`
mutated its `impl_->graph`/`impl_->nodes` members *in place*, but the
existing `Scheduler` — whose lane worker threads hold plain `const&`
references into exactly those members, established at construction and
relied on for the Scheduler's entire lifetime — kept running against
them throughout. Calling `loadGraph` a second time on a `Runtime` already
handling traffic reassigned those members while lane threads could be
concurrently reading through the old references: a use-after-free/data
race reachable simply by using the documented reload path, not an edge
case.

**The fix is an RCU-style (read-copy-update) snapshot**, not a mutex
around the mutation. `GraphSnapshot` (`runtime.cpp`) bundles a `Graph`,
its loaded `ModelNode`s, the router, and the `Scheduler` built to
reference *that specific* `Graph`/`ModelNode` pair — constructed once,
never mutated again. `Runtime::Impl` holds `shared_ptr<GraphSnapshot> current`,
read via `std::atomic_load` (lock-free) by every `submit()`/`run()`/`graph()`
call and replaced via `std::atomic_store` by `loadGraph`/`reloadGraph`
(serialized against each other by a mutex, so two concurrent reloads
can't interleave their JSON-parsing/session-loading side effects — the
atomic pointer swap itself doesn't need that). Building the new snapshot
— parsing JSON, loading every ONNX session — happens *before* the swap,
so the old graph keeps serving every job for the whole duration; there is
no window where traffic pauses.

A submitted job pins its snapshot alive for exactly as long as it runs:
`Scheduler::submitJob` takes an opaque `shared_ptr<void> keep_alive`,
stored in `JobState`, which `Runtime::submit` sets to (a proxy around)
the snapshot the job was submitted against. Once every in-flight job
against an old snapshot has settled and every reference to it — the
retired `current` slot included — has dropped, the snapshot, and with it
its `Scheduler`, is destroyed.

**That destruction needed one more piece to be safe.** A job's last
completion callback runs on one of its own `Scheduler`'s lane worker
threads — necessarily, since that's where node execution happens. If
that callback also happens to drop a `GraphSnapshot`'s last reference
(because a reload already replaced `current` and this was the last job
still pinning the old one), destroying the snapshot destroys its
`Scheduler`, whose destructor joins its own `BackendLane` worker threads
— including the very thread the callback is running on. Joining a
`std::thread` from itself is undefined behavior; in practice, an uncaught
`std::system_error` on a thread with no caller frame to catch it, i.e.
`std::terminate`. (This is exactly the failure mode
`tests/test_runtime_reload.cpp`'s continuous-load stress test caught
during development — real concurrent load finding a real bug in the
first draft of this feature, which is the whole point of a test shaped
that way rather than a single reload-then-assert.)

The fix: `Runtime::Impl::deferredKeepAlive` never hands a job the real
`shared_ptr<GraphSnapshot>`. It hands out a proxy `shared_ptr<void>`
whose custom deleter — which *does* run on whatever thread drops the
last reference, lane thread included — does nothing more than push the
real `shared_ptr<GraphSnapshot>` onto a small queue (the "graveyard").
A single dedicated graveyard thread, owned by `Runtime::Impl` and
entirely separate from any `Scheduler`'s lane pool, drains that queue and
is where the actual `~GraphSnapshot()`/`~Scheduler()` calls happen — on
its own stack, nowhere near a lane thread. `examples/reload_demo.cpp`
demonstrates the whole feature under continuous concurrent load: several
producer threads submit jobs back-to-back while the main thread reloads
the graph six times in a row, and every single job completes
successfully regardless of which snapshot it happened to run against.

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
`job_rejected` (admission control — see "Deadlines and resilience"),
`error`. `MetricsRegistry` is the separate, complementary piece: a
rolling per-node latency window (p50/p95/mean), per-lane queue depth,
per-node in-flight count, and per-node outcome (success/failure) history
— all of what `RoutingContext` and the router policies actually read.
Logger is an append-only record of *what happened*; MetricsRegistry is
the *current state* the router reasons about.

**`log()` never performs I/O on the calling thread.** It serializes the
event to a JSON string (pure CPU work) and pushes the line onto a queue;
a single dedicated writer thread drains that queue and performs the
actual (batch-flushed) write to stdout/file. `log()` is called from every
scheduler lane worker thread on every scheduling event, and an earlier
revision wrote directly with `std::endl` (an implicit flush) under one
shared mutex on whichever lane thread produced the event — a burst of
concurrent events from both simulated backends could stall the very lane
workers the scheduler depends on for throughput. This wasn't visible in
any individual recorded node latency (those are stamped before the log
call runs), only in end-to-end job throughput under load — the kind of
cost that's easy to ship because nothing in the numbers you'd normally
look at points at it. `Logger::flush()` blocks until every line enqueued
before the call has been written (or dropped — `droppedCount()`), for
tests or tools that need the log *file's* contents rather than
`recentLines()` (which reflects every event immediately regardless of
writer backlog).

**`MetricsRegistry` is sharded per key, not one global mutex.** Each
node/lane key owns its own entry (its own `std::mutex` for the latency/
outcome deques, its own `std::atomic<size_t>` for queue depth and
in-flight count); a `shared_mutex` guards only the *map structure* itself
(inserting a brand-new key, which happens at most once per key over the
registry's lifetime). An earlier revision used one mutex for the whole
registry, meaning the CPU and GPU_SIM lanes — nominally independent, with
deliberately different cost profiles (see "Simulated backends" above) —
serialized on every queue-depth update and every `stats()` call, which
sorts the whole latency window while holding the lock: two backends with
genuinely different performance characteristics sharing a lock on their
single hottest operation.

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

## C API

`include/loomcore/c_api.h` / `src/c_api.cpp` is a second, independent
`extern "C"` surface over the same `loomcore_core` — opaque handles, POD
structs (`LoomcoreCTensor`, `LoomcoreNamedCTensor`), integer status codes
instead of C++ exceptions — covering `Runtime`'s lifecycle, loading a
graph with C-callable binders, running one job synchronously, and reading
node latency stats. It exists because `Runtime`'s C++ API (`std::string`,
`std::function`, `std::shared_ptr`) requires a compatible compiler *and*
C++ ABI to consume — the `/wd4251` suppression on `loomcore_core` (see
CMakeLists.txt) is explicit that this only holds because every in-repo
consumer links the exact `.dll` built alongside it — which makes a
"shared library" whose only consumable surface requires that a vendored
header, not a library other tooling/languages can actually link against.

The one real design problem this surface has to solve: a C function
pointer can't return `std::vector<NamedTensor>`, and `NodeExecutionContext`'s
`map<node_id, vector<NamedTensor>>` has no natural C representation
either. `LoomcoreBinderFn` flattens both the job's external inputs and
every upstream node's outputs into one array each of
`(owner_node_id, tensor)` pairs (`owner_node_id == NULL` marks a graph
input) and writes its outputs into a caller-provided buffer — a
synchronous handoff (the buffer only needs to stay valid until the call
returns; Loomcore copies out of it immediately, before the callback
returns) that keeps ownership rules simple without needing a C-side
allocator callback. `bindings/c/smoke_test.c` is a real, plain-C program
(compiled with a C compiler, not C++ — see `bindings/c/CMakeLists.txt`)
built and run against nothing but this header and the compiled library;
see `docs/CLAIMS.md` #13 to run it yourself.

This is intentionally a *subset* of the C++ API — no async
`submit()`/`future`, no custom `IRoutingPolicy`, no confidence-gated
nodes yet — covering what a first cross-language consumer actually needs,
not a mechanical translation of every C++ method.

## Perfetto trace export

`loomcore/perfetto_export.h`'s `exportPerfettoTrace` is a pure
post-processing step — it reads a JSON-lines log `Logger` already wrote
and converts it to Chrome's Trace Event Format, the format both Perfetto
(`https://ui.perfetto.dev`) and `chrome://tracing` read natively with no
library needed on either end. It never runs inline with a job (exporting
a trace costs nothing during the run being traced): each simulated
backend lane becomes its own track with a duration bar per real
`ModelVariant::run()` call (from `batch_flushed` events, sized to their
measured `latency_ms`), routing decisions become instant markers on a
separate "Router" track, each job's life (submitted to completed, or to
the error that failed it) becomes a span on a "Jobs" track, so the gaps
between its bars read as batch-window waits and queueing, and each job's
*observed* path through the DAG
becomes connecting flow arrows, from the bar that ran a node for that job
to the bar that ran its next node — read from the log's actual
timestamps, not reconstructed from the `Graph`'s static topology, so what
you see is what that specific run actually did.

Two details of the log shape the exporter has to account for, both
covered by `tests/test_perfetto_export.cpp`. `batch_flushed` is logged
when a batch *finishes* (its `LogEvent` is built after the run is timed),
so a bar starts `latency_ms` before that event's timestamp; an earlier
revision used the timestamp as the start and drew every bar one full
duration late. And a batch serving several jobs logs `job_id` as
`"(N jobs)"`; the exporter recovers which jobs it served from the
`node_completed` events the scheduler logs for each of them immediately
afterwards (same node, lane, precision and `latency_ms`), lists them in
the bar's `args.jobs`, and anchors each flow arrow inside the two bars it
connects, on their own lane tracks, which is what Perfetto needs to bind
a flow to its slices (the earlier revision put them on a track with no
slices at all, so they never rendered).
`tools/trace_export.cpp` is the CLI wrapper (`loomcore_trace_export
logs/loomcore.jsonl trace.json`); `tests/test_perfetto_export.cpp` checks
the output is structurally valid Trace Event Format against a small
synthetic log.
