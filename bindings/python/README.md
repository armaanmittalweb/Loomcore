# Loomcore Python bindings

A thin pybind11 wrapper around the C++ runtime (`loomcore_core`) — the
orchestrator itself is C++; this package exists for tooling, scripting,
and testing against it. See `docs/ARCHITECTURE.md` "Python bindings" for
how tensors, node binders, and routing policies cross the language
boundary, and for a real pybind11 lifetime bug found and fixed while
building this.

## Build

Built by the main CMake project (`LOOMCORE_BUILD_PYTHON_BINDINGS=ON`, the
default) as part of `cmake --build build`. It is **not** a separate pip
package yet — `loomcore/__init__.py` locates the compiled extension next
to whichever CMake build produced it (via `LOOMCORE_BUILD_DIR` or the
default `build/bin/` layout). See `../../docs/BUILD.md`.

## Use

```python
import sys
sys.path.insert(0, "bindings/python")   # or set PYTHONPATH
import numpy as np
import loomcore

def my_binder(graph_inputs, upstream_outputs):
    return [("data", graph_inputs["data"])]

runtime = loomcore.Runtime()
runtime.load_graph(
    "examples/graph_config.json",
    binders={"mobilenet": my_binder, "bert_tiny": ...},
    router=loomcore.LatencyBudgetPolicy(30.0),   # or a CompositeRouter, or your own RoutingPolicy subclass
)
result = runtime.run({"data": np.zeros((1, 3, 224, 224), dtype=np.float32)})
```

See `examples/run_example.py` for a complete, runnable version — MobileNetV2
+ bert_tiny with a **Python-implemented** confidence-gated skip policy
(`SkipIfConfident`, a `loomcore.RoutingPolicy` subclass) alongside the
built-in C++ `LatencyBudgetPolicy` and `LoadAwareBackendPolicy`.

## API surface

- `loomcore.Runtime()` — `.load_graph(config_path, binders, confidence_extractors=None, router=None, options=None)`,
  `.reload_graph(...)` (same arguments: hot-swaps the whole graph under live traffic, see
  `docs/ARCHITECTURE.md` "Hot-reloading a graph"),
  `.run(inputs: dict[str, np.ndarray], priority=0, time_budget_ms=-1.0) -> dict[str, list[np.ndarray]]`,
  `.graph() -> {nodes, topo_order, sinks}` (plain data, copied out of the current snapshot),
  `.node_stats(node_id) -> dict`, `.recent_logs(n=100) -> list[str]` (JSON lines).
- `loomcore.RuntimeOptions()` — `log_file`, `log_to_stdout`, `intra_op_threads_per_variant`, and
  `scheduler`, a `loomcore.SchedulerConfig` carrying the opt-in deadline flags
  (`enable_admission_control`, `enable_precision_planning`, `enable_deadline_cancellation`,
  `deadline_reaper_poll_ms`, `use_edf_scoring`) plus the lane settings. A graph config's own
  `"scheduler"` block still overrides the lane settings, as in C++.
- `loomcore.Backend`, `loomcore.Precision` — enums matching the C++ ones.
- `loomcore.RoutingContext` — read-only: `job_id`, `node_id`,
  `cpu_queue_depth`, `gpu_queue_depth`, `time_budget_remaining_ms`,
  `upstream_confidence_source` (`list[np.ndarray] | None`).
- `loomcore.RoutingDecision(precision=None, backend=None, skip=False, reason="")`.
- `loomcore.RoutingPolicy` — subclass and implement `decide(self, ctx) -> RoutingDecision | None`.
- All seven built-in C++ policies: `LatencyBudgetPolicy(threshold_ms)`, `LoadAwareBackendPolicy()`,
  `ConfidenceGatePolicy(skip_above)`, `PlannedPrecisionPolicy()`,
  `CircuitBreakerPolicy(error_rate_threshold, min_samples, cooldown_ms)`, `BulkheadPolicy(max_concurrent)`,
  and `CompositeRouter()` (`.add(policy)`), usable standalone or mixed with Python ones.
- `loomcore.WordPieceTokenizer(vocab_path, max_seq_len=32)` — the C++ tokenizer the reference
  pipeline uses; `.encode(text)` returns `{input_ids, attention_mask, token_type_ids}` as int64
  arrays of shape `[1, max_seq_len]`.
- `loomcore.export_perfetto_trace(jsonl_path, output_json_path) -> int` — the Perfetto/Chrome Trace
  Event exporter (`loomcore/perfetto_export.h`).
- Errors: `loomcore.LoomcoreError` (a `RuntimeError`), and its subclasses `JobRejectedError`
  (admission control refused the job before dispatching anything) and `DeadlineExceededError`
  (the deadline reaper cancelled it).

Python callables handed to the runtime (binders, confidence extractors) are held so that their
reference counts are only ever touched with the GIL held, including when a graph retired by
`reload_graph` is torn down on the runtime's own teardown thread.

## Tests

```
python -m pytest bindings/python/tests
```

Hermetic (the committed fixture graphs in `assets/` only); skipped with the import error as the
reason when the extension isn't built.

`ConfidenceGatePolicy` was originally left out in favour of showing the same rule written in
Python (`examples/run_example.py`'s `SkipIfConfident`, still the better illustration of a
Python policy); it is bound now because the live Space (`space/server.py`) lets visitors toggle
every built-in policy. The tokenizer is bound for the same reason: the Space feeds `bert_tiny`
through the real C++ tokenizer rather than a Python copy of it.
