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

- `loomcore.Runtime()` — `.load_graph(config_path, binders, confidence_extractors=None, router=None)`,
  `.run(inputs: dict[str, np.ndarray], priority=0, time_budget_ms=-1.0) -> dict[str, list[np.ndarray]]`,
  `.node_stats(node_id) -> dict`, `.recent_logs(n=100) -> list[str]` (JSON lines).
- `loomcore.Backend`, `loomcore.Precision` — enums matching the C++ ones.
- `loomcore.RoutingContext` — read-only: `job_id`, `node_id`,
  `cpu_queue_depth`, `gpu_queue_depth`, `time_budget_remaining_ms`,
  `upstream_confidence_source` (`list[np.ndarray] | None`).
- `loomcore.RoutingDecision(precision=None, backend=None, skip=False, reason="")`.
- `loomcore.RoutingPolicy` — subclass and implement `decide(self, ctx) -> RoutingDecision | None`.
- `loomcore.LatencyBudgetPolicy(threshold_ms)`, `loomcore.LoadAwareBackendPolicy()`,
  `loomcore.CompositeRouter()` (`.add(policy)`) — the built-in C++ policies, usable
  standalone or mixed with Python ones in a `CompositeRouter`.

Not exposed to Python (deliberately — see `docs/ARCHITECTURE.md` "Python
bindings" for why each is out of scope): `loomcore::WordPieceTokenizer`
(re-implemented compactly in pure Python inside `run_example.py` instead,
since re-exporting it would need its own binding surface for no benefit
here), and `ConfidenceGatePolicy` specifically (its C++ demo lives in
`run_example.cpp`; the Python demo shows the *same rule* implemented
natively in Python instead, which is the more interesting thing to show
from this side of the binding).
