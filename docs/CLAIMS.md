# Claims

Every factual assertion this repository makes about its own behavior,
next to the exact command that checks it. The rule this document exists
to enforce: **a claim without a command next to it is a claim nobody has
actually checked.**

Run everything below from the repo root after building
(`docs/BUILD.md`). `loomcore_tests.exe` on Windows /
`loomcore_tests` on Linux — the examples use the Windows name to match
`docs/BUILD.md`'s own quick-start; drop the `.exe` on Linux/macOS.

| # | Claim | Command | Where it's asserted |
|---|---|---|---|
| 1 | The full test suite passes | `ctest --test-dir build -C RelWithDebInfo --output-on-failure` | `tests/` (27+ test cases, see `tests/CMakeLists.txt`) |
| 2 | A job's promise is settled exactly once, even under a deadline reaper racing a node's natural completion (no `std::terminate`) | `build\bin\RelWithDebInfo\loomcore_tests.exe --test-case="Deadline cancellation fails the job and never double-settles the promise under repeated stress"` | `tests/test_scheduler_deadlines.cpp` — 200 trials per run against real ORT sessions |
| 3 | Batching never mixes incompatible non-batch shapes for the same node | `build\bin\RelWithDebInfo\loomcore_tests.exe --test-case="Batching keys on shape, not just (node, precision): mixed widths never cross-contaminate"` | `tests/test_scheduler_deadlines.cpp`, real ORT session over `assets/test_variable_width_identity.onnx` |
| 4 | `Scheduler::shutdown()` actually stops accepting new jobs (not a no-op) | `build\bin\RelWithDebInfo\loomcore_tests.exe --test-case="Scheduler::shutdown() actually stops accepting new jobs"` | `tests/test_scheduler_deadlines.cpp` |
| 5 | Admission control rejects a job the precision planner judges undeliverable, and dispatches nothing first | `build\bin\RelWithDebInfo\loomcore_tests.exe --test-case="Admission control rejects a job the precision planner judges undeliverable"` | `tests/test_scheduler_deadlines.cpp` |
| 6 | The precision-downgrade knapsack finds the true minimum-quality-loss set, which a savings-only greedy would not | `build\bin\RelWithDebInfo\loomcore_tests.exe --test-case="PrecisionPlanner: minimal-weight knapsack beats greedy-by-largest-saving"` | `tests/test_planner.cpp` |
| 7 | The circuit breaker lets through exactly one probe after cooldown, never more, under concurrent load | `build\bin\RelWithDebInfo\loomcore_tests.exe --test-case="CircuitBreakerPolicy: exactly one caller is let through as the probe after cooldown"` | `tests/test_router.cpp` — 16 concurrent callers per run |
| 8 | `Runtime::reloadGraph` hot-swaps the graph under continuous concurrent load with zero jobs lost | `build\bin\RelWithDebInfo\loomcore_tests.exe --test-case="Runtime::reloadGraph swaps graphs under continuous concurrent load with zero jobs lost"` | `tests/test_runtime_reload.cpp`, and see #9 below for the real-model version |
| 9 | The same hot-swap holds against the real mobilenet+bert_tiny pipeline, with per-swap timing reported | `build\bin\RelWithDebInfo\loomcore_reload_demo.exe` | `examples/reload_demo.cpp` — prints `PASS`/`FAIL` and an exit code; needs `python scripts/prepare_all_models.py` run first |
| 10 | The reference pipeline runs end to end (real image → classification → confidence-gated embedding) and reports real per-node p50/p95 | `build\bin\RelWithDebInfo\loomcore_example.exe examples/sample.jpg` | `examples/run_example.cpp` |
| 11 | INT8 MobileNetV2 is measurably *slower* than FP32 on the reference (non-VNNI) machine — not fabricated, not silently skipped | `build\bin\RelWithDebInfo\loomcore_bench.exe --warmup 20 --iters 100` | `benchmarks/latency_bench.cpp`, numbers and explanation in `docs/BENCHMARKS.md` |
| 12 | The ORT dependency is fully encapsulated: the benchmark links and runs with no ONNX Runtime include/link dependency of its own | `grep -rn "onnxruntime" benchmarks/latency_bench.cpp` (expect: no matches) then build+run #11 | `benchmarks/latency_bench.cpp`'s header comment; enforced structurally, not just documented — only `src/model_node.cpp` and `src/environment.cpp` `#include <onnxruntime_cxx_api.h>` (`grep -rl onnxruntime_cxx_api.h src/` confirms exactly these two) |
| 13 | The C API is a real, separate consumable surface: a pure-C translation unit links and runs against it with no C++ Loomcore headers | `cmake --build build --config RelWithDebInfo --target loomcore_c_smoke_test && build\bin\RelWithDebInfo\loomcore_c_smoke_test.exe` | `bindings/c/smoke_test.c`, `include/loomcore/c_api.h` |
| 14 | The installed package is consumable via plain `find_package(Loomcore)` — not just "it builds in-tree" | see `examples/consumer/README.md` for the two-command repro (`cmake --install` then a from-scratch configure against the install tree) | `examples/consumer/`, `cmake/LoomcoreConfig.cmake.in` |
| 15 | CI builds and tests both platforms (Windows + Linux) on every push/PR | badge/history at the repo's Actions tab | `.github/workflows/build.yml` |
| 16 | A run's JSON-lines log converts to a real, structurally valid Perfetto/Chrome Trace Event Format timeline — lane tracks, per-execution duration bars, routing markers, DAG flow arrows | `build\bin\RelWithDebInfo\loomcore_example.exe examples/sample.jpg && build\bin\RelWithDebInfo\loomcore_trace_export.exe logs\loomcore.jsonl trace.json` then drop `trace.json` onto `https://ui.perfetto.dev` | `tools/trace_export.cpp`, `src/perfetto_export.cpp`; structural correctness unit-tested in `tests/test_perfetto_export.cpp` |

## What's deliberately not on this list

Being honest about scope is itself part of the claims contract — a list
that only ever grows and never says what it excludes reads as more
complete than it is:

- **No sanitizer (TSan/ASan/UBSan) CI job yet.** `.github/workflows/build.yml`
  builds and tests on MSVC (Windows) and GCC/Ninja (Linux); a
  Clang+TSan job needs a third toolchain in the matrix and hasn't been
  added. The concurrency-sensitive code (`Scheduler`, `BackendLane`,
  `Runtime`'s snapshot swap) is instead covered by the stress-style tests
  in claims #2, #3, #7, #8/#9 — repeated-trial and concurrent-caller
  tests that would very likely surface a data race, but that is evidence,
  not a guarantee a sanitizer run provides.
- **No fuzz targets yet.** `Graph::validate`'s cycle/reference checks and
  `NamedTensor::concatBatch`/`sliceBatch`'s shape algebra are good,
  contained libFuzzer targets (pure functions, no I/O) that simply
  haven't been written.
- **No published package.** `examples/consumer/` proves `find_package`
  works against a local `cmake --install` tree (claim #14); there is no
  vcpkg port, Conan recipe, or PyPI wheel published from that scaffolding
  yet.
- **No deterministic/seeded scheduler simulator.** The stress tests above
  vary real OS thread scheduling, not a seeded, bit-exactly-replayable
  interleaving — a full deterministic-simulation harness (FoundationDB-
  style) would require replacing `BackendLane`'s real thread pool with a
  cooperative single-threaded executor behind a `Clock`/`Executor`
  abstraction, which is a materially larger, riskier change than
  anything else in this document and was deliberately not attempted
  alongside it.
