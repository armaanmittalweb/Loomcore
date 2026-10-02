# Loomcore runtime server

This image builds and runs [Loomcore](https://github.com/armaanmittalweb/loomcore), a C++ runtime that
loads several ONNX models as a dependency graph (MobileNetV2, then a confidence gate, then bert_tiny),
schedules them with dynamic batching across two backend lanes, routes each job through a chain of
policies, enforces deadlines, and hot-swaps the whole graph under load. The console that drives it and
draws what it did is at **[loomcore.amittal.dev](https://loomcore.amittal.dev)**.

Nothing here is a mock: `/run` executes the real `loomcore::Runtime` through its Python bindings, and
every response carries the Chrome Trace Event JSON the runtime's own exporter produced from that
request's log.

It runs on an Oracle Cloud Always Free Arm VM (Ampere A1, Neoverse N1), always on, behind a
Cloudflare Tunnel at `https://loomcore-api.amittal.dev`; see [deploy/oracle](../deploy/oracle/README.md).

## API

JSON in and out. CORS allows `https://loomcore.amittal.dev` and `http://localhost:5177`.

| Method, path | Body | Returns |
|---|---|---|
| `GET /health` | | `{ ok, ready, version, models, cpu, uptimeS, busy, bench }` |
| `GET /graph` | | nodes, edges, the router policy chain (order, what each reads and decides), scheduler flags, samples |
| `POST /run` | `{ sample?, timeBudgetMs?, policies? }`, or multipart with `image` (JPEG/PNG, at most 2 MB) | label, confidence, top 5, the first 16 of bert_tiny's 128 embedding values, skipped nodes, router decisions with reasons, per-node precision/lane/ms, `trace` |
| `POST /load` | `{ jobs: 1..64, concurrency: 1..16, timeBudgetMs?, policies?, sample? }` | per-node p50/p95, batch sizes, router decisions by policy, completed / rejected / cancelled / shed, `trace` |
| `POST /reload` | `{ jobs: 8..64, swaps: 1..6 }` | jobs submitted / completed / lost (must be 0), per-swap build time and position, `trace` |
| `GET /bench` | | `loomcore_bench` (FP32 vs INT8, 20 warmup + 100 iterations) as run once at start-up, and the CPU's AVX2 / AVX-512 / VNNI flags |

`policies` is any subset of `circuit-breaker`, `bulkhead`, `confidence-gate`, `precision-planner`,
`latency-budget`, `load-aware`, applied in that order. A different set from the loaded one hot-swaps the
graph with the new router first (`graphSwapMs` says how long that took). Admission control, deadline
cancellation, precision planning and EDF lane scoring are on, so a `timeBudgetMs` is binding: a job the
planner judges undeliverable is rejected before anything runs, and one that runs out of time is
cancelled mid-inference.

Guards (the rate limit keys on Cloudflare's `CF-Connecting-IP`): one `/load` or `/reload` at a time (others get `429` with `retryAfter`), one job-running request
in the runtime at a time, 20 POSTs a minute per IP, request timeouts (25 s for `/run`, 90 s for the
others), and nothing written outside `/tmp`.

## Image

`Dockerfile` clones the source at `LOOMCORE_REF` (default `live`), prepares the models with
`scripts/prepare_all_models.py`, builds and tests exactly as the repository's CI Linux job does (the
build fails unless `ctest` passes), runs the binding and server tests against the real models, and
assembles a `python:3.11-slim` image without the compiler or torch.

Sample photos are from Wikimedia Commons; see `samples/LICENSES.md`.
