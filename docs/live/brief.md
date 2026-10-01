# Loomcore live: brief

Loomcore replaces LatentBook in the Lab at amittal.dev (the owner's decision, 1 Oct 2026). This brief puts the real C++ runtime online for free, with a console that lets a visitor watch it schedule, route, batch and hot-swap.

## Shape

| Piece | Where | What |
|---|---|---|
| Runtime server | Hugging Face Docker Space `armaanmittalweb/loomcore` (free CPU Basic: 2 vCPU, 16 GB RAM, sleeps after 48 h idle) → `https://armaanmittalweb-loomcore.hf.space` | The real Loomcore build (Linux, CMake + Ninja, ONNX Runtime fetched by `cmake/FetchOnnxRuntime.cmake`), the reference models prepared at image build time, and a small FastAPI server over the Python bindings. New directory `space/` |
| Console | Vercel, root `web/` → `https://loomcore.amittal.dev` | A static app that drives the Space and draws what the runtime did. New directory `web/` |

No Worker, no database: the Space is stateless.

## Space API (JSON, CORS for `https://loomcore.amittal.dev` and `http://localhost:5177`)

| Method, path | Body → response |
|---|---|
| GET /health | `{ ok, version, models: { mobilenet: ['FP32','INT8'], bert_tiny: [...] }, cpu: string }` |
| GET /graph | the loaded graph: nodes, variants, dependencies, router policy chain |
| POST /run | multipart `image` (≤ 2 MB JPEG/PNG) or `{ sample: string }`, plus `{ timeBudgetMs?, policies?: PolicyName[] }` → `RunResult` |
| POST /load | `{ jobs: 1..64, concurrency: 1..16, timeBudgetMs?, policies?, sample? }` → `LoadResult` (per-node p50/p95, batches formed, router decisions counted, jobs rejected by admission control / cancelled by deadline, and the trace) |
| POST /reload | `{ jobs: 8..64, swaps: 1..6 }` → `ReloadResult`: hot-swaps the graph under continuous load (as `examples/reload_demo.cpp` does) and reports jobs submitted / completed / lost (must be 0) and per-swap timings, plus the trace |
| GET /bench | the latest `latency_bench` run on this machine (run once at container start, cached): FP32 vs INT8 mean/p50/p95 per model, plus the CPU flags (AVX2, AVX-512, VNNI) so the console can explain the result |

`RunResult = { label, confidence, top5, embedding: number[16] (first 16 of 128), skipped: string[], decisions: Decision[], nodes: { id, precision, backend, ms }[], totalMs, trace: TraceEvent[] }`. `trace` is the Perfetto/Chrome Trace Event JSON that `perfetto_export` produces for the run (expose that function, and `Runtime.reload_graph`, to Python in `bindings/python/loomcore_py.cpp` if they aren't yet: small additive bindings with tests). `PolicyName = 'latency-budget' | 'load-aware' | 'confidence-gate' | 'precision-planner' | 'circuit-breaker' | 'bulkhead'`.

Guards: one `/load` or `/reload` at a time (others get 429 with `retryAfter`), 20 requests a minute per IP, request timeouts, no file is written outside a temp dir. Sample images (6, from Wikimedia Commons or similar with licences recorded in `space/samples/LICENSES.md`) baked into the image.

The Docker image: multi-stage. Stage 1 prepares models with `scripts/prepare_all_models.py` (CPU-only torch). Stage 2 builds Loomcore and the bindings (`RelWithDebInfo`, tests run with `ctest` and must pass, or the build fails). The final stage is `python:3.11-slim` with the built library, the bindings, the models and the server; keep it small (no torch in the final image). `space/README.md` carries the Space YAML header (`sdk: docker`, `app_port: 7860`).

## Console (`web/`)

Audience: an engineer or recruiter who should understand in 30 seconds that this is a real scheduler running real models, then be able to poke it.

- **Opens on a real run**, not a hero: a recorded run captured from the Space (committed as `web/src/recorded/*.json`, labelled "Recorded on <date> on the live runtime") renders instantly, while the app wakes the Space in the background ("Live runtime: waking up, about a minute" → "Live"). Every panel says whether it shows recorded or live data.
- **The timeline is the centrepiece**: the run's trace drawn in-page (own canvas/SVG renderer, no Perfetto embed): one track per backend lane (CPU, GPU_SIM), duration bars per node execution coloured by precision, batch groupings, router decision markers you can hover for the policy's reason string, and DAG flow arrows. Zoom and pan. "Open in Perfetto" downloads nothing (just explains how; the trace can be copied).
- **The graph** beside it: MobileNetV2 → confidence gate → bert_tiny, with the router policy chain listed in order, each policy toggleable for the next run.
- **Controls**: pick a sample or upload an image; time budget slider; Run. Load test (jobs, concurrency) → per-node p50/p95, batch sizes, admission rejects, deadline cancels. Hot-swap under load → a counter of jobs submitted/completed/lost (0) and swap timings drawn on the timeline.
- **Benchmark** panel: the Space's own FP32 vs INT8 numbers next to the committed Ryzen 9 6900HX numbers from `docs/BENCHMARKS.md`, with the CPU-flag explanation (INT8 MobileNetV2 is slower without VNNI; that's the honest result).
- **Claims**: the table from `docs/CLAIMS.md`, each linking to the test on GitHub.
- Footer: repo link, "Part of the Lab at amittal.dev", page-view beacon `countViews('loomcore')` (copy `C:\Users\Armaan Mittal\Desktop\pp\SafeSpace\web\src\beacon.ts`).

### Look

An instrument for systems engineers: dense, precise, quiet, like a well-made tracing tool or a hardware datasheet, not a SaaS landing page. Light (paper white, graphite ink) and dark (near-black, phosphor-free; no neon) themes, both designed. One accent per meaning: FP32 and INT8 get two distinct hues used everywhere; red only for cancels, rejects and lost jobs. Type: a technical sans for UI and a monospace for numbers, ids and reason strings (self-hosted, OFL). No hero section, no feature cards, no gradients, no glows, no emoji, no stock imagery, no marketing copy. Motion only on the timeline (bars appearing in time order on a live run, ≤ 600 ms total; none with reduced motion). Works at 390 px (the timeline scrolls horizontally inside its panel; controls stack), 0 axe violations, keyboard operable.

Stack: Vite + TypeScript + Preact (or React), plain CSS with tokens. `vercel.json` with a strict CSP (connect-src self, the Space URL, `https://api.amittal.dev`), real `robots.txt`/`sitemap.xml`, title "Loomcore: a C++ runtime that schedules models as a graph", description, OG tags, prerendered text for crawlers.
