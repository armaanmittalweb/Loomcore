# Building Loomcore

One CMake configuration builds `loomcore_core` as a `.dll` (MSVC) or `.so`
(GCC/Clang) — the "shared library, cross-platform" qualification is a
build-system property, not a per-platform fork. This has been built and
fully exercised (tests, example, benchmark, Python bindings) with MSVC
2022 on Windows; the Linux/GCC-Clang path is written to the same portable
C++17 (platform-specific bits are isolated behind `#ifdef _WIN32` in
`src/model_node.cpp` and behind `WIN32`/`UNIX` branches in
`CMakeLists.txt`) but has not been independently compiled in this
environment — see `.github/workflows/build.yml`, which builds both
platforms in CI.

## Prerequisites

- CMake ≥ 3.20
- A C++17 compiler: MSVC 2019+, GCC 9+, or Clang 10+
- Python ≥ 3.9 (for the bindings, and for `scripts/*.py` if you're
  regenerating the models) with `pip install -r scripts/requirements.txt`
- Network access on first configure: CMake downloads a prebuilt ONNX
  Runtime release (~150-200MB, cached under `build/_deps/` after the
  first run) plus a handful of single-header dependencies (nlohmann/json,
  doctest, stb_image). Point `-DONNXRUNTIME_ROOT_DIR=<path>` at an
  existing install to skip that download entirely.

## 1. Prepare the models (once)

```
pip install -r scripts/requirements.txt
python scripts/prepare_all_models.py
```

This downloads MobileNetV2 + ImageNet labels, relaxes MobileNetV2 to a
dynamic batch axis, exports bert_tiny (`prajjwal1/bert-tiny`) to ONNX,
quantizes both to INT8, and regenerates the tiny fixture models the unit
tests use. See `docs/ARCHITECTURE.md` "Quantization" for why each model
is quantized differently. Everything lands under `models/` and `assets/`
(gitignored except the small fixtures/labels — see `.gitignore`).

Run the individual `scripts/*.py` files instead if you only need one
step (e.g. you already have the models and just want to re-quantize).

## 2. Configure and build

**Windows (MSVC, Visual Studio generator — multi-config):**

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config RelWithDebInfo --parallel
```

**Linux / macOS (GCC or Clang, Ninja or Makefiles — single-config):**

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel
```

Either way, every target (the `loomcore_core` shared library, the
examples, the benchmark, the tests, the C API smoke test, the trace
exporter, and the `_loomcore` Python extension) lands in one shared
output directory (`build/bin/` — or
`build/bin/<Config>/` for the multi-config Visual Studio generator) so
the ONNX Runtime and `loomcore_core` shared libraries are always
discoverable next to whatever links them, without extra `PATH`/`LD_LIBRARY_PATH`
setup. `CMakeLists.txt`'s "Loomcore configuration summary" at the end of
configure shows what got enabled.

Useful `-D` options (all documented at the top of `CMakeLists.txt`):

| Option | Default | Effect |
|---|---|---|
| `LOOMCORE_BUILD_TESTS` | `ON` | doctest unit + scheduler-integration tests |
| `LOOMCORE_BUILD_EXAMPLES` | `ON` | `loomcore_example` (needs the prepared models) |
| `LOOMCORE_BUILD_BENCHMARKS` | `ON` | `loomcore_bench` (needs the prepared models) |
| `LOOMCORE_BUILD_PYTHON_BINDINGS` | `ON` | `_loomcore` pybind11 extension (auto-disables if no Python dev headers) |
| `LOOMCORE_DOWNLOAD_ONNXRUNTIME` | `ON` | auto-fetch ORT; set `OFF` + `ONNXRUNTIME_ROOT_DIR` to use an existing install |

## 3. Run everything

```
ctest --test-dir build -C RelWithDebInfo --output-on-failure   # tests only need assets/, not models/

./build/bin/RelWithDebInfo/loomcore_example examples/sample.jpg
./build/bin/RelWithDebInfo/loomcore_bench --warmup 20 --iters 100
./build/bin/RelWithDebInfo/loomcore_reload_demo          # hot-swaps the graph 6x under continuous load
./build/bin/RelWithDebInfo/loomcore_trace_export logs/loomcore.jsonl trace.json   # -> ui.perfetto.dev

# Python bindings (point at your build dir once; see bindings/python/loomcore/__init__.py)
export LOOMCORE_BUILD_DIR=$PWD/build      # PowerShell: $env:LOOMCORE_BUILD_DIR = "$PWD/build"
python examples/run_example.py
```

`docs/CLAIMS.md` has the full, up-to-date list of what to run to check
each specific claim this project makes about itself — including the C
API smoke test and the from-scratch `find_package(Loomcore)` consumer in
`examples/consumer/`, neither of which is part of the default build
target list above.

(Adjust `RelWithDebInfo` → nothing, or drop the `<Config>` path segment,
on a single-config Ninja/Makefiles build.)

## Troubleshooting

- **"failed to download ONNX Runtime"** — no network access at configure
  time, or a corporate proxy blocking GitHub releases. Download the
  matching `onnxruntime-<platform>-<version>.{zip,tgz}` yourself from
  https://github.com/microsoft/onnxruntime/releases and pass
  `-DONNXRUNTIME_ROOT_DIR=<extracted path>` `-DLOOMCORE_DOWNLOAD_ONNXRUNTIME=OFF`.
- **Python bindings silently skipped** — `Python3` development headers
  weren't found (`find_package(Python3 COMPONENTS Development)` failed).
  On Debian/Ubuntu install `python3-dev`; on Windows, use a full Python
  install (not the Microsoft Store app-execution-alias stub).
- **`loomcore_example`/`loomcore_bench` can't open a `models/*.onnx`
  file** — you skipped step 1. Run `python scripts/prepare_all_models.py`
  from the repo root first.
- **A `_loomcore` import error mentioning `LOOMCORE_BUILD_DIR`** — see
  `bindings/python/loomcore/__init__.py`'s docstring; it searches
  `build/bin/<config>/` relative to the repo root by default, or wherever
  `LOOMCORE_BUILD_DIR` points.
