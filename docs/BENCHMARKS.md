# Benchmarks: FP32 vs INT8 (Milestone 3)

`benchmarks/latency_bench.cpp` loads both precisions of both reference
models through Loomcore's own public `loomcore::ModelVariant` (the exact
class the scheduler uses to run one batch — see
[ARCHITECTURE.md](ARCHITECTURE.md#modelvariant--modelnode)), feeds each a
fixed-shape synthetic input, and reports mean/p50/p95 wall-clock latency
over N iterations after a warmup period. Build and run it with:

```
cmake --build build --config Release --target loomcore_bench
./build/bin/Release/loomcore_bench --warmup 20 --iters 100
```

It also writes a CSV to `benchmarks/results/latency_<timestamp>.csv`.

**This measures latency only, never accuracy.** The INT8 MobileNetV2 was
calibrated with synthetic random activations (`scripts/quantize_mobilenet.py`),
not a real image dataset, because this project is about the orchestration
layer, not model quality — see `docs/ARCHITECTURE.md` "Quantization" for
why static vs. dynamic quantization was chosen per-model.

## Measured results

Machine: AMD Ryzen 9 6900HX (Zen 3+, 8 cores / 16 threads, **no AVX-512**),
Windows 11, ONNX Runtime 1.30.0 CPU execution provider, `intra_op_threads=1`
per session (Loomcore's scheduler — not ONNX Runtime's own intra-op
threading — is the intended source of parallelism; see ARCHITECTURE.md
"Scheduler"), single-item batches, 20 warmup + 100 measured iterations.

| model       | precision | mean (ms) | p50 (ms) | p95 (ms) | n   |
|-------------|-----------|-----------|----------|----------|-----|
| mobilenetv2 | FP32      | 8.193     | 8.172    | 8.419    | 100 |
| mobilenetv2 | INT8      | 8.734     | 8.779    | 9.118    | 100 |
| bert_tiny   | FP32      | 0.281     | 0.278    | 0.295    | 100 |
| bert_tiny   | INT8      | 0.250     | 0.247    | 0.261    | 100 |

INT8 p50 speedup vs FP32: **0.93x** for mobilenetv2 (i.e. *slower*), **1.13x**
for bert_tiny.

## Why INT8 MobileNetV2 isn't faster here — and why that's the honest result

It would be easy to assume "INT8 is always faster" and either fabricate a
speedup number or quietly not run the benchmark on hardware that
contradicts it. It doesn't hold on this machine, and the reason is a real,
explainable property of the stack rather than a bug in Loomcore:

- **No AVX-512 VNNI.** ONNX Runtime's biggest x86 INT8 GEMM/Conv speedups
  come from VNNI (`VPDPBUSD`-family instructions), available on
  Intel Cascade Lake and newer, and on AMD starting with Zen 4 (AVX-512).
  The 6900HX is Zen 3+ — no AVX-512 at all — so its INT8 kernels fall back
  to a less-specialized AVX2 path.
- **Static QDQ quantization adds nodes.** MobileNetV2 was statically
  quantized (`QuantFormat.QDQ`): every quantized op is now flanked by
  `QuantizeLinear`/`DequantizeLinear` nodes. Without a fast INT8 compute
  kernel to amortize that overhead against, the extra nodes are a net
  cost.
- **Batch size 1, single-threaded.** At this scale, per-op dispatch
  overhead is a proportionally larger share of total latency than at
  larger batch sizes, which is exactly where INT8's per-element compute
  advantage would otherwise show up.

bert_tiny's small INT8 win (1.13x) is consistent with this: it was
quantized *dynamically* (`quantize_dynamic`, only MatMul/Gemm weights),
which adds no QDQ nodes and targets exactly the op types dynamic
quantization is designed for on transformer encoders.

**Takeaway for the router:** `LatencyBudgetPolicy` (see
[router.h](../include/loomcore/router.h)) still demonstrates the
*mechanism* of trading precision for latency under a tight time budget
correctly and is unit-tested end-to-end
(`tests/test_scheduler.cpp`, "LatencyBudgetPolicy's INT8 decision is
actually applied by the scheduler"). Whether that trade is actually worth
taking is hardware- and model-dependent — which is precisely why it's a
runtime *policy* decision rather than a compile-time constant, and why
re-running this benchmark on your own target hardware before trusting the
policy's threshold in production is the right instinct.
