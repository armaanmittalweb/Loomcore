// FP32 vs INT8 latency benchmark — Milestone 3.
//
// Measures loomcore::ModelVariant::run() latency directly (the same call
// path the scheduler uses per batch item) for both precisions of both
// reference models, entirely through Loomcore's own public API — no raw
// ONNX Runtime calls here, which doubles as a check that the ORT
// dependency really is fully encapsulated inside loomcore_core (see
// loomcore/environment.h).
//
// This measures latency ONLY. It does not check prediction accuracy: the
// INT8 MobileNetV2 was calibrated on synthetic random activations (see
// scripts/quantize_mobilenet.py) specifically because this project is
// about the orchestration layer, not model quality — see
// docs/BENCHMARKS.md for that caveat restated next to the numbers it
// affects.
#include "loomcore/metrics.h"
#include "loomcore/model_node.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace loomcore;
using clock_type = std::chrono::steady_clock;

namespace {

struct BenchCase {
    std::string model_name;
    std::string fp32_path;
    std::string int8_path;
    std::vector<NamedTensor> (*make_inputs)();
};

std::vector<NamedTensor> mobilenetInputs() {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(static_cast<size_t>(3) * 224 * 224);
    for (auto& v : data) v = dist(rng);
    return {NamedTensor::makeFloat("data", {1, 3, 224, 224}, std::move(data))};
}

std::vector<NamedTensor> bertTinyInputs() {
    constexpr int64_t kSeqLen = 16;
    std::vector<int64_t> ids(kSeqLen, 1000); // arbitrary valid vocab ids; content doesn't affect latency
    std::vector<int64_t> mask(kSeqLen, 1);
    std::vector<int64_t> type_ids(kSeqLen, 0);
    return {
        NamedTensor::makeInt64("input_ids", {1, kSeqLen}, ids),
        NamedTensor::makeInt64("attention_mask", {1, kSeqLen}, mask),
        NamedTensor::makeInt64("token_type_ids", {1, kSeqLen}, type_ids),
    };
}

struct Timing {
    MetricsRegistry::Stats stats;
    double total_ms = 0.0;
};

Timing timeVariant(ModelVariant& variant, const std::vector<NamedTensor>& inputs, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) (void)variant.run(inputs);

    MetricsRegistry metrics(static_cast<size_t>(iters));
    auto t_start = clock_type::now();
    for (int i = 0; i < iters; ++i) {
        auto t0 = clock_type::now();
        (void)variant.run(inputs);
        double ms = std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
        metrics.recordLatency("bench", ms);
    }
    double total_ms = std::chrono::duration<double, std::milli>(clock_type::now() - t_start).count();
    return {metrics.stats("bench"), total_ms};
}

void printRow(std::ostream& os, const std::string& model, const std::string& precision, const Timing& t) {
    os << std::left << std::setw(14) << model << std::setw(8) << precision << std::fixed << std::setprecision(3)
       << std::setw(12) << t.stats.mean_ms << std::setw(12) << t.stats.p50_ms << std::setw(12) << t.stats.p95_ms
       << std::setw(10) << t.stats.count << "\n";
}

} // namespace

int main(int argc, char** argv) {
    std::filesystem::current_path(LOOMCORE_PROJECT_ROOT);

    int warmup = 10;
    int iters = 50;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
        if (arg == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    }

    std::vector<BenchCase> cases = {
        {"mobilenetv2", "models/mobilenetv2.onnx", "models/mobilenetv2.int8.onnx", mobilenetInputs},
        {"bert_tiny", "models/bert_tiny.onnx", "models/bert_tiny.int8.onnx", bertTinyInputs},
    };

    std::cout << "Loomcore FP32 vs INT8 latency benchmark  (warmup=" << warmup << ", iters=" << iters << ")\n\n";
    std::cout << std::left << std::setw(14) << "model" << std::setw(8) << "prec" << std::setw(12) << "mean_ms"
              << std::setw(12) << "p50_ms" << std::setw(12) << "p95_ms" << std::setw(10) << "n" << "\n";
    std::cout << std::string(68, '-') << "\n";

    std::filesystem::create_directories("benchmarks/results");
    auto now = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream fname;
    fname << "benchmarks/results/latency_" << now_t << ".csv";
    std::ofstream csv(fname.str());
    csv << "model,precision,mean_ms,p50_ms,p95_ms,n\n";

    for (const auto& c : cases) {
        auto inputs = c.make_inputs();
        ModelVariant fp32(Environment::shared(), c.fp32_path, Precision::FP32, /*intra_op_threads=*/1);
        ModelVariant int8(Environment::shared(), c.int8_path, Precision::INT8, /*intra_op_threads=*/1);

        auto fp32_timing = timeVariant(fp32, inputs, warmup, iters);
        auto int8_timing = timeVariant(int8, inputs, warmup, iters);

        printRow(std::cout, c.model_name, "FP32", fp32_timing);
        printRow(std::cout, c.model_name, "INT8", int8_timing);
        csv << c.model_name << ",FP32," << fp32_timing.stats.mean_ms << "," << fp32_timing.stats.p50_ms << ","
            << fp32_timing.stats.p95_ms << "," << fp32_timing.stats.count << "\n";
        csv << c.model_name << ",INT8," << int8_timing.stats.mean_ms << "," << int8_timing.stats.p50_ms << ","
            << int8_timing.stats.p95_ms << "," << int8_timing.stats.count << "\n";

        double speedup = fp32_timing.stats.p50_ms / std::max(1e-6, int8_timing.stats.p50_ms);
        std::cout << "  -> INT8 p50 speedup vs FP32: " << std::fixed << std::setprecision(2) << speedup << "x\n";
    }

    std::cout << "\nWrote " << fname.str() << "\n";
    return 0;
}
