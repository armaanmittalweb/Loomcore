// Loomcore — live proof of the RCU graph hot-swap (docs/ARCHITECTURE.md
// "Hot-reloading a graph"): a background thread hammers Runtime::run()
// continuously while the main thread calls Runtime::reloadGraph()
// mid-flight, swapping the mobilenet+bert_tiny reference pipeline out for
// itself several times over. Every single job must complete successfully
// throughout — none dropped, none failed, no crash — regardless of
// whether it happened to run against the graph snapshot from before or
// after any given swap.
//
// This is the same reference pipeline run_example.cpp uses (see
// examples/graph_config.json), just driven continuously instead of once,
// specifically to put load on the reload path while it's happening.
#include "loomcore/runtime.h"
#include "loomcore/tokenizer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

using namespace loomcore;

namespace {

NamedTensor syntheticImageTensor() {
    std::vector<float> chw(static_cast<size_t>(3) * 224 * 224);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < 224; ++y) {
            for (int x = 0; x < 224; ++x) {
                float fx = static_cast<float>(x) / 224.0f;
                float fy = static_cast<float>(y) / 224.0f;
                chw[static_cast<size_t>(c) * 224 * 224 + static_cast<size_t>(y) * 224 + x] =
                    0.3f * std::sin(6.0f * fx + static_cast<float>(c)) * std::cos(6.0f * fy - static_cast<float>(c));
            }
        }
    }
    return NamedTensor::makeFloat("data", {1, 3, 224, 224}, std::move(chw));
}

std::map<std::string, NodeInputBinder> makeBinders(WordPieceTokenizer& tokenizer) {
    std::map<std::string, NodeInputBinder> binders;
    binders["mobilenet"] = [](const NodeExecutionContext& ctx) {
        return std::vector<NamedTensor>{ctx.graph_inputs->at("data")};
    };
    binders["bert_tiny"] = [&tokenizer](const NodeExecutionContext& ctx) {
        const NamedTensor& logits = ctx.upstreamFirst("mobilenet");
        auto [idx, score] = argmax(logits);
        (void)score;
        return tokenizer.encodeToTensors("a photo of class " + std::to_string(idx));
    };
    return binders;
}

// graph_config.json declares bert_tiny's "confidence_source": "mobilenet",
// so Runtime::loadGraph requires a registered extractor for it regardless
// of whether this demo's router (none — see main()) actually uses
// ConfidenceGatePolicy. Just extracting the top-class softmax mass, same
// as run_example.cpp, though this demo doesn't do anything with the value.
std::map<std::string, ConfidenceExtractor> makeConfidenceExtractors() {
    std::map<std::string, ConfidenceExtractor> extractors;
    extractors["bert_tiny"] = [](const std::vector<NamedTensor>& upstream) -> std::optional<float> {
        if (upstream.empty() || upstream.front().dtype != DType::Float32 || upstream.front().f32.empty()) {
            return std::nullopt;
        }
        const auto& logits = upstream.front().f32;
        float max_logit = *std::max_element(logits.begin(), logits.end());
        double sum = 0.0;
        for (float v : logits) sum += std::exp(static_cast<double>(v - max_logit));
        return static_cast<float>(1.0 / sum);
    };
    return extractors;
}

} // namespace

int main() {
    std::filesystem::current_path(LOOMCORE_PROJECT_ROOT);

    WordPieceTokenizer tokenizer("models/bert_tiny_tokenizer/vocab.txt", /*max_seq_len=*/16);
    auto binders = makeBinders(tokenizer);
    auto confidence_extractors = makeConfidenceExtractors();

    RuntimeOptions options;
    options.log_to_stdout = false; // this demo's own printed counters are the point, not the JSONL stream
    options.log_file = "logs/reload_demo.jsonl";

    Runtime runtime;
    runtime.loadGraph("examples/graph_config.json", binders, confidence_extractors, nullptr, options);

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> submitted{0}, succeeded{0}, failed{0};

    std::cout << "Loomcore reload demo: hammering Runtime::run() while reloading the graph 6 times.\n\n";

    NamedTensor image = syntheticImageTensor();
    std::vector<std::thread> producers;
    unsigned n_producers = std::max(2u, std::thread::hardware_concurrency() / 4);
    for (unsigned p = 0; p < n_producers; ++p) {
        producers.emplace_back([&] {
            while (!stop.load(std::memory_order_relaxed)) {
                TensorMap inputs;
                inputs["data"] = image;
                ++submitted;
                try {
                    auto result = runtime.run(inputs);
                    if (result.count("bert_tiny") == 1) {
                        ++succeeded;
                    } else {
                        ++failed; // ran without throwing but produced an unexpected result shape
                    }
                } catch (const std::exception&) {
                    ++failed;
                }
            }
        });
    }

    auto t_start = std::chrono::steady_clock::now();
    for (int swap_i = 0; swap_i < 6; ++swap_i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        auto before = succeeded.load();
        auto t0 = std::chrono::steady_clock::now();
        runtime.reloadGraph("examples/graph_config.json", binders, confidence_extractors, nullptr, options);
        double swap_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        std::cout << "t=" << std::fixed << std::setprecision(1) << elapsed_s << "s  reload #" << (swap_i + 1)
                  << " took " << std::setprecision(2) << swap_ms << "ms to build+swap"
                  << "  (throughput just before: " << (succeeded.load() - before) << " jobs since last print)\n";
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    stop.store(true);
    for (auto& t : producers) t.join();

    std::cout << "\nsubmitted: " << submitted.load() << "   succeeded: " << succeeded.load()
              << "   failed: " << failed.load() << "\n";
    std::cout << (failed.load() == 0 && submitted.load() == succeeded.load()
                      ? "PASS: zero jobs lost across 6 concurrent graph reloads.\n"
                      : "FAIL: some job did not complete successfully — see logs/reload_demo.jsonl\n");
    return failed.load() == 0 ? 0 : 1;
}
