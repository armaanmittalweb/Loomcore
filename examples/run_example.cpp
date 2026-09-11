// Loomcore reference pipeline: MobileNetV2 (image -> class logits) feeding
// bert_tiny (predicted-label text -> a 128-dim sentence embedding), wired
// together as a 2-node DAG with an agentic router deciding, per job:
//   - whether to run bert_tiny at all (ConfidenceGatePolicy: skip it once
//     mobilenet is already confident — no point describing what you're
//     already sure of),
//   - which precision to run mobilenet at (LatencyBudgetPolicy: fall back
//     to INT8 under a tight time budget),
//   - which simulated backend lane to use (LoadAwareBackendPolicy).
//
// See docs/ARCHITECTURE.md for the full design and README.md "Running the
// example" for build/run instructions.
#include "loomcore/runtime.h"
#include "loomcore/tokenizer.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <vector>

using namespace loomcore;

namespace {

std::vector<std::string> loadLabels(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open labels file: " + path);
    std::vector<std::string> labels;
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (!line.empty()) labels.push_back(line);
    }
    return labels;
}

// exp(x_max - x_max) / sum(exp(x_i - x_max)) == 1 / sum(...): the
// probability mass on the arg-max class, i.e. the model's own confidence
// in its top prediction.
float softmaxMaxProb(const NamedTensor& logits) {
    if (logits.dtype != DType::Float32 || logits.f32.empty()) return 0.0f;
    float max_logit = *std::max_element(logits.f32.begin(), logits.f32.end());
    double sum = 0.0;
    for (float v : logits.f32) sum += std::exp(static_cast<double>(v - max_logit));
    return static_cast<float>(1.0 / sum);
}

// Real end-to-end preprocessing: decode -> resize to 224x224 -> scale to
// [0,1] -> per-channel ImageNet mean/std normalize -> NCHW. Returns
// nullopt if stb_image can't decode `path` (missing file, unsupported
// format, ...).
std::optional<NamedTensor> loadImageTensor(const std::string& path) {
    int w = 0, h = 0, channels = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &channels, 3);
    if (!pixels) return std::nullopt;

    std::vector<unsigned char> resized(static_cast<size_t>(224) * 224 * 3);
    stbir_resize_uint8_linear(pixels, w, h, 0, resized.data(), 224, 224, 0, STBIR_RGB);
    stbi_image_free(pixels);

    static constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> chw(static_cast<size_t>(3) * 224 * 224);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < 224; ++y) {
            for (int x = 0; x < 224; ++x) {
                unsigned char v = resized[(static_cast<size_t>(y) * 224 + x) * 3 + c];
                float norm = (static_cast<float>(v) / 255.0f - kMean[c]) / kStd[c];
                chw[static_cast<size_t>(c) * 224 * 224 + static_cast<size_t>(y) * 224 + x] = norm;
            }
        }
    }
    return NamedTensor::makeFloat("data", {1, 3, 224, 224}, std::move(chw));
}

// Deterministic fallback so the example always has something plausibly
// scaled to run when no real photo is available.
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

struct DemoSideChannel {
    std::string label = "(unknown)";
    float confidence = 0.0f;
};

} // namespace

int main(int argc, char** argv) {
    std::filesystem::current_path(LOOMCORE_PROJECT_ROOT);

    auto labels = loadLabels("assets/imagenet_labels.txt");
    WordPieceTokenizer tokenizer("models/bert_tiny_tokenizer/vocab.txt", /*max_seq_len=*/16);

    std::string image_path = argc > 1 ? argv[1] : "examples/sample.jpg";
    NamedTensor image_tensor;
    if (auto loaded = loadImageTensor(image_path)) {
        image_tensor = std::move(*loaded);
        std::cout << "Loaded image: " << image_path << "\n";
    } else {
        image_tensor = syntheticImageTensor();
        std::cout << "No decodable image at '" << image_path
                  << "' -- using a synthetic pattern instead.\n"
                     "Pass a real photo path as argv[1] for a meaningful classification, e.g.:\n"
                     "  loomcore_example examples/sample.jpg\n";
    }

    // Populated by the bert_tiny node's ConfidenceExtractor purely so this
    // demo can pretty-print the intermediate classification; JobResult
    // itself only ever carries sink-node outputs by design (see
    // docs/ARCHITECTURE.md "Graph config schema").
    DemoSideChannel side_channel;

    std::map<std::string, NodeInputBinder> binders;
    binders["mobilenet"] = [](const NodeExecutionContext& ctx) {
        return std::vector<NamedTensor>{ctx.graph_inputs->at("data")};
    };
    binders["bert_tiny"] = [&](const NodeExecutionContext& ctx) {
        const NamedTensor& logits = ctx.upstreamFirst("mobilenet");
        auto [idx, score] = argmax(logits);
        (void)score;
        std::string label = (idx >= 0 && static_cast<size_t>(idx) < labels.size()) ? labels[idx] : "object";
        std::string text = "a photo of a " + label;
        std::cout << "  [bert_tiny] embedding text: \"" << text << "\"\n";
        return tokenizer.encodeToTensors(text);
    };

    std::map<std::string, ConfidenceExtractor> confidence_extractors;
    confidence_extractors["bert_tiny"] = [&](const std::vector<NamedTensor>& upstream) -> std::optional<float> {
        if (upstream.empty()) return std::nullopt;
        auto [idx, score] = argmax(upstream.front());
        (void)score;
        side_channel.confidence = softmaxMaxProb(upstream.front());
        side_channel.label = (idx >= 0 && static_cast<size_t>(idx) < labels.size()) ? labels[idx] : "object";
        return side_channel.confidence;
    };

    auto router = std::make_shared<CompositeRouter>();
    router->add(std::make_shared<ConfidenceGatePolicy>(0.85f)); // skip the text node once vision is confident
    router->add(std::make_shared<LatencyBudgetPolicy>(30.0));   // fall back to INT8 mobilenet under a tight budget
    router->add(std::make_shared<LoadAwareBackendPolicy>());    // balance across the two simulated lanes

    Runtime runtime;
    runtime.loadGraph("examples/graph_config.json", binders, confidence_extractors, router);

    std::cout << "\n=== Single synchronous run ===\n";
    TensorMap inputs;
    inputs["data"] = image_tensor;
    auto result = runtime.run(inputs);

    std::cout << "Predicted class : " << side_channel.label << " (confidence " << side_channel.confidence << ")\n";
    if (result.count("bert_tiny") && result["bert_tiny"].size() >= 2) {
        const auto& embedding = result["bert_tiny"][1]; // [input_ids,mask,type] -> [last_hidden_state, pooler_output]
        std::cout << "Text embedding  : [" << embedding.f32[0] << ", " << embedding.f32[1] << ", ...] (dim "
                  << embedding.elementCount() << ")\n";
    } else {
        std::cout << "bert_tiny was skipped by ConfidenceGatePolicy: mobilenet was already confident enough.\n";
    }

    std::cout << "\n=== Concurrent load: 8 jobs submitted at once (exercises batching + priority) ===\n";
    std::vector<std::future<JobResult>> futures;
    futures.reserve(8);
    for (int i = 0; i < 8; ++i) {
        TensorMap job_inputs;
        job_inputs["data"] = image_tensor; // same image; the point is concurrent scheduling, not variety
        futures.push_back(runtime.submit(job_inputs, /*priority=*/i % 3));
    }
    for (auto& f : futures) f.get();

    auto mobilenet_stats = runtime.nodeStats("mobilenet");
    auto bert_stats = runtime.nodeStats("bert_tiny");
    std::cout << "mobilenet  p50=" << mobilenet_stats.p50_ms << "ms  p95=" << mobilenet_stats.p95_ms
              << "ms  n=" << mobilenet_stats.count << "\n";
    std::cout << "bert_tiny  p50=" << bert_stats.p50_ms << "ms  p95=" << bert_stats.p95_ms
              << "ms  n=" << bert_stats.count << "\n";

    std::cout << "\nStructured per-event logs were written to logs/loomcore.jsonl\n";
    return 0;
}
