// Integration test for Runtime::reloadGraph (see docs/ARCHITECTURE.md
// "Hot-reloading a graph"): the RCU-style GraphSnapshot design that fixes
// what used to be a use-after-free (Runtime::loadGraph mutating
// impl_->graph/impl_->nodes in place while the previous Scheduler's lane
// threads still referenced them — reachable simply by calling loadGraph()
// a second time on a Runtime already handling traffic). Proves the fix
// under real concurrent load, not just "it compiles": jobs are submitted
// continuously from a background thread while a reload happens mid-flight,
// and every single one must resolve successfully — none dropped, none
// throwing, no crash — regardless of which graph (old or new) it actually
// ran against.
#include <doctest/doctest.h>

#include <atomic>
#include <fstream>
#include <thread>

#include "loomcore/runtime.h"

using namespace loomcore;

namespace {

#ifndef LOOMCORE_TEST_ASSETS_DIR
#define LOOMCORE_TEST_ASSETS_DIR "assets"
#endif

std::string assetPath(const char* name) { return std::string(LOOMCORE_TEST_ASSETS_DIR) + "/" + name; }

// Writes a minimal single-node graph config (node "a": identity over
// test_identity.onnx) to `path`, so loadGraph/reloadGraph have a real file
// to parse — Runtime's JSON schema is exercised here, not just Scheduler's
// C++ construction API (already covered by test_scheduler*.cpp).
void writeSingleNodeConfig(const std::string& path) {
    std::ofstream out(path);
    out << R"({
  "nodes": [
    { "id": "a", "backend": "CPU", "max_batch_size": 1,
      "depends_on": [],
      "variants": [ { "precision": "FP32", "model_path": ")"
        << assetPath("test_identity.onnx") << R"(" } ] }
  ]
})";
}

} // namespace

TEST_CASE("Runtime::reloadGraph swaps graphs under continuous concurrent load with zero jobs lost") {
    std::string config_path = std::string(LOOMCORE_TEST_ASSETS_DIR) + "/reload_test_config.json";
    writeSingleNodeConfig(config_path);

    Runtime runtime;
    std::map<std::string, NodeInputBinder> binders;
    binders["a"] = [](const NodeExecutionContext& ctx) { return std::vector<NamedTensor>{ctx.graph_inputs->at("x")}; };
    runtime.loadGraph(config_path, binders);

    std::atomic<bool> stop{false};
    std::atomic<int> submitted{0};
    std::atomic<int> succeeded{0};
    std::atomic<int> failed{0};

    // Background producer: submits jobs back-to-back for the whole test.
    std::thread producer([&] {
        while (!stop.load()) {
            TensorMap inputs;
            inputs["x"] = NamedTensor::makeFloat("x", {1, 4}, {1, 2, 3, 4});
            ++submitted;
            try {
                auto result = runtime.run(inputs);
                if (result.count("a") == 1 && result["a"][0].f32 == std::vector<float>{1, 2, 3, 4}) {
                    ++succeeded;
                } else {
                    ++failed;
                }
            } catch (...) {
                ++failed;
            }
        }
    });

    // Let a handful of jobs land on the *old* graph, then reload — building
    // and swapping in an entirely new snapshot — several times in a row
    // while the producer keeps submitting without ever pausing for it.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    for (int i = 0; i < 5; ++i) {
        runtime.reloadGraph(config_path, binders);
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }

    stop.store(true);
    producer.join();

    CHECK(submitted.load() > 0);
    CHECK(failed.load() == 0);
    CHECK(succeeded.load() == submitted.load());

    std::remove(config_path.c_str());
}

TEST_CASE("Runtime::reloadGraph requires loadGraph to have run first") {
    Runtime runtime;
    std::map<std::string, NodeInputBinder> binders;
    binders["a"] = [](const NodeExecutionContext& ctx) { return std::vector<NamedTensor>{ctx.graph_inputs->at("x")}; };
    CHECK_THROWS_AS(runtime.reloadGraph(assetPath("reload_test_config.json"), binders), LoomcoreError);
}
