// Integration tests for Scheduler: real ModelVariants over the tiny fixture
// ONNX graphs in assets/ (see scripts/gen_test_fixtures.py), so dependency
// ordering, cross-job batching + slicing, and routing/skip handling are all
// exercised against the actual ONNX Runtime call path rather than a mock.
#include <doctest/doctest.h>

#include <algorithm>
#include <future>
#include <initializer_list>
#include <memory>
#include <vector>

#include "loomcore/logger.h"
#include "loomcore/router.h"
#include "loomcore/scheduler.h"

using namespace loomcore;

namespace {

#ifndef LOOMCORE_TEST_ASSETS_DIR
#define LOOMCORE_TEST_ASSETS_DIR "assets"
#endif

std::string assetPath(const char* name) { return std::string(LOOMCORE_TEST_ASSETS_DIR) + "/" + name; }

// node "a": y = identity(x). node "b": y = a_out + a_out (depends on "a").
struct TestGraph {
    Graph graph;
    std::map<std::string, ModelNode> nodes;
};

TestGraph buildIdentityThenDoubleGraph(size_t max_batch_size = 1, bool a_has_int8_too = false) {
    TestGraph tg;

    NodeConfig a;
    a.id = "a";
    a.backend = Backend::CPU;
    a.max_batch_size = max_batch_size;
    a.variants.push_back(VariantConfig{Precision::FP32, assetPath("test_identity.onnx")});
    if (a_has_int8_too) {
        // Same underlying file under the INT8 key: this suite tests routing
        // *plumbing* (does the scheduler pick the variant the router asked
        // for), not real quantized numerics — those are covered by
        // benchmarks/latency_bench.cpp against the real MobileNetV2 pair.
        a.variants.push_back(VariantConfig{Precision::INT8, assetPath("test_identity.onnx")});
    }
    a.binder = [](const NodeExecutionContext& ctx) { return std::vector<NamedTensor>{ctx.graph_inputs->at("x")}; };

    NodeConfig b;
    b.id = "b";
    b.backend = Backend::CPU;
    b.max_batch_size = max_batch_size;
    b.variants.push_back(VariantConfig{Precision::FP32, assetPath("test_double.onnx")});
    b.binder = [](const NodeExecutionContext& ctx) { return std::vector<NamedTensor>{ctx.upstreamFirst("a")}; };
    b.depends_on = {"a"};

    tg.graph.addNode(a);
    tg.graph.addNode(b);

    ModelNode ma;
    ma.config = tg.graph.node("a");
    for (const auto& v : ma.config.variants) {
        ma.variants[v.precision] = std::make_shared<ModelVariant>(Environment::shared(), v.model_path, v.precision, 1);
    }
    ModelNode mb;
    mb.config = tg.graph.node("b");
    for (const auto& v : mb.config.variants) {
        mb.variants[v.precision] = std::make_shared<ModelVariant>(Environment::shared(), v.model_path, v.precision, 1);
    }
    tg.nodes["a"] = std::move(ma);
    tg.nodes["b"] = std::move(mb);
    return tg;
}

bool anyLineContainsAll(const std::vector<std::string>& lines, std::initializer_list<const char*> needles) {
    for (const auto& line : lines) {
        bool all = true;
        for (auto* n : needles) {
            if (line.find(n) == std::string::npos) {
                all = false;
                break;
            }
        }
        if (all) return true;
    }
    return false;
}

class AlwaysSkipPolicy : public IRoutingPolicy {
public:
    explicit AlwaysSkipPolicy(std::string target) : target_(std::move(target)) {}
    std::string name() const override { return "AlwaysSkipPolicy"; }
    std::optional<RoutingDecision> decide(const RoutingContext& ctx) const override {
        if (!ctx.node || ctx.node->id != target_) return std::nullopt;
        RoutingDecision d;
        d.policy_name = name();
        d.skip = true;
        d.reason = "test forced skip";
        return d;
    }

private:
    std::string target_;
};

} // namespace

TEST_CASE("Scheduler runs a 2-node DAG in dependency order and produces the right sink output") {
    auto tg = buildIdentityThenDoubleGraph();
    MetricsRegistry metrics;
    Scheduler sched(tg.graph, tg.nodes, metrics, std::make_shared<CompositeRouter>(), SchedulerConfig{});

    TensorMap inputs;
    inputs["x"] = NamedTensor::makeFloat("x", {1, 4}, {1, 2, 3, 4});
    auto result = sched.runSync(inputs);

    REQUIRE(result.count("b") == 1);
    REQUIRE(result["b"].size() == 1);
    CHECK(result["b"][0].f32 == std::vector<float>{2, 4, 6, 8});
    // "a" is not a sink (it has a dependent), so it should not appear in
    // the final JobResult even though it executed.
    CHECK(result.count("a") == 0);
}

TEST_CASE("Scheduler batches concurrent jobs on the same node without cross-contaminating results") {
    auto tg = buildIdentityThenDoubleGraph(/*max_batch_size=*/4);
    MetricsRegistry metrics;
    Scheduler sched(tg.graph, tg.nodes, metrics, std::make_shared<CompositeRouter>(), SchedulerConfig{});

    constexpr int kJobs = 12;
    std::vector<std::future<JobResult>> futures;
    for (int i = 0; i < kJobs; ++i) {
        TensorMap inputs;
        float base = static_cast<float>(i);
        inputs["x"] = NamedTensor::makeFloat("x", {1, 4}, {base, base + 1, base + 2, base + 3});
        futures.push_back(sched.submitJob(inputs));
    }
    for (int i = 0; i < kJobs; ++i) {
        auto result = futures[i].get();
        float base = static_cast<float>(i);
        std::vector<float> expected{2 * base, 2 * (base + 1), 2 * (base + 2), 2 * (base + 3)};
        REQUIRE(result.count("b") == 1);
        CHECK(result["b"][0].f32 == expected);
    }
}

TEST_CASE("A routing policy that skips a node is honored end-to-end") {
    auto tg = buildIdentityThenDoubleGraph();
    MetricsRegistry metrics;
    Logger::instance().clear();
    auto router = std::make_shared<CompositeRouter>();
    router->add(std::make_shared<AlwaysSkipPolicy>("b"));
    Scheduler sched(tg.graph, tg.nodes, metrics, router, SchedulerConfig{});

    TensorMap inputs;
    inputs["x"] = NamedTensor::makeFloat("x", {1, 4}, {1, 2, 3, 4});
    auto result = sched.runSync(inputs);

    REQUIRE(result.count("b") == 1);
    CHECK(result["b"].empty());
    auto logs = Logger::instance().recentLines(200);
    CHECK(anyLineContainsAll(logs, {"\"event\":\"node_skipped\"", "\"node_id\":\"b\""}));
}

TEST_CASE("LatencyBudgetPolicy's INT8 decision is actually applied by the scheduler") {
    auto tg = buildIdentityThenDoubleGraph(/*max_batch_size=*/1, /*a_has_int8_too=*/true);
    MetricsRegistry metrics;
    Logger::instance().clear();
    auto router = std::make_shared<CompositeRouter>();
    router->add(std::make_shared<LatencyBudgetPolicy>(1000.0));
    Scheduler sched(tg.graph, tg.nodes, metrics, router, SchedulerConfig{});

    TensorMap inputs;
    inputs["x"] = NamedTensor::makeFloat("x", {1, 4}, {1, 2, 3, 4});
    // A 1ms budget is far below the 1000ms threshold, so node "a" (which
    // has an INT8 variant registered) should be downgraded.
    auto result = sched.runSync(inputs, /*priority=*/0, /*time_budget_ms=*/1.0);
    REQUIRE(result.count("b") == 1);
    CHECK(result["b"][0].f32 == std::vector<float>{2, 4, 6, 8}); // same file either way; plumbing is what's tested

    auto logs = Logger::instance().recentLines(200);
    CHECK(anyLineContainsAll(logs, {"\"event\":\"node_completed\"", "\"node_id\":\"a\"", "\"precision\":\"INT8\""}));
}
