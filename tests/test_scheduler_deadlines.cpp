// Integration tests for the Phase 04 scheduler features (see
// docs/ARCHITECTURE.md "Deadlines"): admission control, deadline
// cancellation, and real shutdown() semantics — plus the batching
// correctness fix (PendingKey now includes shape, not just node+precision)
// from docs/ARCHITECTURE.md "Scheduler". All against real ModelVariants
// over the tiny fixture ONNX graphs in assets/, same style as
// test_scheduler.cpp.
#include <doctest/doctest.h>

#include <atomic>
#include <future>
#include <thread>
#include <vector>

#include "loomcore/logger.h"
#include "loomcore/planner.h"
#include "loomcore/router.h"
#include "loomcore/scheduler.h"

using namespace loomcore;

namespace {

#ifndef LOOMCORE_TEST_ASSETS_DIR
#define LOOMCORE_TEST_ASSETS_DIR "assets"
#endif

std::string assetPath(const char* name) { return std::string(LOOMCORE_TEST_ASSETS_DIR) + "/" + name; }

struct TestGraph {
    Graph graph;
    std::map<std::string, ModelNode> nodes;
};

// Single node "a": y = identity(x), over the *variable-width* fixture (its
// non-batch dimension is symbolic, unlike test_identity.onnx's fixed-4
// input) — see scripts/gen_test_fixtures.py.
TestGraph buildVariableWidthGraph(size_t max_batch_size, int batch_window_ms) {
    TestGraph tg;
    NodeConfig a;
    a.id = "a";
    a.backend = Backend::CPU;
    a.max_batch_size = max_batch_size;
    a.batch_window_ms = batch_window_ms;
    a.variants.push_back(VariantConfig{Precision::FP32, assetPath("test_variable_width_identity.onnx")});
    a.binder = [](const NodeExecutionContext& ctx) { return std::vector<NamedTensor>{ctx.graph_inputs->at("x")}; };
    tg.graph.addNode(a);

    ModelNode ma;
    ma.config = tg.graph.node("a");
    for (const auto& v : ma.config.variants) {
        ma.variants[v.precision] = std::make_shared<ModelVariant>(Environment::shared(), v.model_path, v.precision, 1);
    }
    tg.nodes["a"] = std::move(ma);
    return tg;
}

TestGraph buildSingleIdentityGraph(size_t max_batch_size = 1) {
    TestGraph tg;
    NodeConfig a;
    a.id = "a";
    a.backend = Backend::CPU;
    a.max_batch_size = max_batch_size;
    a.variants.push_back(VariantConfig{Precision::FP32, assetPath("test_identity.onnx")});
    a.variants.push_back(VariantConfig{Precision::INT8, assetPath("test_identity.onnx")});
    a.binder = [](const NodeExecutionContext& ctx) { return std::vector<NamedTensor>{ctx.graph_inputs->at("x")}; };
    tg.graph.addNode(a);

    ModelNode ma;
    ma.config = tg.graph.node("a");
    for (const auto& v : ma.config.variants) {
        ma.variants[v.precision] = std::make_shared<ModelVariant>(Environment::shared(), v.model_path, v.precision, 1);
    }
    tg.nodes["a"] = std::move(ma);
    return tg;
}

} // namespace

TEST_CASE("Batching keys on shape, not just (node, precision): mixed widths never cross-contaminate") {
    // batch_window_ms is long and max_batch_size is large enough that,
    // under the pre-fix key of (node_id, precision) alone, concurrent
    // requests of *different* widths would have been offered to each
    // other as batch-mates — concatBatch requires every non-batch
    // dimension to match, so that would throw and fail every job in the
    // batch, not just the mismatched ones. With shape folded into the
    // key, width-4 and width-8 requests simply land in separate pending
    // queues and each gets back exactly its own input, unchanged.
    auto tg = buildVariableWidthGraph(/*max_batch_size=*/8, /*batch_window_ms=*/25);
    MetricsRegistry metrics;
    Scheduler sched(tg.graph, tg.nodes, metrics, std::make_shared<CompositeRouter>(), SchedulerConfig{});

    std::vector<std::future<JobResult>> futures;
    std::vector<std::vector<float>> expected;
    for (int i = 0; i < 6; ++i) {
        bool wide = (i % 2 == 0);
        std::vector<float> data;
        if (wide) {
            data = {float(i), float(i) + 1, float(i) + 2, float(i) + 3, float(i) + 4, float(i) + 5, float(i) + 6,
                    float(i) + 7};
        } else {
            data = {float(i), float(i) + 1, float(i) + 2, float(i) + 3};
        }
        TensorMap inputs;
        inputs["x"] = NamedTensor::makeFloat("x", {1, static_cast<int64_t>(data.size())}, data);
        expected.push_back(data);
        futures.push_back(sched.submitJob(inputs));
    }

    for (size_t i = 0; i < futures.size(); ++i) {
        auto result = futures[i].get();
        REQUIRE(result.count("a") == 1);
        REQUIRE(result["a"].size() == 1);
        CHECK(result["a"][0].f32 == expected[i]);
        CHECK(result["a"][0].shape[1] == static_cast<int64_t>(expected[i].size()));
    }
}

TEST_CASE("Scheduler::shutdown() actually stops accepting new jobs") {
    auto tg = buildSingleIdentityGraph();
    MetricsRegistry metrics;
    Scheduler sched(tg.graph, tg.nodes, metrics, std::make_shared<CompositeRouter>(), SchedulerConfig{});

    TensorMap inputs;
    inputs["x"] = NamedTensor::makeFloat("x", {1, 4}, {1, 2, 3, 4});
    auto ok = sched.runSync(inputs); // works before shutdown
    CHECK(ok["a"][0].f32 == std::vector<float>{1, 2, 3, 4});

    sched.shutdown();
    sched.shutdown(); // idempotent: calling twice must not throw or misbehave

    auto future = sched.submitJob(inputs);
    CHECK_THROWS_AS(future.get(), LoomcoreError);
}

TEST_CASE("Admission control rejects a job the precision planner judges undeliverable") {
    SchedulerConfig cfg;
    cfg.enable_admission_control = true;
    auto tg = buildSingleIdentityGraph();
    MetricsRegistry metrics;
    Scheduler sched(tg.graph, tg.nodes, metrics, std::make_shared<CompositeRouter>(), cfg);

    TensorMap inputs;
    inputs["x"] = NamedTensor::makeFloat("x", {1, 4}, {1, 2, 3, 4});

    // Cold registry: has_estimate is false, so admission control has
    // nothing to judge by yet and must let the job through (this also
    // warms up metrics for node "a" at both precisions).
    auto warm = sched.runSync(inputs, 0, /*time_budget_ms=*/1000.0);
    CHECK(warm["a"][0].f32 == std::vector<float>{1, 2, 3, 4});
    // Second call so both FP32 *and* INT8 get a measured sample (the
    // router never picked INT8 above, so seed it directly the way the
    // scheduler itself would via its precision-qualified metrics key).
    metrics.recordLatency(precisionMetricsKey("a", Precision::FP32), metrics.stats("a").mean_ms);
    metrics.recordLatency(precisionMetricsKey("a", Precision::INT8), metrics.stats("a").mean_ms);

    // An impossibly tight budget (far below any measured cost, and node
    // "a"'s INT8 variant here is numerically identical to FP32 — see
    // buildSingleIdentityGraph — so there is zero achievable saving)
    // must be rejected outright, with nothing ever dispatched.
    Logger::instance().clear();
    auto future = sched.submitJob(inputs, 0, /*time_budget_ms=*/0.0001);
    CHECK_THROWS_AS(future.get(), JobRejectedError);

    auto logs = Logger::instance().recentLines(50);
    bool saw_scheduled = false;
    for (const auto& line : logs) {
        if (line.find("\"event\":\"node_scheduled\"") != std::string::npos) saw_scheduled = true;
    }
    CHECK_FALSE(saw_scheduled); // admission control must reject *before* dispatching anything
}

TEST_CASE("Deadline cancellation fails the job and never double-settles the promise under repeated stress") {
    // Regression coverage for the settled-CAS arbitration in
    // Scheduler::Impl (see JobState::settled in scheduler.cpp): the
    // deadline reaper can call failJob() for a job at the same moment
    // that job's last node is completing successfully on a lane worker.
    // Before that arbitration existed, whichever side ran second would
    // call promise.set_value()/set_exception() on an already-satisfied
    // std::promise, which throws std::future_error on a thread with no
    // caller frame to catch it — i.e. std::terminate. Run enough trials
    // with the deadline set right at the node's own typical latency that
    // the race window is actually exercised, not just theoretically
    // possible, and confirm every single trial resolves to exactly one
    // outcome (success or DeadlineExceededError), never a crash.
    SchedulerConfig cfg;
    cfg.enable_deadline_cancellation = true;
    cfg.deadline_reaper_poll_ms = 1.0;
    auto tg = buildSingleIdentityGraph();
    MetricsRegistry metrics;
    Scheduler sched(tg.graph, tg.nodes, metrics, std::make_shared<CompositeRouter>(), cfg);

    TensorMap inputs;
    inputs["x"] = NamedTensor::makeFloat("x", {1, 4}, {1, 2, 3, 4});

    // Warm up so we know roughly how long node "a" takes.
    for (int i = 0; i < 5; ++i) sched.runSync(inputs);
    double typical_ms = std::max(0.05, metrics.stats("a").mean_ms);

    int successes = 0, deadline_failures = 0;
    constexpr int kTrials = 200;
    for (int i = 0; i < kTrials; ++i) {
        auto future = sched.submitJob(inputs, 0, /*time_budget_ms=*/typical_ms);
        try {
            auto result = future.get();
            REQUIRE(result.count("a") == 1);
            ++successes;
        } catch (const DeadlineExceededError&) {
            ++deadline_failures;
        }
        // Anything else propagating (in particular std::future_error, or
        // the process having already terminated) fails this test loudly.
    }
    CHECK(successes + deadline_failures == kTrials);
}

TEST_CASE("PlannedPrecisionPolicy driven end-to-end by the scheduler's own precision plan") {
    SchedulerConfig cfg;
    cfg.enable_precision_planning = true;
    auto tg = buildSingleIdentityGraph();
    MetricsRegistry metrics;
    auto router = std::make_shared<CompositeRouter>();
    router->add(std::make_shared<PlannedPrecisionPolicy>());
    Scheduler sched(tg.graph, tg.nodes, metrics, router, cfg);

    TensorMap inputs;
    inputs["x"] = NamedTensor::makeFloat("x", {1, 4}, {1, 2, 3, 4});
    for (int i = 0; i < 5; ++i) sched.runSync(inputs); // warm FP32 stats
    // Node "a"'s INT8 variant is the identical file (see
    // buildSingleIdentityGraph), so it produces INT8-keyed samples too the
    // moment anything routes it there — seed that directly so the planner
    // has both precisions to compare, matching what a real INT8 variant
    // with genuinely lower latency would look like from the planner's
    // point of view (it only ever reads MetricsRegistry, never re-measures).
    double fp32_ms = metrics.stats("a").mean_ms;
    for (int i = 0; i < 5; ++i) metrics.recordLatency(precisionMetricsKey("a", Precision::FP32), fp32_ms);
    for (int i = 0; i < 5; ++i) metrics.recordLatency(precisionMetricsKey("a", Precision::INT8), fp32_ms / 4.0);

    Logger::instance().clear();
    // Budget tighter than FP32 but comfortably above INT8: the plan must
    // downgrade node "a", and the scheduler must actually run it at INT8.
    auto result = sched.runSync(inputs, 0, /*time_budget_ms=*/fp32_ms / 2.0);
    CHECK(result["a"][0].f32 == std::vector<float>{1, 2, 3, 4});

    auto logs = Logger::instance().recentLines(50);
    bool ran_int8 = false;
    for (const auto& line : logs) {
        if (line.find("\"event\":\"node_completed\"") != std::string::npos &&
            line.find("\"node_id\":\"a\"") != std::string::npos && line.find("\"precision\":\"INT8\"") != std::string::npos) {
            ran_int8 = true;
        }
    }
    CHECK(ran_int8);
}
