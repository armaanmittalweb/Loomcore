#include <doctest/doctest.h>

#include "loomcore/router.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace loomcore;

namespace {
NodeConfig nodeWithVariants(bool with_int8) {
    NodeConfig cfg;
    cfg.id = "n";
    cfg.backend = Backend::CPU;
    cfg.variants.push_back(VariantConfig{Precision::FP32, "a.onnx"});
    if (with_int8) cfg.variants.push_back(VariantConfig{Precision::INT8, "a.int8.onnx"});
    return cfg;
}
} // namespace

TEST_CASE("LatencyBudgetPolicy defers when there is no deadline") {
    LatencyBudgetPolicy policy(50.0);
    auto cfg = nodeWithVariants(true);
    RoutingContext ctx;
    ctx.node = &cfg;
    ctx.time_budget_remaining_ms = -1.0;
    CHECK(!policy.decide(ctx).has_value());
}

TEST_CASE("LatencyBudgetPolicy defers when the node has no INT8 variant") {
    LatencyBudgetPolicy policy(50.0);
    auto cfg = nodeWithVariants(false);
    RoutingContext ctx;
    ctx.node = &cfg;
    ctx.time_budget_remaining_ms = 1.0;
    CHECK(!policy.decide(ctx).has_value());
}

// Enum-class comparisons below are cast to int before CHECK(): doctest's
// operand-stringification SFINAE doesn't reliably pick up an ADL-found
// operator<< for a scoped enum on MSVC, which otherwise fails to compile
// (not a Loomcore bug — see doctest issues re: has_insertion_operator +
// MSVC two-phase lookup). Comparing as int sidesteps it without losing
// anything the test actually cares about.
TEST_CASE("LatencyBudgetPolicy downgrades to INT8 under a tight budget") {
    LatencyBudgetPolicy policy(50.0);
    auto cfg = nodeWithVariants(true);
    RoutingContext ctx;
    ctx.node = &cfg;
    ctx.time_budget_remaining_ms = 5.0;
    auto decision = policy.decide(ctx);
    REQUIRE(decision.has_value());
    REQUIRE(decision->precision.has_value());
    CHECK(static_cast<int>(*decision->precision) == static_cast<int>(Precision::INT8));
    CHECK(!decision->skip);
}

TEST_CASE("LatencyBudgetPolicy leaves precision alone above the threshold") {
    LatencyBudgetPolicy policy(50.0);
    auto cfg = nodeWithVariants(true);
    RoutingContext ctx;
    ctx.node = &cfg;
    ctx.time_budget_remaining_ms = 500.0;
    CHECK(!policy.decide(ctx).has_value());
}

TEST_CASE("LoadAwareBackendPolicy routes to the shallower queue") {
    LoadAwareBackendPolicy policy;
    RoutingContext ctx;
    ctx.cpu_queue_depth = 1;
    ctx.gpu_queue_depth = 5;
    auto d1 = policy.decide(ctx);
    REQUIRE(d1.has_value());
    REQUIRE(d1->backend.has_value());
    CHECK(static_cast<int>(*d1->backend) == static_cast<int>(Backend::CPU));

    ctx.cpu_queue_depth = 5;
    ctx.gpu_queue_depth = 1;
    auto d2 = policy.decide(ctx);
    REQUIRE(d2.has_value());
    REQUIRE(d2->backend.has_value());
    CHECK(static_cast<int>(*d2->backend) == static_cast<int>(Backend::GPU_SIM));

    ctx.cpu_queue_depth = 3;
    ctx.gpu_queue_depth = 3;
    CHECK(!policy.decide(ctx).has_value());
}

TEST_CASE("ConfidenceGatePolicy skips once confidence clears the threshold") {
    ConfidenceGatePolicy policy(0.8f);
    NodeConfig cfg;
    cfg.id = "text_embed";
    cfg.variants.push_back(VariantConfig{Precision::FP32, "a.onnx"});
    cfg.confidence_source_node = "classifier";
    cfg.confidence = [](const std::vector<NamedTensor>& t) -> std::optional<float> {
        if (t.empty()) return std::nullopt;
        return argmax(t.front()).second;
    };

    auto highConf = NamedTensor::makeFloat("logits", {1, 3}, {0.1f, 0.85f, 0.05f});
    std::vector<NamedTensor> highConfVec{highConf};
    RoutingContext ctx;
    ctx.node = &cfg;
    ctx.upstream_confidence_source = &highConfVec;
    auto decision = policy.decide(ctx);
    REQUIRE(decision.has_value());
    CHECK(decision->skip);

    auto lowConf = NamedTensor::makeFloat("logits", {1, 3}, {0.4f, 0.35f, 0.25f});
    std::vector<NamedTensor> lowConfVec{lowConf};
    ctx.upstream_confidence_source = &lowConfVec;
    CHECK(!policy.decide(ctx).has_value());
}

TEST_CASE("CompositeRouter merges independent fields and short-circuits on skip") {
    NodeConfig cfg = nodeWithVariants(true);
    cfg.confidence_source_node = "up";
    cfg.confidence = [](const std::vector<NamedTensor>&) -> std::optional<float> { return 0.99f; };

    SUBCASE("merges precision from one policy and backend from another") {
        CompositeRouter router;
        router.add(std::make_shared<LatencyBudgetPolicy>(50.0));
        router.add(std::make_shared<LoadAwareBackendPolicy>());

        RoutingContext ctx;
        ctx.node = &cfg;
        ctx.time_budget_remaining_ms = 1.0; // triggers LatencyBudgetPolicy -> INT8
        ctx.cpu_queue_depth = 0;
        ctx.gpu_queue_depth = 4; // triggers LoadAwareBackendPolicy -> CPU

        auto decision = router.decide(ctx);
        REQUIRE(decision.has_value());
        REQUIRE(decision->precision.has_value());
        CHECK(static_cast<int>(*decision->precision) == static_cast<int>(Precision::INT8));
        REQUIRE(decision->backend.has_value());
        CHECK(static_cast<int>(*decision->backend) == static_cast<int>(Backend::CPU));
    }

    SUBCASE("a skip decision short-circuits later policies") {
        CompositeRouter router;
        router.add(std::make_shared<ConfidenceGatePolicy>(0.5f));
        router.add(std::make_shared<LoadAwareBackendPolicy>());

        std::vector<NamedTensor> src{NamedTensor::makeFloat("x", {1, 1}, {1.0f})};
        RoutingContext ctx;
        ctx.node = &cfg;
        ctx.upstream_confidence_source = &src;
        ctx.cpu_queue_depth = 0;
        ctx.gpu_queue_depth = 9;

        auto decision = router.decide(ctx);
        REQUIRE(decision.has_value());
        CHECK(decision->skip);
        // The skip came from ConfidenceGatePolicy, not the (also-applicable)
        // LoadAwareBackendPolicy, and no backend override should be set.
        CHECK(!decision->backend.has_value());
    }
}

// ---------------------------------------------------------------------------
// PlannedPrecisionPolicy
// ---------------------------------------------------------------------------

TEST_CASE("PlannedPrecisionPolicy: no opinion without a usable plan, downgrades when the plan says so") {
    NodeConfig cfg;
    cfg.id = "n";
    RoutingContext ctx;
    ctx.node = &cfg;

    CHECK(!PlannedPrecisionPolicy{}.decide(ctx).has_value()); // no plan pointer at all

    PrecisionPlan cold_plan; // has_estimate defaults to false: cold-start, nothing to plan from
    ctx.precision_plan = &cold_plan;
    CHECK(!PlannedPrecisionPolicy{}.decide(ctx).has_value());

    PrecisionPlan plan;
    plan.has_estimate = true;
    plan.downgrade_to_int8["n"] = Precision::INT8;
    ctx.precision_plan = &plan;
    auto d = PlannedPrecisionPolicy{}.decide(ctx);
    REQUIRE(d.has_value());
    REQUIRE(d->precision.has_value());
    CHECK(static_cast<int>(*d->precision) == static_cast<int>(Precision::INT8));

    // A node the plan didn't mention keeps the scheduler's default.
    NodeConfig other;
    other.id = "untouched";
    ctx.node = &other;
    CHECK(!PlannedPrecisionPolicy{}.decide(ctx).has_value());
}

// ---------------------------------------------------------------------------
// CircuitBreakerPolicy
// ---------------------------------------------------------------------------

TEST_CASE("CircuitBreakerPolicy: healthy node gets no opinion; enough failures trips it open") {
    MetricsRegistry metrics;
    NodeConfig cfg;
    cfg.id = "n";
    CircuitBreakerPolicy breaker(/*error_rate_threshold=*/0.5, /*min_samples=*/4, /*cooldown_ms=*/50.0);
    RoutingContext ctx;
    ctx.node = &cfg;
    ctx.metrics = &metrics;

    CHECK(!breaker.decide(ctx).has_value());

    metrics.recordOutcome("n", true);
    metrics.recordOutcome("n", true);
    metrics.recordOutcome("n", false);
    metrics.recordOutcome("n", false);
    metrics.recordOutcome("n", false); // 3/5 failed >= 0.5, 5 >= min_samples

    auto d1 = breaker.decide(ctx);
    REQUIRE(d1.has_value());
    CHECK(d1->skip);

    auto d2 = breaker.decide(ctx); // still cooling down
    REQUIRE(d2.has_value());
    CHECK(d2->skip);
}

TEST_CASE("CircuitBreakerPolicy: exactly one caller is let through as the probe after cooldown") {
    // This is the specific bug a naive "bool tripped" implementation has:
    // the request that flips the breaker back to "not tripped" to admit
    // itself as the probe makes every *other* concurrent request look
    // healthy too, letting all of them through instead of just the one
    // probe. See router.h's NodeBreakerState comment for the fix (a
    // Closed/Open/HalfOpen tri-state machine).
    MetricsRegistry metrics;
    NodeConfig cfg;
    cfg.id = "n";
    CircuitBreakerPolicy breaker(0.5, 2, /*cooldown_ms=*/30.0);
    RoutingContext ctx;
    ctx.node = &cfg;
    ctx.metrics = &metrics;

    metrics.recordOutcome("n", false);
    metrics.recordOutcome("n", false);
    REQUIRE(breaker.decide(ctx)->skip); // trips

    std::this_thread::sleep_for(std::chrono::milliseconds(40)); // let cooldown elapse

    std::atomic<int> let_through{0};
    std::atomic<int> shed{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 16; ++i) {
        threads.emplace_back([&] {
            auto d = breaker.decide(ctx);
            if (!d.has_value())
                ++let_through;
            else
                ++shed;
        });
    }
    for (auto& t : threads) t.join();

    CHECK(let_through.load() == 1);
    CHECK(shed.load() == 15);
}

TEST_CASE("CircuitBreakerPolicy: a successful probe closes the breaker again") {
    MetricsRegistry metrics;
    NodeConfig cfg;
    cfg.id = "n";
    CircuitBreakerPolicy breaker(0.5, 2, /*cooldown_ms=*/20.0);
    RoutingContext ctx;
    ctx.node = &cfg;
    ctx.metrics = &metrics;

    metrics.recordOutcome("n", false);
    metrics.recordOutcome("n", false);
    REQUIRE(breaker.decide(ctx)->skip);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    REQUIRE(!breaker.decide(ctx).has_value()); // this call is the probe

    metrics.recordOutcome("n", true); // scheduler records the probe's real outcome

    CHECK(!breaker.decide(ctx).has_value()); // closed again: healthy
}

TEST_CASE("CircuitBreakerPolicy: a failed probe re-opens and restarts the cooldown") {
    MetricsRegistry metrics;
    NodeConfig cfg;
    cfg.id = "n";
    CircuitBreakerPolicy breaker(0.5, 2, /*cooldown_ms=*/20.0);
    RoutingContext ctx;
    ctx.node = &cfg;
    ctx.metrics = &metrics;

    metrics.recordOutcome("n", false);
    metrics.recordOutcome("n", false);
    REQUIRE(breaker.decide(ctx)->skip);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    REQUIRE(!breaker.decide(ctx).has_value()); // probe let through
    metrics.recordOutcome("n", false);         // probe fails

    auto after = breaker.decide(ctx);
    REQUIRE(after.has_value());
    CHECK(after->skip); // re-opened; a fresh cooldown starts
}

// ---------------------------------------------------------------------------
// BulkheadPolicy
// ---------------------------------------------------------------------------

TEST_CASE("BulkheadPolicy: sheds once the concurrency cap is reached, recovers once it drops") {
    MetricsRegistry metrics;
    NodeConfig cfg;
    cfg.id = "n";
    BulkheadPolicy bulkhead(/*max_concurrent=*/2);
    RoutingContext ctx;
    ctx.node = &cfg;
    ctx.metrics = &metrics;

    CHECK(!bulkhead.decide(ctx).has_value());
    metrics.incrementInFlight("n");
    CHECK(!bulkhead.decide(ctx).has_value());
    metrics.incrementInFlight("n");

    auto d = bulkhead.decide(ctx);
    REQUIRE(d.has_value());
    CHECK(d->skip);

    metrics.decrementInFlight("n");
    CHECK(!bulkhead.decide(ctx).has_value());
}
