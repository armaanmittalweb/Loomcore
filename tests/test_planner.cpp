// Unit tests for PrecisionPlanner (loomcore/planner.h): pure-function tests
// against a Graph + MetricsRegistry, no ONNX Runtime session ever loaded
// (the planner only reads NodeConfig::{variants,quality_weight} and
// measured latency; it never touches a ModelNode's actual loaded
// ModelVariant), so these run in milliseconds and need no fixture models.
#include <doctest/doctest.h>

#include "loomcore/planner.h"

using namespace loomcore;

namespace {

NodeConfig makeNode(std::string id, std::vector<std::string> depends_on, double quality_weight = 1.0) {
    NodeConfig cfg;
    cfg.id = std::move(id);
    cfg.depends_on = std::move(depends_on);
    cfg.quality_weight = quality_weight;
    cfg.variants.push_back(VariantConfig{Precision::FP32, cfg.id + ".fp32.onnx"});
    cfg.variants.push_back(VariantConfig{Precision::INT8, cfg.id + ".int8.onnx"});
    cfg.binder = [](const NodeExecutionContext&) { return std::vector<NamedTensor>{}; };
    return cfg;
}

struct Fixture {
    Graph graph;
    std::map<std::string, ModelNode> nodes;

    void addNode(const NodeConfig& cfg) {
        graph.addNode(cfg);
        ModelNode mnode;
        mnode.config = cfg;
        nodes[cfg.id] = std::move(mnode);
    }

    void seed(const std::string& id, Precision p, double latency_ms) {
        // stats() returns p95 from the recorded window; recording the same
        // value 5 times makes p95 == that exact value regardless of the
        // percentile interpolation details, so tests can assert exact costs.
        for (int i = 0; i < 5; ++i) metrics.recordLatency(precisionMetricsKey(id, p), latency_ms);
    }

    MetricsRegistry metrics;
};

} // namespace

TEST_CASE("PrecisionPlanner: fits already, no downgrade needed") {
    Fixture f;
    f.addNode(makeNode("a", {}));
    f.addNode(makeNode("b", {"a"}));
    f.seed("a", Precision::FP32, 10.0);
    f.seed("a", Precision::INT8, 6.0);
    f.seed("b", Precision::FP32, 10.0);
    f.seed("b", Precision::INT8, 6.0);

    auto plan = planPrecisionForBudget(f.graph, f.nodes, f.metrics, /*budget_ms=*/50.0);
    REQUIRE(plan.has_estimate);
    CHECK(plan.critical_path_fp32_ms == doctest::Approx(20.0));
    CHECK(plan.downgrade_to_int8.empty());
    CHECK(plan.feasible);
}

TEST_CASE("PrecisionPlanner: cold registry reports no estimate rather than guessing") {
    Fixture f;
    f.addNode(makeNode("a", {}));
    // No metrics recorded at all.
    auto plan = planPrecisionForBudget(f.graph, f.nodes, f.metrics, /*budget_ms=*/1.0);
    CHECK_FALSE(plan.has_estimate);
    CHECK(plan.downgrade_to_int8.empty());
}

TEST_CASE("PrecisionPlanner: minimal-weight knapsack beats greedy-by-largest-saving") {
    // A linear critical path A -> B -> C. A's FP32->INT8 saving is the
    // single largest (10ms) but comes at the highest quality_weight (5);
    // B and C together save slightly more (11ms) for far less weight (2).
    // A saving-only greedy (sort by saving, take the fewest items that
    // cover the gap) would stop at {A} alone — 10ms >= the 10ms needed —
    // and never even look at B or C, since one item already "solves" it.
    // Minimizing total quality_weight instead of item count requires
    // actually comparing {A} (weight 5) against {B, C} (weight 2), which
    // is exactly the 0/1 knapsack PrecisionPlanner solves.
    Fixture f;
    f.addNode(makeNode("a", {}, /*quality_weight=*/5.0));
    f.addNode(makeNode("b", {"a"}, /*quality_weight=*/1.0));
    f.addNode(makeNode("c", {"b"}, /*quality_weight=*/1.0));
    f.seed("a", Precision::FP32, 30.0);
    f.seed("a", Precision::INT8, 20.0); // save 10
    f.seed("b", Precision::FP32, 20.0);
    f.seed("b", Precision::INT8, 14.0); // save 6
    f.seed("c", Precision::FP32, 15.0);
    f.seed("c", Precision::INT8, 10.0); // save 5

    // Total FP32 = 65ms; budget 55ms => 10ms of slack must be found.
    auto plan = planPrecisionForBudget(f.graph, f.nodes, f.metrics, /*budget_ms=*/55.0);
    REQUIRE(plan.has_estimate);
    CHECK(plan.critical_path == std::vector<std::string>{"a", "b", "c"});
    CHECK(plan.critical_path_fp32_ms == doctest::Approx(65.0));

    CHECK(plan.downgrade_to_int8.count("a") == 0); // the DP must NOT pick the expensive single item
    CHECK(plan.downgrade_to_int8.count("b") == 1);
    CHECK(plan.downgrade_to_int8.count("c") == 1);
    CHECK(plan.quality_weight_spent == doctest::Approx(2.0));
    CHECK(plan.critical_path_planned_ms == doctest::Approx(54.0)); // 65 - (6+5)
    CHECK(plan.feasible);
}

TEST_CASE("PrecisionPlanner: reports infeasible (best-effort) when no achievable set fits") {
    Fixture f;
    f.addNode(makeNode("a", {}));
    f.seed("a", Precision::FP32, 100.0);
    f.seed("a", Precision::INT8, 90.0); // only 10ms available to save

    auto plan = planPrecisionForBudget(f.graph, f.nodes, f.metrics, /*budget_ms=*/10.0);
    REQUIRE(plan.has_estimate);
    CHECK_FALSE(plan.feasible);
    CHECK(plan.downgrade_to_int8.count("a") == 1); // best effort: take everything available anyway
    CHECK(plan.critical_path_planned_ms == doctest::Approx(90.0));
}

TEST_CASE("PrecisionPlanner: unbounded budget never downgrades") {
    Fixture f;
    f.addNode(makeNode("a", {}));
    f.seed("a", Precision::FP32, 1000.0);
    f.seed("a", Precision::INT8, 1.0);
    auto plan = planPrecisionForBudget(f.graph, f.nodes, f.metrics, /*budget_ms=*/-1.0);
    REQUIRE(plan.has_estimate);
    CHECK(plan.downgrade_to_int8.empty());
    CHECK(plan.feasible);
}
