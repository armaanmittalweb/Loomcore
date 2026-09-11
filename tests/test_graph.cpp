#include <doctest/doctest.h>

#include <algorithm>

#include "loomcore/graph.h"

using namespace loomcore;

namespace {
NodeConfig simpleNode(const std::string& id, std::vector<std::string> deps = {}) {
    NodeConfig cfg;
    cfg.id = id;
    cfg.depends_on = std::move(deps);
    cfg.variants.push_back(VariantConfig{Precision::FP32, "unused.onnx"});
    cfg.binder = [](const NodeExecutionContext&) { return std::vector<NamedTensor>{}; };
    return cfg;
}
} // namespace

TEST_CASE("topoOrder respects dependencies") {
    Graph g;
    g.addNode(simpleNode("a"));
    g.addNode(simpleNode("b", {"a"}));
    g.addNode(simpleNode("c", {"a", "b"}));

    auto order = g.topoOrder();
    REQUIRE(order.size() == 3);
    auto pos = [&](const std::string& id) {
        return std::distance(order.begin(), std::find(order.begin(), order.end(), id));
    };
    CHECK(pos("a") < pos("b"));
    CHECK(pos("b") < pos("c"));
}

TEST_CASE("Graph::validate throws GraphError on an unknown dependency") {
    Graph g;
    g.addNode(simpleNode("b", {"missing"}));
    CHECK_THROWS_AS(g.validate(), GraphError);
}

TEST_CASE("Graph::validate throws GraphError on a cycle") {
    Graph g;
    g.addNode(simpleNode("a", {"b"}));
    g.addNode(simpleNode("b", {"a"}));
    CHECK_THROWS_AS(g.validate(), GraphError);
}

TEST_CASE("addNode rejects duplicate ids") {
    Graph g;
    g.addNode(simpleNode("a"));
    CHECK_THROWS_AS(g.addNode(simpleNode("a")), GraphError);
}

TEST_CASE("dependents and sinkNodes report the right topology") {
    Graph g;
    g.addNode(simpleNode("a"));
    g.addNode(simpleNode("b", {"a"}));
    g.addNode(simpleNode("c", {"a"}));

    auto deps_of_a = g.dependents("a");
    CHECK(deps_of_a.size() == 2);

    auto sinks = g.sinkNodes();
    REQUIRE(sinks.size() == 2);
    CHECK(std::find(sinks.begin(), sinks.end(), "b") != sinks.end());
    CHECK(std::find(sinks.begin(), sinks.end(), "c") != sinks.end());
}
