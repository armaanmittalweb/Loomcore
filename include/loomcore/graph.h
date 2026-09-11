// Loomcore — DAG topology: nodes + declared dependencies.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "loomcore/export.h"
#include "loomcore/model_node.h"

namespace loomcore {

class LOOMCORE_API GraphError : public LoomcoreError {
public:
    explicit GraphError(const std::string& what) : LoomcoreError(what) {}
};

// An immutable-once-built dependency graph over NodeConfigs. Validates on
// construction: every dependency id must exist, node ids must be unique,
// and the graph must be acyclic.
class LOOMCORE_API Graph {
public:
    void addNode(NodeConfig cfg);

    // Checks referential integrity and acyclicity. Throws GraphError on
    // failure. Called automatically by topoOrder()/dependents() but
    // exposed so Runtime can fail fast at load time with a clear message.
    void validate() const;

    const NodeConfig& node(const std::string& id) const;
    bool hasNode(const std::string& id) const;
    const std::vector<NodeConfig>& nodes() const { return nodes_; }

    // Nodes in an order such that every node appears after all of its
    // dependencies (Kahn's algorithm). Deterministic for a fixed insertion
    // order and fixed dependency set.
    std::vector<std::string> topoOrder() const;

    // Nodes that directly declare `id` as a dependency.
    std::vector<std::string> dependents(const std::string& id) const;

    // Nodes with no dependents — the graph's "sinks". A graph run's final
    // result is the union of these nodes' outputs.
    std::vector<std::string> sinkNodes() const;

private:
    std::vector<NodeConfig> nodes_;
    std::map<std::string, size_t> index_by_id_;
};

} // namespace loomcore
