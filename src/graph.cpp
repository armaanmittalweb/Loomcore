#include "loomcore/graph.h"

#include <deque>
#include <set>

namespace loomcore {

void Graph::addNode(NodeConfig cfg) {
    if (cfg.id.empty()) {
        throw GraphError("Graph::addNode: node id must not be empty");
    }
    if (index_by_id_.count(cfg.id)) {
        throw GraphError("Graph::addNode: duplicate node id '" + cfg.id + "'");
    }
    if (cfg.variants.empty()) {
        throw GraphError("Graph::addNode: node '" + cfg.id + "' declares no model variants");
    }
    index_by_id_[cfg.id] = nodes_.size();
    nodes_.push_back(std::move(cfg));
}

bool Graph::hasNode(const std::string& id) const { return index_by_id_.count(id) != 0; }

const NodeConfig& Graph::node(const std::string& id) const {
    auto it = index_by_id_.find(id);
    if (it == index_by_id_.end()) {
        throw GraphError("Graph::node: no such node '" + id + "'");
    }
    return nodes_[it->second];
}

void Graph::validate() const {
    for (const auto& n : nodes_) {
        for (const auto& dep : n.depends_on) {
            if (!hasNode(dep)) {
                throw GraphError("Graph::validate: node '" + n.id + "' depends on unknown node '" + dep + "'");
            }
            if (dep == n.id) {
                throw GraphError("Graph::validate: node '" + n.id + "' depends on itself");
            }
        }
    }
    // Acyclicity is verified as a side effect of topoOrder() succeeding;
    // re-run it here (cheap at demo scale) so validate() alone is a
    // complete health check.
    (void)topoOrder();
}

std::vector<std::string> Graph::topoOrder() const {
    std::map<std::string, int> indegree;
    std::map<std::string, std::vector<std::string>> adj; // dep -> dependents
    for (const auto& n : nodes_) indegree[n.id] = 0;
    for (const auto& n : nodes_) {
        for (const auto& dep : n.depends_on) {
            if (!hasNode(dep)) {
                throw GraphError("Graph::topoOrder: node '" + n.id + "' depends on unknown node '" + dep + "'");
            }
            adj[dep].push_back(n.id);
            indegree[n.id]++;
        }
    }

    std::deque<std::string> ready;
    // Insertion order first, for determinism.
    for (const auto& n : nodes_) {
        if (indegree[n.id] == 0) ready.push_back(n.id);
    }

    std::vector<std::string> order;
    order.reserve(nodes_.size());
    while (!ready.empty()) {
        std::string id = ready.front();
        ready.pop_front();
        order.push_back(id);
        for (const auto& dependent : adj[id]) {
            if (--indegree[dependent] == 0) ready.push_back(dependent);
        }
    }

    if (order.size() != nodes_.size()) {
        throw GraphError("Graph::topoOrder: cycle detected among the declared node dependencies");
    }
    return order;
}

std::vector<std::string> Graph::dependents(const std::string& id) const {
    if (!hasNode(id)) {
        throw GraphError("Graph::dependents: no such node '" + id + "'");
    }
    std::vector<std::string> result;
    for (const auto& n : nodes_) {
        for (const auto& dep : n.depends_on) {
            if (dep == id) {
                result.push_back(n.id);
                break;
            }
        }
    }
    return result;
}

std::vector<std::string> Graph::sinkNodes() const {
    std::set<std::string> has_dependent;
    for (const auto& n : nodes_) {
        for (const auto& dep : n.depends_on) has_dependent.insert(dep);
    }
    std::vector<std::string> sinks;
    for (const auto& n : nodes_) {
        if (!has_dependent.count(n.id)) sinks.push_back(n.id);
    }
    return sinks;
}

} // namespace loomcore
