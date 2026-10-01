#include "loomcore/planner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace loomcore {

std::string precisionMetricsKey(const std::string& node_id, Precision p) {
    return node_id + "#" + toString(p);
}

namespace {

// Measured FP32-latency-weighted longest path through the DAG, expressed
// as `cost_to_sink[node]` = this node's own cost plus the most expensive
// remaining path through any of its dependents. Nodes with no measured
// FP32 samples yet make the whole estimate unusable (returns false) —
// see PrecisionPlan::has_estimate.
bool computeCostToSink(const Graph& graph, const MetricsRegistry& metrics,
                       std::unordered_map<std::string, double>& cost_to_sink) {
    auto topo = graph.topoOrder();
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        const std::string& id = *it;
        auto stats = metrics.stats(precisionMetricsKey(id, Precision::FP32));
        if (stats.count == 0) return false; // cold: no data to plan from
        double own_cost = stats.p95_ms;
        double best_downstream = 0.0;
        for (const auto& dependent : graph.dependents(id)) {
            auto dit = cost_to_sink.find(dependent);
            if (dit != cost_to_sink.end()) best_downstream = std::max(best_downstream, dit->second);
        }
        cost_to_sink[id] = own_cost + best_downstream;
    }
    return true;
}

std::vector<std::string> reconstructCriticalPath(const Graph& graph,
                                                   const std::unordered_map<std::string, double>& cost_to_sink) {
    std::vector<std::string> roots;
    for (const auto& n : graph.nodes()) {
        if (n.depends_on.empty()) roots.push_back(n.id);
    }
    if (roots.empty()) return {};

    std::string cur = roots.front();
    double best = cost_to_sink.at(cur);
    for (const auto& r : roots) {
        double c = cost_to_sink.at(r);
        if (c > best) {
            best = c;
            cur = r;
        }
    }

    std::vector<std::string> path;
    path.push_back(cur);
    while (true) {
        const auto& dependents = graph.dependents(cur);
        if (dependents.empty()) break;
        std::string best_dep;
        double best_cost = -1.0;
        for (const auto& d : dependents) {
            auto it = cost_to_sink.find(d);
            double c = it == cost_to_sink.end() ? 0.0 : it->second;
            if (c > best_cost) {
                best_cost = c;
                best_dep = d;
            }
        }
        if (best_dep.empty()) break;
        path.push_back(best_dep);
        cur = best_dep;
    }
    return path;
}

} // namespace

PrecisionPlan planPrecisionForBudget(const Graph& graph, const std::map<std::string, ModelNode>& nodes,
                                      const MetricsRegistry& metrics, double budget_ms, double dt_ms) {
    PrecisionPlan plan;
    plan.budget_ms = budget_ms;

    std::unordered_map<std::string, double> cost_to_sink;
    plan.has_estimate = computeCostToSink(graph, metrics, cost_to_sink);
    if (!plan.has_estimate) return plan;

    plan.critical_path = reconstructCriticalPath(graph, cost_to_sink);
    if (plan.critical_path.empty()) return plan;

    plan.critical_path_fp32_ms = 0.0;
    for (const auto& id : plan.critical_path) {
        plan.critical_path_fp32_ms += metrics.stats(precisionMetricsKey(id, Precision::FP32)).p95_ms;
    }
    plan.critical_path_planned_ms = plan.critical_path_fp32_ms;

    if (budget_ms < 0.0 || plan.critical_path_fp32_ms <= budget_ms) {
        return plan; // fits already; no downgrade needed
    }

    double slack_needed_ms = plan.critical_path_fp32_ms - budget_ms;

    // --- Build the knapsack candidate list: critical-path nodes with a
    // measured, positive FP32->INT8 saving. ---
    struct Candidate {
        std::string node_id;
        double save_ms;
        double weight;
    };
    std::vector<Candidate> candidates;
    for (const auto& id : plan.critical_path) {
        auto it = nodes.find(id);
        if (it == nodes.end() || !it->second.config.hasVariant(Precision::INT8)) continue;
        auto fp32_stats = metrics.stats(precisionMetricsKey(id, Precision::FP32));
        auto int8_stats = metrics.stats(precisionMetricsKey(id, Precision::INT8));
        if (fp32_stats.count == 0 || int8_stats.count == 0) continue; // no INT8 measurement yet
        double save = fp32_stats.p95_ms - int8_stats.p95_ms;
        if (save <= 0.0) continue; // INT8 isn't actually faster for this node on this hardware
        candidates.push_back({id, save, it->second.config.quality_weight});
    }

    if (candidates.empty()) {
        plan.feasible = false; // nothing left to try; report the shortfall honestly
        return plan;
    }

    // --- 0/1 knapsack: minimize sum(weight) s.t. sum(save) >= slack_needed_ms. ---
    // dt_ms-wide buckets; bucket `target` means ">= slack_needed_ms covered".
    dt_ms = std::max(dt_ms, 1e-6);
    size_t target = static_cast<size_t>(std::ceil(slack_needed_ms / dt_ms));
    size_t n = candidates.size();
    const double kInf = std::numeric_limits<double>::infinity();

    // dp[i][j] = min weight using the first i candidates to reach >= j buckets of savings.
    std::vector<std::vector<double>> dp(n + 1, std::vector<double>(target + 1, kInf));
    for (size_t i = 0; i <= n; ++i) dp[i][0] = 0.0;

    for (size_t i = 1; i <= n; ++i) {
        const auto& c = candidates[i - 1];
        size_t save_buckets = std::min(target, static_cast<size_t>(std::ceil(c.save_ms / dt_ms)));
        for (size_t j = 0; j <= target; ++j) {
            double without = dp[i - 1][j];
            double with = kInf;
            if (dp[i - 1][j >= save_buckets ? j - save_buckets : 0] != kInf) {
                size_t prev_j = j >= save_buckets ? j - save_buckets : 0;
                // Taking this item from state `prev_j` always lands at
                // least at bucket `prev_j + save_buckets`, which for
                // prev_j==0 (item alone covers the rest) already reaches
                // `target` when save_buckets >= j; the clamp above and
                // the min() with target keep every transition valid.
                with = dp[i - 1][prev_j] + c.weight;
            }
            dp[i][j] = std::min(without, with);
        }
        // Every state reachable via "take item i" that would overshoot
        // past `target` collapses onto `target` itself (>= target is all
        // that matters), so also relax dp[i][target] using every smaller
        // prev_j once more directly (handles save_buckets aloneexceeding
        // `target` from prev_j==0, already covered above since
        // std::min(target, j) with j==target and prev_j=target-save_buckets
        // clamped to 0 is exactly that case).
    }

    double best_weight = dp[n][target];
    if (best_weight == kInf) {
        // Even every candidate together can't close the gap: take all of
        // them (best effort) and report infeasible.
        plan.feasible = false;
        double total_save = 0.0;
        for (const auto& c : candidates) {
            plan.downgrade_to_int8[c.node_id] = Precision::INT8;
            plan.quality_weight_spent += c.weight;
            total_save += c.save_ms;
        }
        plan.critical_path_planned_ms = plan.critical_path_fp32_ms - total_save;
        return plan;
    }

    // Backtrack to find which candidates were chosen: at each step, item i
    // was taken iff dp[i][j] was reached via the "with" transition from
    // dp[i-1][prev_j] rather than carried over unchanged from dp[i-1][j].
    // Both dp[i][j] and the "with" value here are the exact same
    // floating-point expression evaluated the same way in the forward
    // pass, so the equality test is exact, not an epsilon comparison.
    std::vector<bool> chosen(n, false);
    size_t j = target;
    for (size_t i = n; i >= 1; --i) {
        const auto& c = candidates[i - 1];
        size_t save_buckets = std::min(target, static_cast<size_t>(std::ceil(c.save_ms / dt_ms)));
        size_t prev_j = j >= save_buckets ? j - save_buckets : 0;
        if (dp[i - 1][prev_j] != kInf && dp[i - 1][prev_j] + c.weight == dp[i][j]) {
            chosen[i - 1] = true;
            j = prev_j;
        }
    }

    double total_save = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!chosen[i]) continue;
        plan.downgrade_to_int8[candidates[i].node_id] = Precision::INT8;
        plan.quality_weight_spent += candidates[i].weight;
        total_save += candidates[i].save_ms;
    }
    plan.critical_path_planned_ms = plan.critical_path_fp32_ms - total_save;
    plan.feasible = plan.critical_path_planned_ms <= budget_ms + 1e-9;
    return plan;
}

} // namespace loomcore
