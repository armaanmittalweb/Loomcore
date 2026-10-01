// Loomcore — deadline-aware precision planning.
//
// LatencyBudgetPolicy (router.h) makes a *local*, per-node, threshold-based
// call: "is the remaining budget below X ms? if so, and this node has an
// INT8 variant, use it." That is honest and cheap, but it is blind to the
// rest of the DAG: it has no notion of which nodes actually sit on the
// critical path, and no way to prefer downgrading a cheap-to-degrade node
// over an expensive-to-degrade one when both would satisfy the deadline.
//
// PrecisionPlanner solves the real version of that problem once, at job
// submission time (not per-node-dispatch — this is O(nodes) work over
// already-measured stats, done once, not on the scheduler's hot path):
//
//   1. Find the DAG's critical path (the longest FP32-cost path from a
//      root to a sink) using each node's measured p95 latency.
//   2. If the critical path's total cost already fits the job's time
//      budget, no downgrade is needed.
//   3. Otherwise, compute how much latency must be shaved off
//      (`slack_needed_ms`) and choose which critical-path nodes to run at
//      INT8 instead of FP32 to close that gap while disturbing the fewest
//      "quality-weighted" nodes possible — a 0/1 knapsack: minimize total
//      NodeConfig::quality_weight spent, subject to the selected nodes'
//      FP32-minus-INT8 latency savings summing to at least
//      `slack_needed_ms`. This is a genuine DP, not a greedy shortcut:
//      greedy-by-largest-saving is provably optimal only when every node
//      has the same quality_weight; the moment weights differ (a node the
//      caller has flagged as more precision-sensitive via a higher
//      quality_weight), greedy can pick a strictly worse set than the DP
//      — see tests/test_planner.cpp for a concrete case where they diverge.
//
// The resulting plan is a plain data structure (RoutingContext::plan below)
// that PlannedPrecisionPolicy just looks up per node — the DP itself never
// runs on a lane worker thread or inside a routing decision.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "loomcore/export.h"
#include "loomcore/graph.h"
#include "loomcore/metrics.h"
#include "loomcore/model_node.h"
#include "loomcore/types.h"

namespace loomcore {

struct LOOMCORE_API PrecisionPlan {
    // Only nodes the planner actually chose to downgrade appear here
    // (value is always Precision::INT8). A node absent from this map keeps
    // whatever the scheduler's own default / other policies decide —
    // the planner only ever expresses "downgrade", never "upgrade", since
    // FP32 is always the safe default.
    std::map<std::string, Precision> downgrade_to_int8;

    std::vector<std::string> critical_path;      // root -> sink, in order
    double critical_path_fp32_ms = 0.0;          // before any downgrade
    double critical_path_planned_ms = 0.0;       // after applying downgrade_to_int8
    double budget_ms = -1.0;
    double quality_weight_spent = 0.0;
    // false when even downgrading every eligible critical-path node isn't
    // enough to fit the budget (the plan still applies every downgrade it
    // found — best-effort degradation rather than giving up).
    bool feasible = true;
    // true when the critical path could not be costed at all (some node on
    // it has no measured latency yet — a cold-start system has nothing to
    // plan from). LatencyBudgetPolicy remains the right fallback then.
    bool has_estimate = false;
};

// Builds a plan for one job's remaining graph run. `metrics` is read via
// the precision-qualified keys the scheduler records
// (see docs/ARCHITECTURE.md "Metrics": "<node_id>#FP32" / "<node_id>#INT8");
// a node with no samples at a given precision is treated as un-costed.
// `dt_ms` is the DP's bucket resolution (smaller = more precise, more
// buckets; 0.05ms is plenty fine for millisecond-scale model latencies).
LOOMCORE_API PrecisionPlan planPrecisionForBudget(const Graph& graph, const std::map<std::string, ModelNode>& nodes,
                                                   const MetricsRegistry& metrics, double budget_ms,
                                                   double dt_ms = 0.05);

// The metrics key scheduler.cpp records per-(node,precision) latency
// under, alongside the plain per-node key `nodeStats()` reads. Exposed so
// the planner and its tests share one definition instead of restating the
// separator.
LOOMCORE_API std::string precisionMetricsKey(const std::string& node_id, Precision p);

} // namespace loomcore
