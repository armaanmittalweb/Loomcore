// Loomcore — the agentic routing layer.
//
// "Agentic" here means: a small, inspectable decision layer that looks at
// *runtime* state (recent latency, queue depth, time budget, upstream
// confidence) rather than a fixed static plan, and picks which concrete
// model variant / backend / branch to invoke next — echoing the
// policy-router pattern (a chain of small, named, independently testable
// policies, each free to decline and defer to the next) used for
// model/path selection in the author's PRISM-Home project. Every decision
// carries a human-readable `reason` and is logged, so routing is auditable
// rather than a black box.
#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "loomcore/export.h"
#include "loomcore/metrics.h"
#include "loomcore/model_node.h"
#include "loomcore/planner.h"
#include "loomcore/types.h"

namespace loomcore {

// Everything a policy is allowed to base a decision on. Deliberately a
// flat, read-only snapshot rather than a handle into scheduler internals,
// so policies stay pure and unit-testable.
struct RoutingContext {
    std::string job_id;
    const NodeConfig* node = nullptr;
    const MetricsRegistry* metrics = nullptr;

    size_t cpu_queue_depth = 0;
    size_t gpu_queue_depth = 0;

    // < 0 means "no deadline" (the common case outside latency-budget demos).
    double time_budget_remaining_ms = -1.0;

    // Outputs already produced upstream in this job, if the node has a
    // ConfidenceExtractor registered (see NodeConfig::confidence).
    const std::vector<NamedTensor>* upstream_confidence_source = nullptr;

    // Set by the scheduler once per job (only when a time budget was given
    // and enough metrics history exists — see PrecisionPlan::has_estimate)
    // to the DAG-wide critical-path knapsack plan from loomcore/planner.h.
    // PlannedPrecisionPolicy is the only built-in policy that reads this;
    // it is exposed on the context (rather than kept scheduler-private) so
    // a custom policy can also read or override it.
    const PrecisionPlan* precision_plan = nullptr;
};

struct RoutingDecision {
    std::optional<Precision> precision; // unset = keep the scheduler's default
    std::optional<Backend> backend;     // unset = keep the node's configured backend
    bool skip = false;                  // true = do not execute this node at all
    std::string policy_name;
    std::string reason;
};

// One named, independently testable routing rule. `decide` returns
// std::nullopt to mean "no opinion" so a CompositeRouter can defer to the
// next policy in the chain.
class LOOMCORE_API IRoutingPolicy {
public:
    virtual ~IRoutingPolicy() = default;
    virtual std::string name() const = 0;
    virtual std::optional<RoutingDecision> decide(const RoutingContext& ctx) const = 0;
};

// Picks INT8 over FP32 for a node once the job's remaining time budget
// drops below `threshold_ms`, provided the node actually registers an
// INT8 variant. This is the "quantized path" milestone wired into runtime
// decision-making rather than left as a standalone benchmark curiosity.
class LOOMCORE_API LatencyBudgetPolicy : public IRoutingPolicy {
public:
    explicit LatencyBudgetPolicy(double threshold_ms) : threshold_ms_(threshold_ms) {}
    std::string name() const override { return "LatencyBudgetPolicy"; }
    std::optional<RoutingDecision> decide(const RoutingContext& ctx) const override;

private:
    double threshold_ms_;
};

// Sends a node to whichever simulated backend lane currently has the
// shallower queue, when the node hasn't pinned itself to one backend via
// config. A simple, honest load-balancing rule over the runtime state
// the scheduler already exposes through MetricsRegistry.
class LOOMCORE_API LoadAwareBackendPolicy : public IRoutingPolicy {
public:
    std::string name() const override { return "LoadAwareBackendPolicy"; }
    std::optional<RoutingDecision> decide(const RoutingContext& ctx) const override;
};

// Skips a node when an upstream confidence signal (extracted via the
// node's NodeConfig::confidence callback applied to *its own* declared
// source node's outputs — wired up by Runtime, see runtime.cpp) is
// already above `skip_above`. Used by the reference pipeline to skip the
// text-embedding node when the image classifier is already confident,
// which is the clearest demonstration of "which model to invoke" being an
// honest runtime decision instead of a fixed static graph traversal.
class LOOMCORE_API ConfidenceGatePolicy : public IRoutingPolicy {
public:
    explicit ConfidenceGatePolicy(float skip_above) : skip_above_(skip_above) {}
    std::string name() const override { return "ConfidenceGatePolicy"; }
    std::optional<RoutingDecision> decide(const RoutingContext& ctx) const override;

private:
    float skip_above_;
};

// Looks up ctx.precision_plan (see RoutingContext above) for this node and,
// if the plan chose to downgrade it, returns that decision. Unlike
// LatencyBudgetPolicy's fixed per-call threshold, the plan behind this was
// computed once, DAG-wide, by solving a knapsack over which critical-path
// nodes to downgrade — see loomcore/planner.h for the full rationale. Put
// this *before* LatencyBudgetPolicy in a CompositeRouter chain so the
// informed, whole-graph decision wins when a plan exists, falling through
// to LatencyBudgetPolicy's simpler local rule when it doesn't (e.g. cold
// start, before enough per-precision latency samples exist).
class LOOMCORE_API PlannedPrecisionPolicy : public IRoutingPolicy {
public:
    std::string name() const override { return "PlannedPrecisionPolicy"; }
    std::optional<RoutingDecision> decide(const RoutingContext& ctx) const override;
};

// Trips to fail-fast once a node's rolling error rate
// (MetricsRegistry::outcomeStats) crosses `error_rate_threshold` over at
// least `min_samples` observations, protecting the rest of the DAG (and
// the caller) from continuing to dispatch work to a node that is
// consistently failing. After `cooldown_ms`, the breaker "half-opens":
// it lets exactly one probe request through (to test recovery) while
// continuing to fail-fast everything else, and fully closes again once
// that probe's outcome is recorded as a success.
//
// A tripped or half-open-but-not-yet-probed decision is expressed as
// `skip = true` — the same semantics ConfidenceGatePolicy already uses
// for "do not execute this node", so no scheduler change was needed to
// add this policy; it only needed the outcome tracking in
// MetricsRegistry::recordOutcome, which the scheduler now calls from
// every node's completion/failure callback.
class LOOMCORE_API CircuitBreakerPolicy : public IRoutingPolicy {
public:
    CircuitBreakerPolicy(double error_rate_threshold, size_t min_samples, double cooldown_ms)
        : error_rate_threshold_(error_rate_threshold), min_samples_(min_samples), cooldown_ms_(cooldown_ms) {}
    std::string name() const override { return "CircuitBreakerPolicy"; }
    std::optional<RoutingDecision> decide(const RoutingContext& ctx) const override;

private:
    double error_rate_threshold_;
    size_t min_samples_;
    double cooldown_ms_;
    // Trip state is per-node and shared across calls/threads, so it lives
    // here rather than in the (const, per-call) RoutingContext. A tri-state
    // machine, not a bool: Closed (healthy) -> Open (tripped, cooling down)
    // -> HalfOpen (cooldown elapsed, exactly one probe request let through)
    // -> Closed (probe succeeded) or -> Open (probe failed, cooldown
    // restarts). See router.cpp for why this needs all three states rather
    // than collapsing HalfOpen into "tripped == false": with only a bool,
    // the very request that flips it back to "healthy" to admit the probe
    // makes every *other* concurrent request look healthy too, letting
    // them all through instead of just the one probe.
    enum class BreakerState { Closed, Open, HalfOpen };
    struct NodeBreakerState {
        std::atomic<BreakerState> state{BreakerState::Closed};
        std::atomic<long long> tripped_at_steady_ns{0};
    };
    mutable std::mutex state_mutex_;
    mutable std::map<std::string, std::unique_ptr<NodeBreakerState>> state_;
    NodeBreakerState& stateFor(const std::string& node_id) const;
};

// Caps how many requests for one node may be in flight at once
// (MetricsRegistry::inFlight, incremented at dispatch / decremented at
// completion by the scheduler). Once `max_concurrent` is reached, further
// requests for that node are shed via `skip = true` rather than queuing
// unboundedly — a bulkhead protects the rest of the DAG (and other jobs
// sharing the node's lane) from one overloaded node backing up everything
// behind it. This is a load-shedding decision, the same skip semantics as
// ConfidenceGatePolicy and CircuitBreakerPolicy above, not a "wait" —
// Loomcore's RoutingDecision has no "defer/retry" outcome, and honestly
// shedding is preferable to a hidden unbounded queue anyway.
class LOOMCORE_API BulkheadPolicy : public IRoutingPolicy {
public:
    explicit BulkheadPolicy(size_t max_concurrent) : max_concurrent_(max_concurrent) {}
    std::string name() const override { return "BulkheadPolicy"; }
    std::optional<RoutingDecision> decide(const RoutingContext& ctx) const override;

private:
    size_t max_concurrent_;
};

// Chain-of-responsibility over a list of policies: the first one to return
// an opinion wins for each field it sets; unset fields fall through to
// later policies, and any field still unset after the whole chain keeps
// the scheduler's compile-time defaults (the node's configured backend,
// and Precision::FP32).
class LOOMCORE_API CompositeRouter : public IRoutingPolicy {
public:
    void add(std::shared_ptr<IRoutingPolicy> policy);
    std::string name() const override { return "CompositeRouter"; }
    std::optional<RoutingDecision> decide(const RoutingContext& ctx) const override;

private:
    std::vector<std::shared_ptr<IRoutingPolicy>> policies_;
};

} // namespace loomcore
