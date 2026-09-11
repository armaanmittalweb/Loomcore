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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "loomcore/export.h"
#include "loomcore/metrics.h"
#include "loomcore/model_node.h"
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
