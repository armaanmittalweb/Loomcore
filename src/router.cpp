#include "loomcore/router.h"

#include <chrono>

namespace loomcore {

std::optional<RoutingDecision> LatencyBudgetPolicy::decide(const RoutingContext& ctx) const {
    if (ctx.time_budget_remaining_ms < 0.0) return std::nullopt; // no deadline, no opinion
    if (ctx.node == nullptr || !ctx.node->hasVariant(Precision::INT8)) return std::nullopt;
    if (ctx.time_budget_remaining_ms >= threshold_ms_) return std::nullopt;

    RoutingDecision d;
    d.policy_name = name();
    d.precision = Precision::INT8;
    d.reason = "remaining budget " + std::to_string(ctx.time_budget_remaining_ms) + "ms < threshold " +
               std::to_string(threshold_ms_) + "ms; downgrading to INT8 for node '" + ctx.node->id + "'";
    return d;
}

std::optional<RoutingDecision> LoadAwareBackendPolicy::decide(const RoutingContext& ctx) const {
    if (ctx.cpu_queue_depth == ctx.gpu_queue_depth) return std::nullopt; // no opinion, tie
    RoutingDecision d;
    d.policy_name = name();
    d.backend = ctx.cpu_queue_depth < ctx.gpu_queue_depth ? Backend::CPU : Backend::GPU_SIM;
    d.reason = "cpu_queue=" + std::to_string(ctx.cpu_queue_depth) + " gpu_sim_queue=" +
               std::to_string(ctx.gpu_queue_depth) + "; routing to the shallower lane";
    return d;
}

std::optional<RoutingDecision> ConfidenceGatePolicy::decide(const RoutingContext& ctx) const {
    if (ctx.node == nullptr || !ctx.node->confidence || ctx.upstream_confidence_source == nullptr) {
        return std::nullopt;
    }
    auto conf = ctx.node->confidence(*ctx.upstream_confidence_source);
    if (!conf.has_value()) return std::nullopt;
    if (*conf < skip_above_) return std::nullopt;

    RoutingDecision d;
    d.policy_name = name();
    d.skip = true;
    d.reason = "upstream confidence " + std::to_string(*conf) + " >= threshold " + std::to_string(skip_above_) +
               "; skipping node '" + ctx.node->id + "'";
    return d;
}

std::optional<RoutingDecision> PlannedPrecisionPolicy::decide(const RoutingContext& ctx) const {
    if (ctx.node == nullptr || ctx.precision_plan == nullptr || !ctx.precision_plan->has_estimate) {
        return std::nullopt;
    }
    auto it = ctx.precision_plan->downgrade_to_int8.find(ctx.node->id);
    if (it == ctx.precision_plan->downgrade_to_int8.end()) return std::nullopt; // plan wants FP32 (its default) here

    RoutingDecision d;
    d.policy_name = name();
    d.precision = Precision::INT8;
    d.reason = "DAG-wide precision plan (critical path " +
               std::to_string(ctx.precision_plan->critical_path_fp32_ms) + "ms -> " +
               std::to_string(ctx.precision_plan->critical_path_planned_ms) + "ms vs budget " +
               std::to_string(ctx.precision_plan->budget_ms) + "ms) downgrades node '" + ctx.node->id + "'";
    return d;
}

CircuitBreakerPolicy::NodeBreakerState& CircuitBreakerPolicy::stateFor(const std::string& node_id) const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto it = state_.find(node_id);
    if (it == state_.end()) {
        it = state_.emplace(node_id, std::make_unique<NodeBreakerState>()).first;
    }
    return *it->second;
}

std::optional<RoutingDecision> CircuitBreakerPolicy::decide(const RoutingContext& ctx) const {
    if (ctx.node == nullptr || ctx.metrics == nullptr) return std::nullopt;
    auto& node_state = stateFor(ctx.node->id);
    // Mutating a "read-only" context's metrics is unusual, but resetting
    // the outcome window is the only way a HalfOpen probe's result can be
    // told apart from the stale failures that tripped the breaker in the
    // first place — see NodeBreakerState's comment in router.h.
    auto* metrics = const_cast<MetricsRegistry*>(ctx.metrics);

    using clock = std::chrono::steady_clock;
    auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();

    BreakerState s = node_state.state.load();

    if (s == BreakerState::Closed) {
        auto outcomes = metrics->outcomeStats(ctx.node->id);
        size_t total = outcomes.successes + outcomes.failures;
        if (total < min_samples_ || outcomes.errorRate() < error_rate_threshold_) {
            return std::nullopt; // healthy, no opinion
        }
        BreakerState expected = BreakerState::Closed;
        if (node_state.state.compare_exchange_strong(expected, BreakerState::Open)) {
            node_state.tripped_at_steady_ns.store(now_ns);
        }
        s = BreakerState::Open;
    }

    if (s == BreakerState::Open) {
        double elapsed_ms = static_cast<double>(now_ns - node_state.tripped_at_steady_ns.load()) / 1e6;
        if (elapsed_ms < cooldown_ms_) {
            RoutingDecision d;
            d.policy_name = name();
            d.skip = true;
            d.reason = "circuit open for node '" + ctx.node->id + "'; cooling down " +
                       std::to_string(cooldown_ms_ - elapsed_ms) + "ms more";
            return d;
        }
        // Cooldown elapsed: exactly one caller wins the Open->HalfOpen
        // transition and becomes the probe.
        BreakerState expected = BreakerState::Open;
        if (node_state.state.compare_exchange_strong(expected, BreakerState::HalfOpen)) {
            metrics->resetOutcomes(ctx.node->id); // clean slate to judge only this probe
            return std::nullopt;                  // let the probe run normally
        }
        s = BreakerState::HalfOpen;
    }

    // HalfOpen: only the probe (the caller that won the CAS above) should
    // ever actually execute the node — every concurrent caller lands here
    // instead and must be shed, since letting more than one through would
    // defeat the point of probing with a single request. Whether the
    // probe has an outcome recorded yet distinguishes "still shedding
    // while the probe runs" from "the probe just settled the question".
    auto probe_outcome = metrics->outcomeStats(ctx.node->id);
    size_t probe_total = probe_outcome.successes + probe_outcome.failures;
    if (probe_total == 0) {
        RoutingDecision d;
        d.policy_name = name();
        d.skip = true;
        d.reason = "circuit half-open for node '" + ctx.node->id + "'; a probe request is already in flight";
        return d;
    }
    if (probe_outcome.successes > 0) {
        BreakerState expected = BreakerState::HalfOpen;
        node_state.state.compare_exchange_strong(expected, BreakerState::Closed);
        return std::nullopt; // probe succeeded: closed again, healthy
    }
    BreakerState expected = BreakerState::HalfOpen;
    if (node_state.state.compare_exchange_strong(expected, BreakerState::Open)) {
        node_state.tripped_at_steady_ns.store(now_ns); // probe failed: cooldown restarts
    }
    RoutingDecision d;
    d.policy_name = name();
    d.skip = true;
    d.reason = "circuit re-opened for node '" + ctx.node->id + "': half-open probe failed";
    return d;
}

std::optional<RoutingDecision> BulkheadPolicy::decide(const RoutingContext& ctx) const {
    if (ctx.node == nullptr || ctx.metrics == nullptr) return std::nullopt;
    size_t in_flight = ctx.metrics->inFlight(ctx.node->id);
    if (in_flight < max_concurrent_) return std::nullopt;

    RoutingDecision d;
    d.policy_name = name();
    d.skip = true;
    d.reason = "bulkhead: node '" + ctx.node->id + "' already has " + std::to_string(in_flight) +
               " in-flight requests (cap " + std::to_string(max_concurrent_) + "); shedding";
    return d;
}

void CompositeRouter::add(std::shared_ptr<IRoutingPolicy> policy) { policies_.push_back(std::move(policy)); }

std::optional<RoutingDecision> CompositeRouter::decide(const RoutingContext& ctx) const {
    RoutingDecision merged;
    merged.policy_name = name();
    bool any = false;
    std::vector<std::string> reasons;

    for (const auto& p : policies_) {
        auto opinion = p->decide(ctx);
        if (!opinion.has_value()) continue;
        any = true;
        if (opinion->skip) {
            // A skip decision is terminal: later policies don't get a say
            // over whether to run a node that's already been skipped.
            opinion->policy_name = merged.policy_name + " -> " + opinion->policy_name;
            return opinion;
        }
        if (!merged.precision.has_value() && opinion->precision.has_value()) merged.precision = opinion->precision;
        if (!merged.backend.has_value() && opinion->backend.has_value()) merged.backend = opinion->backend;
        reasons.push_back(opinion->policy_name + ": " + opinion->reason);
    }

    if (!any) return std::nullopt;
    for (size_t i = 0; i < reasons.size(); ++i) {
        if (i) merged.reason += " | ";
        merged.reason += reasons[i];
    }
    return merged;
}

} // namespace loomcore
