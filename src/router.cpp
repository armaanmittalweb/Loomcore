#include "loomcore/router.h"

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
