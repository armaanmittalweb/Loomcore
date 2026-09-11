#include "loomcore/metrics.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace loomcore {

void MetricsRegistry::recordLatency(const std::string& node_key, double latency_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& dq = samples_[node_key];
    dq.push_back(latency_ms);
    while (dq.size() > window_) dq.pop_front();
}

MetricsRegistry::Stats MetricsRegistry::stats(const std::string& node_key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    Stats s;
    auto it = samples_.find(node_key);
    if (it == samples_.end() || it->second.empty()) return s;

    std::vector<double> sorted(it->second.begin(), it->second.end());
    std::sort(sorted.begin(), sorted.end());
    s.count = sorted.size();
    s.mean_ms = std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());

    auto pctl = [&](double p) {
        if (sorted.size() == 1) return sorted[0];
        double rank = p * static_cast<double>(sorted.size() - 1);
        size_t lo = static_cast<size_t>(rank);
        size_t hi = std::min(lo + 1, sorted.size() - 1);
        double frac = rank - static_cast<double>(lo);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
    };
    s.p50_ms = pctl(0.50);
    s.p95_ms = pctl(0.95);
    return s;
}

void MetricsRegistry::setQueueDepth(const std::string& lane_key, size_t depth) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_depth_[lane_key] = depth;
}

size_t MetricsRegistry::queueDepth(const std::string& lane_key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = queue_depth_.find(lane_key);
    return it == queue_depth_.end() ? 0 : it->second;
}

} // namespace loomcore
