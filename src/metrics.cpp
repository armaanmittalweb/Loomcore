#include "loomcore/metrics.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace loomcore {

MetricsRegistry::NodeEntry& MetricsRegistry::entryFor(const std::string& key) const {
    {
        std::shared_lock<std::shared_mutex> read_lock(map_mutex_);
        auto it = entries_.find(key);
        if (it != entries_.end()) return *it->second;
    }
    // Not present yet: upgrade to an exclusive lock and insert. Racing
    // inserts of the same brand-new key are resolved by try_emplace; this
    // path is taken at most once per distinct node_key over the registry's
    // lifetime (every subsequent access hits the shared_lock fast path
    // above), so the exclusive lock here costs nothing measurable overall.
    std::unique_lock<std::shared_mutex> write_lock(map_mutex_);
    auto [it, inserted] = entries_.try_emplace(key, nullptr);
    if (inserted) it->second = std::make_unique<NodeEntry>();
    return *it->second;
}

std::atomic<size_t>& MetricsRegistry::queueDepthEntryFor(const std::string& key) const {
    {
        std::shared_lock<std::shared_mutex> read_lock(map_mutex_);
        auto it = queue_depth_.find(key);
        if (it != queue_depth_.end()) return *it->second;
    }
    std::unique_lock<std::shared_mutex> write_lock(map_mutex_);
    auto [it, inserted] = queue_depth_.try_emplace(key, nullptr);
    if (inserted) it->second = std::make_unique<std::atomic<size_t>>(0);
    return *it->second;
}

void MetricsRegistry::recordLatency(const std::string& node_key, double latency_ms) {
    auto& entry = entryFor(node_key);
    std::lock_guard<std::mutex> lock(entry.mutex);
    entry.samples.push_back(latency_ms);
    while (entry.samples.size() > window_) entry.samples.pop_front();
}

MetricsRegistry::Stats MetricsRegistry::stats(const std::string& node_key) const {
    auto& entry = entryFor(node_key);
    Stats s;
    std::lock_guard<std::mutex> lock(entry.mutex);
    if (entry.samples.empty()) return s;

    std::vector<double> sorted(entry.samples.begin(), entry.samples.end());
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
    queueDepthEntryFor(lane_key).store(depth, std::memory_order_relaxed);
}

size_t MetricsRegistry::queueDepth(const std::string& lane_key) const {
    std::shared_lock<std::shared_mutex> read_lock(map_mutex_);
    auto it = queue_depth_.find(lane_key);
    return it == queue_depth_.end() ? 0 : it->second->load(std::memory_order_relaxed);
}

void MetricsRegistry::incrementInFlight(const std::string& node_key) {
    entryFor(node_key).in_flight.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::decrementInFlight(const std::string& node_key) {
    entryFor(node_key).in_flight.fetch_sub(1, std::memory_order_relaxed);
}

size_t MetricsRegistry::inFlight(const std::string& node_key) const {
    return entryFor(node_key).in_flight.load(std::memory_order_relaxed);
}

void MetricsRegistry::recordOutcome(const std::string& node_key, bool success) {
    auto& entry = entryFor(node_key);
    std::lock_guard<std::mutex> lock(entry.mutex);
    entry.outcomes.push_back(success);
    while (entry.outcomes.size() > window_) entry.outcomes.pop_front();
}

MetricsRegistry::OutcomeStats MetricsRegistry::outcomeStats(const std::string& node_key) const {
    auto& entry = entryFor(node_key);
    OutcomeStats s;
    std::lock_guard<std::mutex> lock(entry.mutex);
    for (bool ok : entry.outcomes) {
        if (ok)
            ++s.successes;
        else
            ++s.failures;
    }
    return s;
}

void MetricsRegistry::resetOutcomes(const std::string& node_key) {
    auto& entry = entryFor(node_key);
    std::lock_guard<std::mutex> lock(entry.mutex);
    entry.outcomes.clear();
}

} // namespace loomcore
