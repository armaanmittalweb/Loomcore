// Loomcore — rolling per-node latency stats and queue-depth tracking.
//
// This is the runtime-state feed the router reads from: it is intentionally
// separate from Logger (which is an append-only record of what happened)
// because the router needs cheap, current aggregates, not a log to replay.
#pragma once

#include <cstddef>
#include <deque>
#include <map>
#include <mutex>
#include <string>

#include "loomcore/export.h"

namespace loomcore {

class LOOMCORE_API MetricsRegistry {
public:
    struct Stats {
        double p50_ms = 0.0;
        double p95_ms = 0.0;
        double mean_ms = 0.0;
        size_t count = 0;
    };

    explicit MetricsRegistry(size_t window = 256) : window_(window) {}

    void recordLatency(const std::string& node_key, double latency_ms);
    Stats stats(const std::string& node_key) const;

    void setQueueDepth(const std::string& lane_key, size_t depth);
    size_t queueDepth(const std::string& lane_key) const;

private:
    size_t window_;
    mutable std::mutex mutex_;
    std::map<std::string, std::deque<double>> samples_;
    std::map<std::string, size_t> queue_depth_;
};

} // namespace loomcore
