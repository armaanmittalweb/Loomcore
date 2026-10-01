// Loomcore — rolling per-node latency stats, queue-depth, in-flight and
// error-rate tracking.
//
// This is the runtime-state feed the router reads from: it is intentionally
// separate from Logger (which is an append-only record of what happened)
// because the router needs cheap, current aggregates, not a log to replay.
//
// Sharded by key: each node/lane key owns its own mutex (guarded by a
// shared_mutex over the map structure itself, which is only ever
// write-locked the first time a given key is seen — after warmup every
// access is a shared_lock on the structure plus that one key's own lock).
// Earlier revisions used one global mutex for the whole registry, which
// meant the CPU and GPU_SIM lanes — nominally independent — serialized on
// every queue-depth update and every stats() call. See docs/ARCHITECTURE.md
// "Metrics" for the measured effect of this.
#pragma once

#include <atomic>
#include <cstddef>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
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

    // Rolling success/failure counts for a node, used by CircuitBreakerPolicy.
    struct OutcomeStats {
        size_t successes = 0;
        size_t failures = 0;
        double errorRate() const {
            size_t total = successes + failures;
            return total == 0 ? 0.0 : static_cast<double>(failures) / static_cast<double>(total);
        }
    };

    explicit MetricsRegistry(size_t window = 256) : window_(window) {}

    void recordLatency(const std::string& node_key, double latency_ms);
    Stats stats(const std::string& node_key) const;

    void setQueueDepth(const std::string& lane_key, size_t depth);
    size_t queueDepth(const std::string& lane_key) const;

    // In-flight (dispatched, not yet completed) request count for one node.
    // Incremented by the scheduler at dispatch and decremented at
    // completion/failure; read by BulkheadPolicy to cap per-node concurrency.
    void incrementInFlight(const std::string& node_key);
    void decrementInFlight(const std::string& node_key);
    size_t inFlight(const std::string& node_key) const;

    // Rolling success/failure outcome tracking for CircuitBreakerPolicy.
    void recordOutcome(const std::string& node_key, bool success);
    OutcomeStats outcomeStats(const std::string& node_key) const;
    // Resets the rolling outcome window for a node (used when a breaker
    // half-opens and wants a clean probe window).
    void resetOutcomes(const std::string& node_key);

private:
    struct NodeEntry {
        mutable std::mutex mutex;
        std::deque<double> samples;
        std::atomic<size_t> in_flight{0};
        std::deque<bool> outcomes; // true = success
    };

    NodeEntry& entryFor(const std::string& key) const;

    size_t window_;
    mutable std::shared_mutex map_mutex_; // guards structural changes to entries_/queue_depth_ only
    mutable std::map<std::string, std::unique_ptr<NodeEntry>> entries_;
    mutable std::map<std::string, std::unique_ptr<std::atomic<size_t>>> queue_depth_;

    std::atomic<size_t>& queueDepthEntryFor(const std::string& key) const;
};

} // namespace loomcore
