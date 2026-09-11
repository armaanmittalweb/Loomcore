// Loomcore — structured (JSON-lines) logging of scheduling decisions and
// per-node latency, per docs/ARCHITECTURE.md "Observability".
#pragma once

#include <chrono>
#include <deque>
#include <fstream>
#include <mutex>
#include <ostream>
#include <string>

#include "loomcore/export.h"
#include "loomcore/types.h"

namespace loomcore {

enum class LogEventType {
    JobSubmitted,
    RoutingDecision,
    NodeScheduled,
    BatchFlushed,
    NodeCompleted,
    NodeSkipped,
    JobCompleted,
    Error,
};

LOOMCORE_API const char* toString(LogEventType t);
inline std::ostream& operator<<(std::ostream& os, LogEventType t) { return os << toString(t); }

struct LogEvent {
    LogEventType type;
    std::chrono::system_clock::time_point ts = std::chrono::system_clock::now();
    std::string job_id;
    std::string node_id;
    Backend backend = Backend::CPU;
    Precision precision = Precision::FP32;
    size_t batch_size = 0;
    double latency_ms = 0.0;
    size_t queue_depth = 0;
    std::string message; // free-form detail, e.g. a routing policy's reason
};

// Process-wide structured logger. Every event is serialized to one JSON
// object per line (JSON-lines), written to an optional file and/or stdout,
// and kept in a bounded in-memory ring buffer so the Python bindings and
// tests can retrieve recent activity without re-parsing a log file.
class LOOMCORE_API Logger {
public:
    static Logger& instance();

    // `file_path` empty disables file output. Safe to call multiple times
    // (e.g. once per Runtime instance in a test process).
    void configure(const std::string& file_path, bool also_stdout, size_t ring_capacity = 2048);

    void log(const LogEvent& e);

    // Most recent `n` log lines (JSON, one per entry), oldest first.
    std::vector<std::string> recentLines(size_t n) const;

    void clear();

private:
    Logger() = default;

    mutable std::mutex mutex_;
    bool also_stdout_ = true;
    std::ofstream file_;
    std::deque<std::string> ring_;
    size_t ring_capacity_ = 2048;
};

} // namespace loomcore
