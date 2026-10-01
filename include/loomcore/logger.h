// Loomcore — structured (JSON-lines) logging of scheduling decisions and
// per-node latency, per docs/ARCHITECTURE.md "Observability".
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <ostream>
#include <string>
#include <thread>

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
    JobRejected, // admission control refused the job before dispatching anything; see SchedulerConfig::enable_admission_control
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

// Process-wide structured logger. `log()` only ever serializes an event to
// JSON and pushes the line onto an in-memory queue — it never itself
// touches stdout or the log file. A single dedicated writer thread drains
// that queue and performs the actual (buffered, batch-flushed) I/O. This
// matters because `log()` is called from every scheduler lane worker
// thread on every scheduling event: an earlier revision wrote to
// std::cout/the log file with `std::endl` (an implicit flush) directly on
// whichever lane thread produced the event, under one shared mutex, which
// meant a burst of concurrent events from both simulated backends
// serialized on console/file I/O and could stall the very lane workers
// the scheduler depends on for throughput — not something a caller could
// see in any individual recorded node latency (those are stamped before
// the log call), but a real, measurable drag on end-to-end job throughput
// under load. See docs/ARCHITECTURE.md "Observability".
//
// Every event is still kept in a bounded in-memory ring buffer
// (`recentLines`) independent of the writer queue, so tests and the
// Python bindings can inspect recent activity synchronously without
// waiting on the writer thread or re-parsing a log file.
class LOOMCORE_API Logger {
public:
    static Logger& instance();

    // `file_path` empty disables file output. Safe to call multiple times
    // (e.g. once per Runtime instance in a test process).
    void configure(const std::string& file_path, bool also_stdout, size_t ring_capacity = 2048);

    void log(const LogEvent& e);

    // Most recent `n` log lines (JSON, one per entry), oldest first.
    // Reflects every logged event immediately (backed by the ring buffer,
    // not the writer queue), regardless of whether the writer thread has
    // caught up on file/stdout output yet.
    std::vector<std::string> recentLines(size_t n) const;

    void clear();

    // Blocks until every line enqueued before this call returns has been
    // written to stdout/file (or dropped — see droppedCount()). Tests that
    // assert against the log *file's* contents (rather than recentLines(),
    // which needs no flush) should call this first.
    void flush();

    // Lines that were discarded because the writer's queue was backed up
    // past its cap (a slow/blocked disk, an extreme event burst) rather
    // than block the caller indefinitely. 0 in every ordinary run.
    size_t droppedCount() const { return dropped_.load(); }

private:
    Logger();
    ~Logger();
    void writerLoop();

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable flushed_cv_;
    bool also_stdout_ = true;
    std::ofstream file_;
    std::deque<std::string> ring_;
    size_t ring_capacity_ = 2048;

    std::deque<std::string> pending_; // lines not yet written; drained by writer_thread_
    static constexpr size_t kMaxPendingLines = 200000;
    std::atomic<size_t> dropped_{0};
    uint64_t enqueued_seq_ = 0; // total lines ever pushed
    uint64_t written_seq_ = 0;  // total lines ever written-or-dropped; flush() waits for this to catch up

    std::thread writer_thread_;
    std::atomic<bool> stop_{false};
};

} // namespace loomcore
