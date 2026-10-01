#include "loomcore/logger.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace loomcore {

const char* toString(LogEventType t) {
    switch (t) {
        case LogEventType::JobSubmitted: return "job_submitted";
        case LogEventType::RoutingDecision: return "routing_decision";
        case LogEventType::NodeScheduled: return "node_scheduled";
        case LogEventType::BatchFlushed: return "batch_flushed";
        case LogEventType::NodeCompleted: return "node_completed";
        case LogEventType::NodeSkipped: return "node_skipped";
        case LogEventType::JobCompleted: return "job_completed";
        case LogEventType::JobRejected: return "job_rejected";
        case LogEventType::Error: return "error";
    }
    return "unknown";
}

namespace {
std::string isoTimestamp(std::chrono::system_clock::time_point tp) {
    using namespace std::chrono;
    auto us = duration_cast<microseconds>(tp.time_since_epoch()) % 1000000;
    std::time_t t = system_clock::to_time_t(tp);
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(6) << us.count() << 'Z';
    return oss.str();
}

std::string serialize(const LogEvent& e) {
    nlohmann::json j;
    j["ts"] = isoTimestamp(e.ts);
    j["event"] = toString(e.type);
    if (!e.job_id.empty()) j["job_id"] = e.job_id;
    if (!e.node_id.empty()) j["node_id"] = e.node_id;
    j["backend"] = toString(e.backend);
    j["precision"] = toString(e.precision);
    if (e.batch_size) j["batch_size"] = e.batch_size;
    if (e.latency_ms > 0.0) j["latency_ms"] = e.latency_ms;
    j["queue_depth"] = e.queue_depth;
    if (!e.message.empty()) j["message"] = e.message;
    return j.dump();
}
} // namespace

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

Logger::Logger() {
    writer_thread_ = std::thread([this] { writerLoop(); });
}

Logger::~Logger() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_.store(true);
    }
    cv_.notify_all();
    if (writer_thread_.joinable()) writer_thread_.join();
}

void Logger::configure(const std::string& file_path, bool also_stdout, size_t ring_capacity) {
    std::lock_guard<std::mutex> lock(mutex_);
    also_stdout_ = also_stdout;
    ring_capacity_ = ring_capacity;
    if (!file_path.empty()) {
        std::filesystem::path p(file_path);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
        file_.open(file_path, std::ios::out | std::ios::app);
    }
}

void Logger::log(const LogEvent& e) {
    // Serialization is pure CPU work (no I/O) and happens on the calling
    // thread — typically a scheduler lane worker — same as before. What no
    // longer happens here is the actual write: this only ever pushes onto
    // a queue the writer thread owns.
    std::string line = serialize(e);

    std::lock_guard<std::mutex> lock(mutex_);
    ring_.push_back(line);
    while (ring_.size() > ring_capacity_) ring_.pop_front();

    ++enqueued_seq_;
    if (pending_.size() < kMaxPendingLines) {
        pending_.push_back(std::move(line));
    } else {
        // The writer is backed up past its cap (a stalled disk, or a
        // burst far exceeding sustained I/O throughput). Dropping here
        // (rather than blocking the caller — a scheduler lane worker —
        // until the writer catches up) is the honest choice: a logger
        // must never become the reason inference throughput stalls.
        // recentLines() is unaffected either way (it already has the
        // line, above); only durable file/stdout output loses it.
        ++dropped_;
        ++written_seq_; // a dropped line still counts as "settled" for flush()'s purposes
    }
    cv_.notify_one();
}

void Logger::writerLoop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        cv_.wait(lock, [&] { return stop_.load() || !pending_.empty(); });
        if (pending_.empty() && stop_.load()) return;

        while (!pending_.empty()) {
            std::string line = std::move(pending_.front());
            pending_.pop_front();
            if (also_stdout_) std::cout << line << '\n';
            if (file_.is_open()) file_ << line << '\n';
            ++written_seq_;
        }
        if (also_stdout_) std::cout.flush();
        if (file_.is_open()) file_.flush();
        flushed_cv_.notify_all();

        if (stop_.load() && pending_.empty()) return;
    }
}

void Logger::flush() {
    std::unique_lock<std::mutex> lock(mutex_);
    uint64_t target = enqueued_seq_;
    flushed_cv_.wait(lock, [&] { return written_seq_ >= target; });
}

std::vector<std::string> Logger::recentLines(size_t n) const {
    std::lock_guard<std::mutex> lock(mutex_);
    n = std::min(n, ring_.size());
    return std::vector<std::string>(ring_.end() - static_cast<long>(n), ring_.end());
}

void Logger::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    ring_.clear();
}

} // namespace loomcore
