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
} // namespace

Logger& Logger::instance() {
    static Logger logger;
    return logger;
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

    std::string line = j.dump();

    std::lock_guard<std::mutex> lock(mutex_);
    if (also_stdout_) std::cout << line << std::endl;
    if (file_.is_open()) file_ << line << std::endl;
    ring_.push_back(line);
    while (ring_.size() > ring_capacity_) ring_.pop_front();
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
