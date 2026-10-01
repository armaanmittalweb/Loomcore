#include "loomcore/perfetto_export.h"

#include "loomcore/types.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <ctime>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace loomcore {

namespace {

#if defined(_WIN32)
int64_t timegmPortable(std::tm* tm) { return static_cast<int64_t>(_mkgmtime(tm)); }
#else
int64_t timegmPortable(std::tm* tm) { return static_cast<int64_t>(timegm(tm)); }
#endif

// Parses exactly the format loomcore::Logger writes (logger.cpp's
// isoTimestamp): "YYYY-MM-DDTHH:MM:SS.ffffffZ". Returns microseconds
// since the Unix epoch, or -1 if `s` doesn't match that shape.
int64_t parseIsoTimestampMicros(const std::string& s) {
    if (s.size() != 27 || s[4] != '-' || s[7] != '-' || s[10] != 'T' || s[13] != ':' || s[16] != ':' ||
        s[19] != '.' || s[26] != 'Z') {
        return -1;
    }
    try {
        std::tm tm{};
        tm.tm_year = std::stoi(s.substr(0, 4)) - 1900;
        tm.tm_mon = std::stoi(s.substr(5, 2)) - 1;
        tm.tm_mday = std::stoi(s.substr(8, 2));
        tm.tm_hour = std::stoi(s.substr(11, 2));
        tm.tm_min = std::stoi(s.substr(14, 2));
        tm.tm_sec = std::stoi(s.substr(17, 2));
        int64_t micros = std::stoll(s.substr(20, 6));
        int64_t epoch_seconds = timegmPortable(&tm);
        return epoch_seconds * 1000000LL + micros;
    } catch (...) {
        return -1;
    }
}

int64_t trackIdForBackend(const std::string& backend) {
    if (backend == "CPU") return 1;
    if (backend == "GPU_SIM") return 2;
    return 9; // unknown backend, kept off the two named lane tracks
}

} // namespace

size_t exportPerfettoTrace(const std::string& jsonl_path, const std::string& output_json_path) {
    std::ifstream in(jsonl_path);
    if (!in) throw LoomcoreError("exportPerfettoTrace: cannot open '" + jsonl_path + "'");

    nlohmann::json trace_events = nlohmann::json::array();
    constexpr int64_t kPid = 1;

    trace_events.push_back({{"name", "process_name"}, {"ph", "M"}, {"pid", kPid}, {"tid", 0},
                             {"args", {{"name", "Loomcore scheduler"}}}});
    trace_events.push_back({{"name", "thread_name"}, {"ph", "M"}, {"pid", kPid}, {"tid", 1},
                             {"args", {{"name", "CPU lane"}}}});
    trace_events.push_back({{"name", "thread_name"}, {"ph", "M"}, {"pid", kPid}, {"tid", 2},
                             {"args", {{"name", "GPU_SIM lane"}}}});
    trace_events.push_back({{"name", "thread_name"}, {"ph", "M"}, {"pid", kPid}, {"tid", 3},
                             {"args", {{"name", "Router"}}}});

    int64_t base_ts = -1;
    // For flow arrows: the most recent node_scheduled event's (ts, node_id)
    // per job_id, so the *next* node_scheduled for that job draws an arrow
    // from the previous one — a real, observed sequence through the DAG
    // for that job, not a reconstruction from the Graph's static topology
    // (this exporter only ever reads the log, deliberately).
    struct LastScheduled {
        int64_t ts;
        std::string node_id;
    };
    std::map<std::string, LastScheduled> last_scheduled_by_job;
    int64_t next_flow_id = 1;
    size_t written = 0;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        nlohmann::json ev;
        try {
            ev = nlohmann::json::parse(line);
        } catch (...) {
            continue; // a partially-written trailing line from a killed process, etc.
        }
        if (!ev.contains("ts") || !ev.contains("event")) continue;
        int64_t ts = parseIsoTimestampMicros(ev["ts"].get<std::string>());
        if (ts < 0) continue;
        if (base_ts < 0) base_ts = ts;
        int64_t rel_ts = ts - base_ts;

        std::string type = ev["event"].get<std::string>();
        std::string job_id = ev.value("job_id", "");
        std::string node_id = ev.value("node_id", "");
        std::string backend = ev.value("backend", "");

        if (type == "batch_flushed") {
            double latency_ms = ev.value("latency_ms", 0.0);
            nlohmann::json args = {{"job_id", job_id}, {"precision", ev.value("precision", "")},
                                    {"batch_size", ev.value("batch_size", 0)}};
            trace_events.push_back({{"name", node_id.empty() ? "(node)" : node_id},
                                     {"cat", "execution"},
                                     {"ph", "X"},
                                     {"ts", rel_ts},
                                     {"dur", static_cast<int64_t>(latency_ms * 1000.0)},
                                     {"pid", kPid},
                                     {"tid", trackIdForBackend(backend)},
                                     {"args", args}});
            ++written;
        } else if (type == "routing_decision" || type == "job_rejected" || type == "error") {
            trace_events.push_back({{"name", type},
                                     {"cat", "router"},
                                     {"ph", "i"},
                                     {"s", "t"},
                                     {"ts", rel_ts},
                                     {"pid", kPid},
                                     {"tid", 3},
                                     {"args", {{"job_id", job_id}, {"node_id", node_id}, {"message", ev.value("message", "")}}}});
            ++written;
        } else if (type == "node_scheduled" && !job_id.empty() && !node_id.empty()) {
            auto it = last_scheduled_by_job.find(job_id);
            if (it != last_scheduled_by_job.end()) {
                int64_t flow_id = next_flow_id++;
                trace_events.push_back({{"name", "dag_edge"}, {"cat", "dag_edge"}, {"ph", "s"}, {"id", flow_id},
                                         {"ts", it->second.ts - base_ts}, {"pid", kPid}, {"tid", trackIdForBackend("")}});
                trace_events.push_back({{"name", "dag_edge"}, {"cat", "dag_edge"}, {"ph", "f"}, {"id", flow_id},
                                         {"bp", "e"}, {"ts", rel_ts}, {"pid", kPid}, {"tid", trackIdForBackend("")}});
                written += 2;
            }
            last_scheduled_by_job[job_id] = LastScheduled{ts, node_id};
        }
    }

    nlohmann::json doc = {{"traceEvents", trace_events}, {"displayTimeUnit", "ms"}};
    std::ofstream out(output_json_path);
    if (!out) throw LoomcoreError("exportPerfettoTrace: cannot write '" + output_json_path + "'");
    out << doc.dump(2);
    return written;
}

} // namespace loomcore
