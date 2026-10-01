#include "loomcore/perfetto_export.h"

#include "loomcore/types.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
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

// One real ModelVariant::run() call (one batch), reconstructed from its
// batch_flushed event plus the node_completed events of the jobs it served.
struct Execution {
    int64_t start_us = 0; // absolute
    int64_t end_us = 0;   // absolute
    int64_t dur_us = 0;
    std::string node_id;
    std::string backend;
    std::string precision;
    std::string job_id_field; // batch_flushed's own job_id: a job id, or "(N jobs)" for a multi-job batch
    size_t batch_size = 0;
    double latency_ms = 0.0;
    std::vector<std::string> jobs; // which jobs this one execution served
};

struct Instant {
    int64_t ts_us = 0; // absolute
    std::string type;
    std::string job_id;
    std::string node_id;
    std::string message;
};

bool isRealJobId(const std::string& id) { return !id.empty() && id.front() != '('; }

} // namespace

// Two passes over the log. The first collects every execution and marker;
// the second emits them relative to the earliest instant seen. Two
// properties of the log shape this:
//
// - batch_flushed is logged when a batch *finishes* (BackendLane::executeBatch
//   builds its LogEvent after timing the run), so the event's own timestamp
//   is the end of the bar, not its start. The bar starts latency_ms earlier.
// - A batch serving several jobs logs job_id "(N jobs)". Which jobs it served
//   is recovered from the node_completed events the scheduler logs for each
//   of them straight afterwards, carrying the exact same node, lane,
//   precision and latency_ms (the batch's wall-clock time is attributed to
//   every item; see docs/ARCHITECTURE.md "Scheduler").
//
// Knowing each job's executions lets each DAG edge be drawn as a flow arrow
// from the bar that ran the upstream node for that job to the bar that ran
// the downstream one, on their real lane tracks, which is what Perfetto
// needs to bind a flow to its slices.
size_t exportPerfettoTrace(const std::string& jsonl_path, const std::string& output_json_path) {
    std::ifstream in(jsonl_path);
    if (!in) throw LoomcoreError("exportPerfettoTrace: cannot open '" + jsonl_path + "'");

    std::vector<Execution> executions;
    std::vector<Instant> instants;
    int64_t base_ts = -1;
    auto seeBase = [&](int64_t ts) {
        if (base_ts < 0 || ts < base_ts) base_ts = ts;
    };

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        nlohmann::json ev;
        try {
            ev = nlohmann::json::parse(line);
        } catch (...) {
            continue; // a partially-written trailing line from a killed process, etc.
        }
        if (!ev.is_object() || !ev.contains("ts") || !ev.contains("event") || !ev["ts"].is_string()) continue;
        int64_t ts = parseIsoTimestampMicros(ev["ts"].get<std::string>());
        if (ts < 0) continue;

        std::string type = ev.value("event", "");
        std::string job_id = ev.value("job_id", "");
        std::string node_id = ev.value("node_id", "");
        std::string backend = ev.value("backend", "");
        std::string precision = ev.value("precision", "");

        if (type == "batch_flushed") {
            Execution ex;
            ex.latency_ms = ev.value("latency_ms", 0.0);
            ex.dur_us = static_cast<int64_t>(ex.latency_ms * 1000.0);
            ex.end_us = ts;
            ex.start_us = ts - ex.dur_us;
            ex.node_id = node_id;
            ex.backend = backend;
            ex.precision = precision;
            ex.job_id_field = job_id;
            ex.batch_size = ev.value("batch_size", static_cast<size_t>(0));
            seeBase(ex.start_us);
            executions.push_back(std::move(ex));
        } else if (type == "node_completed") {
            seeBase(ts);
            if (!isRealJobId(job_id)) continue;
            double latency_ms = ev.value("latency_ms", 0.0);
            // Newest matching batch that still has room: node_completed
            // follows its own batch's flush on the same lane thread, so a
            // backwards search finds it within a few entries.
            for (auto it = executions.rbegin(); it != executions.rend(); ++it) {
                if (it->node_id != node_id || it->backend != backend || it->precision != precision) continue;
                if (std::fabs(it->latency_ms - latency_ms) > 1e-9) continue;
                size_t cap = std::max<size_t>(it->batch_size, 1);
                if (it->jobs.size() >= cap) continue;
                if (std::find(it->jobs.begin(), it->jobs.end(), job_id) != it->jobs.end()) break;
                it->jobs.push_back(job_id);
                break;
            }
        } else if (type == "routing_decision" || type == "job_rejected" || type == "error") {
            seeBase(ts);
            instants.push_back(Instant{ts, type, job_id, node_id, ev.value("message", "")});
        } else {
            seeBase(ts);
        }
    }

    // A single-job batch names its job in batch_flushed itself; logs that
    // carry no node_completed events (or a hand-written one) still get it.
    for (auto& ex : executions) {
        if (ex.jobs.empty() && isRealJobId(ex.job_id_field)) ex.jobs.push_back(ex.job_id_field);
    }
    if (base_ts < 0) base_ts = 0;

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

    size_t written = 0;
    for (const auto& ex : executions) {
        nlohmann::json args = {{"job_id", ex.job_id_field}, {"jobs", ex.jobs}, {"precision", ex.precision},
                                {"batch_size", ex.batch_size}};
        trace_events.push_back({{"name", ex.node_id.empty() ? "(node)" : ex.node_id},
                                 {"cat", "execution"},
                                 {"ph", "X"},
                                 {"ts", ex.start_us - base_ts},
                                 {"dur", ex.dur_us},
                                 {"pid", kPid},
                                 {"tid", trackIdForBackend(ex.backend)},
                                 {"args", args}});
        ++written;
    }

    for (const auto& in_ev : instants) {
        trace_events.push_back(
            {{"name", in_ev.type},
             {"cat", "router"},
             {"ph", "i"},
             {"s", "t"},
             {"ts", in_ev.ts_us - base_ts},
             {"pid", kPid},
             {"tid", 3},
             {"args", {{"job_id", in_ev.job_id}, {"node_id", in_ev.node_id}, {"message", in_ev.message}}}});
        ++written;
    }

    // DAG edges, per job: consecutive executions of that job, in start
    // order. The flow starts just inside the end of the upstream bar and
    // finishes ("bp":"e", bind to the enclosing slice) just inside the start
    // of the downstream bar, so each endpoint lies within its slice.
    std::map<std::string, std::vector<const Execution*>> by_job;
    for (const auto& ex : executions) {
        for (const auto& j : ex.jobs) by_job[j].push_back(&ex);
    }
    int64_t next_flow_id = 1;
    for (auto& [job, list] : by_job) {
        std::stable_sort(list.begin(), list.end(),
                         [](const Execution* a, const Execution* b) { return a->start_us < b->start_us; });
        for (size_t i = 1; i < list.size(); ++i) {
            const Execution* from = list[i - 1];
            const Execution* to = list[i];
            int64_t from_ts = from->dur_us >= 2 ? from->end_us - 1 : from->start_us;
            int64_t to_ts = to->dur_us >= 2 ? to->start_us + 1 : to->start_us;
            int64_t flow_id = next_flow_id++;
            nlohmann::json flow_args = {{"job_id", job}, {"from", from->node_id}, {"to", to->node_id}};
            trace_events.push_back({{"name", "dag_edge"}, {"cat", "dag_edge"}, {"ph", "s"}, {"id", flow_id},
                                     {"ts", from_ts - base_ts}, {"pid", kPid},
                                     {"tid", trackIdForBackend(from->backend)}, {"args", flow_args}});
            trace_events.push_back({{"name", "dag_edge"}, {"cat", "dag_edge"}, {"ph", "f"}, {"bp", "e"},
                                     {"id", flow_id}, {"ts", to_ts - base_ts}, {"pid", kPid},
                                     {"tid", trackIdForBackend(to->backend)}, {"args", flow_args}});
            written += 2;
        }
    }

    nlohmann::json doc = {{"traceEvents", trace_events}, {"displayTimeUnit", "ms"}};
    std::ofstream out(output_json_path);
    if (!out) throw LoomcoreError("exportPerfettoTrace: cannot write '" + output_json_path + "'");
    out << doc.dump(2);
    return written;
}

} // namespace loomcore
