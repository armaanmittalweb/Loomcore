// Unit test for loomcore::exportPerfettoTrace (loomcore/perfetto_export.h):
// writes a small synthetic JSONL log matching the exact shape
// loomcore::Logger produces, exports it, and checks the resulting Chrome
// Trace Event Format JSON is structurally correct — not a full Perfetto
// UI round-trip (out of scope for a unit test), but enough to catch a
// broken timestamp parse or a malformed event silently producing an
// empty or invalid trace.
#include <doctest/doctest.h>

#include "loomcore/perfetto_export.h"
#include "loomcore/types.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using namespace loomcore;

namespace {

std::string writeSyntheticLog(const std::string& path) {
    std::ofstream out(path);
    out << R"({"ts":"2026-01-01T00:00:00.000000Z","event":"job_submitted","job_id":"job-0","backend":"CPU","precision":"FP32","queue_depth":0})"
        << "\n";
    out << R"({"ts":"2026-01-01T00:00:00.001000Z","event":"node_scheduled","job_id":"job-0","node_id":"a","backend":"CPU","precision":"FP32","queue_depth":0})"
        << "\n";
    out << R"({"ts":"2026-01-01T00:00:00.002000Z","event":"batch_flushed","job_id":"job-0","node_id":"a","backend":"CPU","precision":"FP32","batch_size":1,"latency_ms":1.5,"queue_depth":0})"
        << "\n";
    out << R"({"ts":"2026-01-01T00:00:00.003500Z","event":"routing_decision","job_id":"job-0","node_id":"b","backend":"CPU","precision":"FP32","message":"test reason","queue_depth":0})"
        << "\n";
    out << R"({"ts":"2026-01-01T00:00:00.004000Z","event":"node_scheduled","job_id":"job-0","node_id":"b","backend":"GPU_SIM","precision":"FP32","queue_depth":0})"
        << "\n";
    out << R"({"ts":"2026-01-01T00:00:00.006000Z","event":"batch_flushed","job_id":"job-0","node_id":"b","backend":"GPU_SIM","precision":"FP32","batch_size":1,"latency_ms":2.0,"queue_depth":0})"
        << "\n";
    out << R"({"ts":"2026-01-01T00:00:00.008000Z","event":"job_completed","job_id":"job-0","backend":"CPU","precision":"FP32","queue_depth":0})"
        << "\n";
    out << "this line is not valid JSON at all\n"; // must be skipped, not fail the export
    return path;
}

} // namespace

TEST_CASE("exportPerfettoTrace produces a structurally valid Chrome Trace Event Format file") {
    std::string jsonl_path = "perfetto_export_test.jsonl";
    std::string out_path = "perfetto_export_test_trace.json";
    writeSyntheticLog(jsonl_path);

    size_t written = exportPerfettoTrace(jsonl_path, out_path);
    CHECK(written > 0);

    nlohmann::json doc;
    {
        std::ifstream in(out_path);
        REQUIRE(in.good());
        in >> doc; // throws if malformed — the test itself is the assertion here
    } // closed before std::remove() below — an open handle blocks deletion on Windows
    REQUIRE(doc.contains("traceEvents"));
    REQUIRE(doc["traceEvents"].is_array());

    bool found_cpu_track = false, found_gpu_track = false, found_duration_a = false, found_duration_b = false,
         found_routing_marker = false, found_flow_start = false, found_flow_finish = false;
    for (const auto& ev : doc["traceEvents"]) {
        if (ev.value("ph", "") == "M" && ev.value("name", "") == "thread_name") {
            std::string track_name = ev["args"].value("name", "");
            if (track_name == "CPU lane") found_cpu_track = true;
            if (track_name == "GPU_SIM lane") found_gpu_track = true;
        }
        if (ev.value("ph", "") == "X") {
            if (ev.value("name", "") == "a" && ev.value("dur", 0) == 1500) found_duration_a = true;
            if (ev.value("name", "") == "b" && ev.value("dur", 0) == 2000) found_duration_b = true;
        }
        if (ev.value("ph", "") == "i" && ev.value("name", "") == "routing_decision") found_routing_marker = true;
        if (ev.value("ph", "") == "s" && ev.value("cat", "") == "dag_edge") found_flow_start = true;
        if (ev.value("ph", "") == "f" && ev.value("cat", "") == "dag_edge") found_flow_finish = true;
    }
    CHECK(found_cpu_track);
    CHECK(found_gpu_track);
    CHECK(found_duration_a);
    CHECK(found_duration_b);
    CHECK(found_routing_marker);
    CHECK(found_flow_start); // node "a" -> node "b" for job-0
    CHECK(found_flow_finish);

    std::remove(jsonl_path.c_str());
    std::remove(out_path.c_str());
}

TEST_CASE("exportPerfettoTrace places each bar at its real start: batch_flushed is logged when the batch ends") {
    // Logger stamps batch_flushed after BackendLane::executeBatch has timed
    // the run, so the event's ts is the bar's END. Exporting it as the start
    // would draw every execution one full duration late (and draw a
    // dependent node's bar overlapping its own upstream).
    std::string jsonl_path = "perfetto_export_start_test.jsonl";
    std::string out_path = "perfetto_export_start_test_trace.json";
    {
        std::ofstream out(jsonl_path);
        out << R"({"ts":"2026-01-01T00:00:00.000000Z","event":"job_submitted","job_id":"job-0","backend":"CPU","precision":"FP32","queue_depth":0})"
            << "\n";
        // Ran from t=1ms to t=11ms.
        out << R"({"ts":"2026-01-01T00:00:00.011000Z","event":"batch_flushed","job_id":"job-0","node_id":"a","backend":"CPU","precision":"FP32","batch_size":1,"latency_ms":10.0,"queue_depth":0})"
            << "\n";
        out << R"({"ts":"2026-01-01T00:00:00.011010Z","event":"node_completed","job_id":"job-0","node_id":"a","backend":"CPU","precision":"FP32","latency_ms":10.0,"queue_depth":0})"
            << "\n";
        // Ran from t=12ms to t=14ms on the other lane.
        out << R"({"ts":"2026-01-01T00:00:00.014000Z","event":"batch_flushed","job_id":"job-0","node_id":"b","backend":"GPU_SIM","precision":"INT8","batch_size":1,"latency_ms":2.0,"queue_depth":0})"
            << "\n";
    }
    exportPerfettoTrace(jsonl_path, out_path);
    nlohmann::json doc;
    {
        std::ifstream in(out_path);
        in >> doc;
    }
    int64_t a_ts = -1, b_ts = -1;
    for (const auto& ev : doc["traceEvents"]) {
        if (ev.value("ph", "") != "X") continue;
        if (ev.value("name", "") == "a") a_ts = ev.value("ts", int64_t{-1});
        if (ev.value("name", "") == "b") b_ts = ev.value("ts", int64_t{-1});
    }
    CHECK(a_ts == 1000);         // 11ms end - 10ms latency, relative to job_submitted at 0
    CHECK(b_ts == 12000);        // 14ms end - 2ms latency
    CHECK(a_ts + 10000 <= b_ts); // the dependent starts after its upstream ends
    std::remove(jsonl_path.c_str());
    std::remove(out_path.c_str());
}

TEST_CASE("exportPerfettoTrace attributes a multi-job batch to its jobs and binds DAG flows to lane slices") {
    // A batch of two jobs logs job_id "(2 jobs)"; the node_completed events
    // the scheduler logs right after it (same node, lane, precision and
    // latency_ms) say which jobs it served. Each job's flow arrow must then
    // run from the bar that served it upstream to the one that served it
    // downstream, on those bars' own lane tracks (tid 1 = CPU, 2 = GPU_SIM),
    // so Perfetto can bind both ends to real slices.
    std::string jsonl_path = "perfetto_export_batch_test.jsonl";
    std::string out_path = "perfetto_export_batch_test_trace.json";
    {
        std::ofstream out(jsonl_path);
        out << R"J({"ts":"2026-01-01T00:00:00.010000Z","event":"batch_flushed","job_id":"(2 jobs)","node_id":"mobilenet","backend":"CPU","precision":"FP32","batch_size":2,"latency_ms":10.0,"queue_depth":0})J"
            << "\n";
        out << R"({"ts":"2026-01-01T00:00:00.010005Z","event":"node_completed","job_id":"job-7","node_id":"mobilenet","backend":"CPU","precision":"FP32","latency_ms":10.0,"queue_depth":0})"
            << "\n";
        out << R"({"ts":"2026-01-01T00:00:00.010006Z","event":"node_completed","job_id":"job-8","node_id":"mobilenet","backend":"CPU","precision":"FP32","latency_ms":10.0,"queue_depth":0})"
            << "\n";
        out << R"({"ts":"2026-01-01T00:00:00.016000Z","event":"batch_flushed","job_id":"job-7","node_id":"bert_tiny","backend":"GPU_SIM","precision":"FP32","batch_size":1,"latency_ms":4.0,"queue_depth":0})"
            << "\n";
        out << R"({"ts":"2026-01-01T00:00:00.016004Z","event":"node_completed","job_id":"job-7","node_id":"bert_tiny","backend":"GPU_SIM","precision":"FP32","latency_ms":4.0,"queue_depth":0})"
            << "\n";
        out << R"({"ts":"2026-01-01T00:00:00.018000Z","event":"batch_flushed","job_id":"job-8","node_id":"bert_tiny","backend":"GPU_SIM","precision":"FP32","batch_size":1,"latency_ms":3.0,"queue_depth":0})"
            << "\n";
    }
    exportPerfettoTrace(jsonl_path, out_path);
    nlohmann::json doc;
    {
        std::ifstream in(out_path);
        in >> doc;
    }

    std::vector<std::string> batch_jobs;
    int flow_starts_on_cpu = 0, flow_ends_on_gpu = 0, flows = 0;
    for (const auto& ev : doc["traceEvents"]) {
        if (ev.value("ph", "") == "X" && ev.value("name", "") == "mobilenet") {
            for (const auto& j : ev["args"]["jobs"]) batch_jobs.push_back(j.get<std::string>());
            CHECK(ev.value("ts", int64_t{-1}) == 0); // 10ms end - 10ms latency
        }
        if (ev.value("cat", "") == "dag_edge") {
            if (ev.value("ph", "") == "s") {
                ++flows;
                if (ev.value("tid", 0) == 1) ++flow_starts_on_cpu;
                CHECK(ev.value("ts", int64_t{-1}) > 0); // inside the mobilenet bar [0, 10000]
                CHECK(ev.value("ts", int64_t{-1}) < 10000);
            }
            if (ev.value("ph", "") == "f") {
                if (ev.value("tid", 0) == 2) ++flow_ends_on_gpu;
                CHECK(ev.value("bp", "") == "e");
            }
        }
    }
    CHECK(batch_jobs == std::vector<std::string>{"job-7", "job-8"});
    CHECK(flows == 2); // one mobilenet -> bert_tiny edge per job
    CHECK(flow_starts_on_cpu == 2);
    CHECK(flow_ends_on_gpu == 2);
    std::remove(jsonl_path.c_str());
    std::remove(out_path.c_str());
}

TEST_CASE("exportPerfettoTrace throws a LoomcoreError for a missing input file") {
    CHECK_THROWS_AS(exportPerfettoTrace("this_file_does_not_exist.jsonl", "out.json"), LoomcoreError);
}
