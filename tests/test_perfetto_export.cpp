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

#include <fstream>
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

TEST_CASE("exportPerfettoTrace throws a LoomcoreError for a missing input file") {
    CHECK_THROWS_AS(exportPerfettoTrace("this_file_does_not_exist.jsonl", "out.json"), LoomcoreError);
}
