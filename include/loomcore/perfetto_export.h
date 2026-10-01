// Loomcore — Perfetto/Chrome Trace Event Format export.
//
// Loomcore's JSON-lines log (loomcore/logger.h) is a complete, ordered
// record of every scheduling decision, batch flush, and node completion —
// but reading a scheduler's behavior out of a scrolling list of JSON
// objects is exactly the kind of task a timeline view is built for. This
// converts a JSONL log file into Chrome's Trace Event Format (the format
// Perfetto and chrome://tracing both read natively — no library needed
// on either end, just an array of small JSON objects with well-known
// field names), so `loomcore_trace_export logs/loomcore.jsonl trace.json`
// followed by dropping trace.json onto https://ui.perfetto.dev renders
// every lane as its own track, each node execution as a duration bar
// sized to its real measured latency, and each job's path through the
// DAG as a connecting flow arrow.
//
// A deliberate post-processing step, not something the hot logging path
// does inline: it reads a *file* Logger already wrote, so exporting a
// trace costs nothing during the run being traced.
#pragma once

#include <string>

#include "loomcore/export.h"

namespace loomcore {

// Reads the JSON-lines log at `jsonl_path` and writes a Chrome Trace
// Event Format JSON file to `output_json_path`. Returns the number of
// trace events written. Throws LoomcoreError if `jsonl_path` can't be
// opened; malformed individual lines are skipped (a partially-written log
// from a still-running or killed process is a real scenario, not one
// worth failing the whole export over).
LOOMCORE_API size_t exportPerfettoTrace(const std::string& jsonl_path, const std::string& output_json_path);

} // namespace loomcore
