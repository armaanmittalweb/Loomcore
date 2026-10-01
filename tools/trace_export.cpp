// loomcore_trace_export <input.jsonl> [output.json]
//
// Converts a Loomcore JSON-lines log into a Chrome Trace Event Format
// file — drop the output onto https://ui.perfetto.dev (or open it via
// chrome://tracing) to see every scheduler lane as its own track, each
// real ONNX Run() call as a duration bar sized to its measured latency,
// routing decisions as markers on their own track, and each job's
// observed path through the DAG as a connecting flow arrow. See
// loomcore/perfetto_export.h for exactly what each field means.
#include "loomcore/perfetto_export.h"

#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: loomcore_trace_export <input.jsonl> [output.json]\n";
        return 2;
    }
    std::string input = argv[1];
    std::string output = argc > 2 ? argv[2] : "trace.json";

    try {
        size_t n = loomcore::exportPerfettoTrace(input, output);
        std::cout << "Wrote " << n << " trace events to " << output << "\n"
                   << "Open https://ui.perfetto.dev and drop this file onto it, "
                   << "or open chrome://tracing and load it there.\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "loomcore_trace_export: " << ex.what() << "\n";
        return 1;
    }
}
