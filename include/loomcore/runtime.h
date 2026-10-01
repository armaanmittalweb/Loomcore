// Loomcore — top-level facade: load a DAG config, run jobs, read metrics
// and logs. This is the one header most consumers (the example programs,
// the benchmark, the Python bindings) need to include.
#pragma once

#include <future>
#include <map>
#include <memory>
#include <string>

#include "loomcore/export.h"
#include "loomcore/graph.h"
#include "loomcore/metrics.h"
#include "loomcore/model_node.h"
#include "loomcore/router.h"
#include "loomcore/scheduler.h"

namespace loomcore {

struct RuntimeOptions {
    std::string log_file;                       // empty = stdout only
    bool log_to_stdout = true;                   // also mirror every event to stdout (Logger writes asynchronously
                                                  // either way — see loomcore/logger.h — so this is purely about
                                                  // whether you want the console stream, not about hot-path cost)
    int intra_op_threads_per_variant = 1;        // ORT's own per-session threading; kept low
                                                  // deliberately since Loomcore's scheduler is
                                                  // the layer meant to provide parallelism
                                                  // across nodes and batches.
    SchedulerConfig scheduler;
};

class LOOMCORE_API Runtime {
public:
    Runtime();
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    // Loads topology and model variants from a JSON graph config (schema:
    // docs/ARCHITECTURE.md "Graph config schema", example:
    // examples/graph_config.json). `binders` must supply a NodeInputBinder
    // for every node id declared in the config — data-flow semantics live
    // in code, not JSON (see loomcore/model_node.h for why). Nodes whose
    // config sets "confidence_source" must have a matching entry in
    // `confidence_extractors`. `router` defaults to a CompositeRouter with
    // no policies (i.e. every node runs its configured backend/FP32 with
    // no skipping) when null.
    void loadGraph(const std::string& config_path, std::map<std::string, NodeInputBinder> binders,
                    std::map<std::string, ConfidenceExtractor> confidence_extractors = {},
                    std::shared_ptr<IRoutingPolicy> router = nullptr, RuntimeOptions options = {});

    // Builds an entirely new graph/model-variant/scheduler snapshot from
    // `config_path` (same schema and arguments as loadGraph) and swaps it
    // in atomically: jobs submitted after this call see the new graph;
    // every job already in flight keeps running to completion against the
    // *old* snapshot, which is only actually destroyed once its last such
    // job settles. No submit()/run() call ever observes a torn or
    // half-updated graph, and none needs to be dropped or delayed for the
    // swap — see docs/ARCHITECTURE.md "Hot-reloading a graph" for the
    // shared_ptr-snapshot design this relies on, and
    // examples/reload_demo.cpp for a live measurement of "zero jobs
    // dropped across a swap under continuous load".
    //
    // Requires loadGraph() to have been called first. MetricsRegistry and
    // the process-wide Logger persist across the swap (so node latency
    // history for a node id that exists in both the old and new graph
    // carries over); the old scheduler's lane threads are not joined by
    // this call — they wind down on their own once its last pinned job
    // finishes, invisibly to this call's caller.
    void reloadGraph(const std::string& config_path, std::map<std::string, NodeInputBinder> binders,
                      std::map<std::string, ConfidenceExtractor> confidence_extractors = {},
                      std::shared_ptr<IRoutingPolicy> router = nullptr, RuntimeOptions options = {});

    std::future<JobResult> submit(TensorMap inputs, int priority = 0, double time_budget_ms = -1.0);
    JobResult run(TensorMap inputs, int priority = 0, double time_budget_ms = -1.0);

    MetricsRegistry::Stats nodeStats(const std::string& node_id) const;
    std::vector<std::string> recentLogs(size_t n = 100) const;

    // Reflects whichever graph snapshot is current *at the moment of this
    // call*. Holding the returned reference across a concurrent
    // reloadGraph() call is unsafe — take a copy of what you need from it
    // instead of retaining the reference if reloads may be happening
    // concurrently with your read of it.
    const Graph& graph() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace loomcore
