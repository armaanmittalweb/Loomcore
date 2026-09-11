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

    std::future<JobResult> submit(TensorMap inputs, int priority = 0, double time_budget_ms = -1.0);
    JobResult run(TensorMap inputs, int priority = 0, double time_budget_ms = -1.0);

    MetricsRegistry::Stats nodeStats(const std::string& node_id) const;
    std::vector<std::string> recentLogs(size_t n = 100) const;
    const Graph& graph() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace loomcore
