// Loomcore — the DAG scheduler: priority + dynamic batching across nodes,
// dispatched onto simulated heterogeneous backend lanes.
//
// One "job" is a single end-to-end run of the whole graph for one set of
// external inputs. Multiple jobs may be in flight concurrently; the
// scheduler batches together requests that land on the *same* (node,
// precision) pair around the same time, independent of which job they
// belong to — the standard dynamic-batching pattern used by inference
// servers. See docs/ARCHITECTURE.md "Scheduler" for the full design and
// its deliberate simplifications.
#pragma once

#include <future>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "loomcore/export.h"
#include "loomcore/graph.h"
#include "loomcore/logger.h"
#include "loomcore/metrics.h"
#include "loomcore/model_node.h"
#include "loomcore/router.h"

namespace loomcore {

using JobResult = std::map<std::string, std::vector<NamedTensor>>;

struct SchedulerConfig {
    int cpu_threads = 0;     // 0 = auto (hardware_concurrency / 2, min 1)
    int gpu_sim_threads = 2; // a real GPU typically serializes far fewer concurrent
                              // execution streams than a CPU has cores; kept small
                              // on purpose to make the two lanes behave differently.
    double gpu_sim_fixed_overhead_ms = 1.5;    // modeled kernel-launch / sync overhead
    double gpu_sim_bytes_per_ms = 250000.0;    // modeled host<->device transfer bandwidth
    double aging_ms_per_priority_point = 50.0; // how fast a waiting task's effective
                                                // priority climbs, to avoid starvation
};

class LOOMCORE_API Scheduler {
public:
    Scheduler(const Graph& graph, const std::map<std::string, ModelNode>& nodes, MetricsRegistry& metrics,
              std::shared_ptr<IRoutingPolicy> router, SchedulerConfig cfg = {}, Logger* logger = nullptr);
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    // Submits one graph run. `time_budget_ms < 0` means unbounded (the
    // LatencyBudgetPolicy then never fires for this job). Returns
    // immediately; the returned future is satisfied once every node has
    // either completed or been routed to skip.
    std::future<JobResult> submitJob(TensorMap external_inputs, int priority = 0, double time_budget_ms = -1.0);

    JobResult runSync(TensorMap external_inputs, int priority = 0, double time_budget_ms = -1.0);

    // Stops accepting new lane work once currently-queued tasks drain.
    // Safe to call more than once; called automatically by the destructor.
    void shutdown();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace loomcore
