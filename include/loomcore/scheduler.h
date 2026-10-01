// Loomcore — the DAG scheduler: priority + dynamic batching across nodes,
// dispatched onto simulated heterogeneous backend lanes.
//
// One "job" is a single end-to-end run of the whole graph for one set of
// external inputs. Multiple jobs may be in flight concurrently; the
// scheduler batches together requests that land on the *same*
// (node, precision, shape) triple around the same time, independent of
// which job they belong to — the standard dynamic-batching pattern used by
// inference servers. See docs/ARCHITECTURE.md "Scheduler" for the full
// design and its deliberate simplifications.
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
#include "loomcore/planner.h"
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

    // --- Deadline features (Phase 04). All default OFF: every one of
    // these is new, additive behavior layered onto a scheduler that
    // already works, and none of it should change what an existing
    // caller who never touches these flags observes. See
    // docs/ARCHITECTURE.md "Deadlines" for the rationale behind shipping
    // them opt-in rather than always-on. ---

    // Reject a job outright at submitJob() (throwing JobRejectedError,
    // dispatching nothing) when PrecisionPlanner judges — from already-
    // measured per-node latency — that even downgrading every eligible
    // critical-path node to INT8 cannot fit the requested time budget.
    // Never rejects on a cold registry (no measurements yet to judge by).
    bool enable_admission_control = false;

    // Compute a PrecisionPlan (loomcore/planner.h) once per budgeted job
    // and expose it via RoutingContext::precision_plan for
    // PlannedPrecisionPolicy (or a custom policy) to read. Implied by
    // enable_admission_control (admission control needs the plan anyway);
    // settable independently to use the plan without rejecting jobs.
    bool enable_precision_planning = false;

    // Start a background "deadline reaper" thread that cancels the
    // in-flight ONNX Runtime call (via CancellationToken) and fails the
    // job with DeadlineExceededError once a job's time budget elapses,
    // rather than merely letting policies *route around* a tight budget
    // ahead of time. See CancellationToken in loomcore/model_node.h.
    bool enable_deadline_cancellation = false;
    double deadline_reaper_poll_ms = 5.0;

    // When true, a (node, precision, shape) key with at least one queued
    // task that has a job deadline is scored by that deadline (earliest
    // first) instead of the aged-priority formula, and always outranks
    // keys with no deadline-bearing task at all. Keys with no
    // deadline-bearing tasks keep using the aged-priority formula among
    // themselves either way.
    bool use_edf_scoring = false;
};

class LOOMCORE_API Scheduler {
public:
    Scheduler(const Graph& graph, const std::map<std::string, ModelNode>& nodes, MetricsRegistry& metrics,
              std::shared_ptr<IRoutingPolicy> router, SchedulerConfig cfg = {}, Logger* logger = nullptr);
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    // Submits one graph run. `time_budget_ms < 0` means unbounded (the
    // LatencyBudgetPolicy then never fires for this job, no admission
    // check runs, and no deadline reaper entry is registered). Returns
    // immediately; the returned future is satisfied once every node has
    // either completed, been routed to skip, the job failed, or — with
    // enable_admission_control on — is satisfied by an already-set
    // exception before anything dispatches at all.
    //
    // `keep_alive`, if non-null, is held for exactly as long as this job's
    // future takes to settle (captured inside the job's own internal
    // state, which already outlives every lane task referencing it). This
    // is what lets Runtime hot-swap its Graph/ModelNode set out from under
    // a live Scheduler without a use-after-free: the caller passes the
    // shared_ptr that owns the Graph/nodes this Scheduler was built with,
    // so it cannot be destroyed while this job is still in flight. See
    // docs/ARCHITECTURE.md "Hot-reloading a graph".
    std::future<JobResult> submitJob(TensorMap external_inputs, int priority = 0, double time_budget_ms = -1.0,
                                      std::shared_ptr<void> keep_alive = nullptr);

    JobResult runSync(TensorMap external_inputs, int priority = 0, double time_budget_ms = -1.0,
                       std::shared_ptr<void> keep_alive = nullptr);

    // Stops accepting new job submissions: submitJob() called after this
    // returns an already-failed future (LoomcoreError), so runSync() then
    // throws that error when it .get()s it. Jobs already in flight keep
    // running to completion — their lane tasks were already enqueued and
    // BackendLane drains its queues before its own worker threads exit. Idempotent;
    // safe to call more than once; called automatically by the destructor
    // (which then blocks joining the lane threads, so any still-draining
    // work has completed by the time the destructor returns).
    void shutdown();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace loomcore
