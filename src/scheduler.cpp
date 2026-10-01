#include "loomcore/scheduler.h"

#include "loomcore/planner.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <thread>

namespace loomcore {

namespace {

std::string newJobId() {
    static std::atomic<uint64_t> counter{0};
    uint64_t n = counter.fetch_add(1);
    std::ostringstream oss;
    oss << "job-" << n;
    return oss.str();
}

// If every name in `expected` appears among `produced`'s tensor names,
// reorder `produced` to match `expected`. Otherwise, trust that the binder
// already produced tensors in positional order (this is the common case
// for hand-written binders that don't bother naming their outputs after
// the model's real input names). Either way the result's size is
// validated against `expected` by the caller.
std::vector<NamedTensor> reorderToExpected(std::vector<NamedTensor> produced, const std::vector<std::string>& expected) {
    if (produced.size() != expected.size()) return produced;
    std::map<std::string, NamedTensor*> by_name;
    for (auto& t : produced) by_name[t.name] = &t;
    bool all_present = true;
    for (const auto& name : expected) {
        if (!by_name.count(name)) {
            all_present = false;
            break;
        }
    }
    if (!all_present) return produced;
    std::vector<NamedTensor> ordered;
    ordered.reserve(expected.size());
    for (const auto& name : expected) ordered.push_back(*by_name[name]);
    return ordered;
}

double approxBytes(const std::vector<NamedTensor>& tensors) {
    double bytes = 0.0;
    for (const auto& t : tensors) {
        size_t n = t.elementCount();
        switch (t.dtype) {
            case DType::Float32: bytes += static_cast<double>(n) * 4.0; break;
            case DType::Int64: bytes += static_cast<double>(n) * 8.0; break;
            case DType::Int32: bytes += static_cast<double>(n) * 4.0; break;
        }
    }
    return bytes;
}

// A canonical, order-preserving encoding of every input tensor's shape
// *excluding* axis 0 (the batch axis every NamedTensor concatenates
// along by convention). Two requests for the same (node, precision) whose
// non-batch shapes don't match can never be validly concatenated by
// concatBatch — see NamedTensor::concatBatch in types.cpp, which requires
// every non-batch dimension to agree across parts. Folding this into the
// batching key (PendingKey below) means such requests are never even
// offered to each other as batch-mates in the first place: they land in
// separate pending queues, each flushed independently. Without this, a
// node whose binder can legitimately produce different non-batch shapes
// (e.g. variable sequence length) would have every job in an
// incompatible-shape batch fail together the moment two such requests
// happened to be dispatched in the same window, rather than only the
// (correctly) never-formed batch.
std::string shapeSignature(const std::vector<NamedTensor>& inputs) {
    std::ostringstream oss;
    for (const auto& t : inputs) {
        oss << t.name << '(';
        for (size_t i = 1; i < t.shape.size(); ++i) oss << t.shape[i] << ',';
        oss << ')';
    }
    return oss.str();
}

struct PendingKey {
    std::string node_id;
    Precision precision;
    std::string shape_sig;
    bool operator<(const PendingKey& o) const {
        if (node_id != o.node_id) return node_id < o.node_id;
        if (precision != o.precision) return precision < o.precision;
        return shape_sig < o.shape_sig;
    }
};

struct NodeTask {
    std::string job_id;
    std::string node_id;
    Precision precision;
    int priority = 0;
    std::chrono::steady_clock::time_point enqueue_time;
    std::optional<std::chrono::steady_clock::time_point> deadline; // job's absolute deadline, if any
    std::vector<NamedTensor> inputs;
    std::shared_ptr<ModelVariant> variant;
    CancellationToken* cancel = nullptr; // non-owning; owned by the job, see JobState::cancel_token
    std::function<void(std::vector<NamedTensor>, double)> on_done;
    std::function<void(std::exception_ptr)> on_error;
};

} // namespace

// ---------------------------------------------------------------------------
// BackendLane — one simulated backend's worker pool + per-(node,precision,
// shape) batching queues. See docs/ARCHITECTURE.md "Scheduler" for the
// design and the deliberate simplicity tradeoffs called out inline below.
// ---------------------------------------------------------------------------
class BackendLane {
public:
    BackendLane(Backend backend, int n_threads, SchedulerConfig cfg, MetricsRegistry& metrics, Logger& logger)
        : backend_(backend), cfg_(cfg), metrics_(metrics), logger_(logger) {
        n_threads = std::max(1, n_threads);
        threads_.reserve(static_cast<size_t>(n_threads));
        for (int i = 0; i < n_threads; ++i) threads_.emplace_back([this] { workerLoop(); });
    }

    ~BackendLane() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto& t : threads_)
            if (t.joinable()) t.join();
    }

    void submit(NodeTask task, size_t max_batch, int window_ms) {
        PendingKey key{task.node_id, task.precision, shapeSignature(task.inputs)};
        std::lock_guard<std::mutex> lock(mutex_);
        auto& dq = pending_[key];
        if (dq.empty()) first_enqueue_[key] = task.enqueue_time;
        max_batch_by_key_[key] = std::max<size_t>(1, max_batch);
        window_by_key_[key] = window_ms;
        dq.push_back(std::move(task));
        metrics_.setQueueDepth(laneKey(), totalQueuedLocked());
        if (!in_ready_.count(key)) {
            in_ready_.insert(key);
            ready_.push_back(key);
        }
        cv_.notify_all();
    }

    std::string laneKey() const { return backend_ == Backend::CPU ? "CPU" : "GPU_SIM"; }

private:
    // Caller holds mutex_. Scores a ready key for "which one flushes next
    // when more than one is eligible": by default the existing aged
    // priority (base priority + time-waited / aging_ms_per_priority_point,
    // a simple starvation-avoidance rule), or — when
    // SchedulerConfig::use_edf_scoring is on — by earliest-deadline-first
    // for any key that has at least one deadline-bearing task queued,
    // which always outranks a key with none. See scheduler.h for why EDF
    // is scored per-*key* (the earliest deadline among that key's queued
    // tasks) rather than per-task: a batch's tasks all flush together, so
    // the whole batch is only as urgent as its most urgent member.
    double scoreKey(const PendingKey& key, std::chrono::steady_clock::time_point now) const {
        const auto& dq = pending_.at(key);
        if (cfg_.use_edf_scoring) {
            bool have_deadline = false;
            std::chrono::steady_clock::time_point earliest{};
            for (const auto& t : dq) {
                if (!t.deadline.has_value()) continue;
                if (!have_deadline || *t.deadline < earliest) {
                    earliest = *t.deadline;
                    have_deadline = true;
                }
            }
            if (have_deadline) {
                double remaining_ms = std::chrono::duration<double, std::milli>(earliest - now).count();
                // Offset so any EDF-scored key always outranks a
                // non-deadline key: aged-priority scores climb by
                // waited_ms/aging_ms_per_priority_point and are bounded in
                // practice by realistic queueing delays, nowhere near 1e9.
                return 1e9 - remaining_ms;
            }
        }
        int base = dq.front().priority;
        double waited_ms = std::chrono::duration<double, std::milli>(now - first_enqueue_.at(key)).count();
        return static_cast<double>(base) + waited_ms / cfg_.aging_ms_per_priority_point;
    }

    size_t totalQueuedLocked() const {
        size_t total = 0;
        for (const auto& [k, dq] : pending_) total += dq.size();
        return total;
    }

    void workerLoop() {
        using clock = std::chrono::steady_clock;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return stopping_ || !ready_.empty(); });
            if (stopping_ && ready_.empty()) return;

            auto now = clock::now();

            // A linear scan over ready keys is intentional: at DAG sizes
            // in the tens of distinct (node, precision, shape) triples
            // this is simpler than an indexed heap and costs nothing
            // measurable. For each ready key, decide whether it's
            // flushable *right now* (full, or its batch window has
            // elapsed, or batching is disabled for it) and, among the
            // flushable ones, which scores highest.
            std::optional<size_t> best_i;
            double best_score = -1e300;
            clock::time_point earliest_wake = clock::time_point::max();

            for (size_t i = 0; i < ready_.size(); ++i) {
                const PendingKey& key = ready_[i];
                auto& dq = pending_.at(key);
                size_t max_batch = max_batch_by_key_.at(key);
                int window_ms = window_by_key_.at(key);
                auto first_time = first_enqueue_.at(key);
                double elapsed_ms = std::chrono::duration<double, std::milli>(now - first_time).count();
                bool flushable = dq.size() >= max_batch || window_ms <= 0 || elapsed_ms >= static_cast<double>(window_ms);
                if (flushable) {
                    double score = scoreKey(key, now);
                    if (score > best_score) {
                        best_score = score;
                        best_i = i;
                    }
                } else {
                    auto wake_at = first_time + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                                     std::chrono::duration<double, std::milli>(window_ms));
                    earliest_wake = std::min(earliest_wake, wake_at);
                }
            }

            if (!best_i.has_value()) {
                // Nothing is flushable yet: every ready key is still
                // inside its batch window. Wait precisely until the
                // earliest one closes, or until new work changes the
                // picture (submit() notifies this condvar), instead of
                // the earlier design's busy-poll (release the lock, sleep
                // 1-2ms, re-check) — which spent CPU on every lane worker
                // continuously while a batch window was open, and
                // quantized the window's real accuracy to that sleep
                // granularity. This isn't a full timing wheel (that would
                // matter once a lane manages hundreds of independently
                // timed keys; at demo scale it's the same "deliberately
                // simple, stated explicitly" tradeoff as the linear scan
                // above), just an exact wait instead of a polled one.
                cv_.wait_until(lock, earliest_wake);
                continue;
            }

            size_t idx = *best_i;
            PendingKey key = ready_[idx];
            auto& dq = pending_.at(key);
            size_t max_batch = max_batch_by_key_.at(key);

            // Flush now: remove the key from the ready set and take up to
            // max_batch queued tasks.
            in_ready_.erase(key);
            ready_.erase(ready_.begin() + static_cast<long>(idx));
            std::vector<NodeTask> batch;
            size_t take = std::min(max_batch, dq.size());
            batch.reserve(take);
            for (size_t i = 0; i < take; ++i) {
                batch.push_back(std::move(dq.front()));
                dq.pop_front();
            }
            if (!dq.empty()) {
                first_enqueue_[key] = dq.front().enqueue_time;
                in_ready_.insert(key);
                ready_.push_back(key);
            } else {
                first_enqueue_.erase(key);
            }
            metrics_.setQueueDepth(laneKey(), totalQueuedLocked());
            lock.unlock();

            executeBatch(key, std::move(batch));
        }
    }

    void executeBatch(const PendingKey& key, std::vector<NodeTask> batch) {
        using clock = std::chrono::steady_clock;
        auto t0 = clock::now();

        std::vector<NamedTensor> merged;
        std::exception_ptr err;
        std::vector<NamedTensor> outputs;
        try {
            size_t n_inputs = batch.front().inputs.size();
            merged.resize(n_inputs);
            for (size_t j = 0; j < n_inputs; ++j) {
                std::vector<NamedTensor> parts;
                parts.reserve(batch.size());
                for (auto& task : batch) parts.push_back(task.inputs[j]);
                merged[j] = batch.size() == 1 ? std::move(parts[0]) : concatBatch(parts);
            }
            // A batch never mixes cancellation tokens across jobs, so
            // binding to any one member's token (they may differ; only
            // the first is used) is only meaningful for batch_size==1 —
            // the common case for a tight per-job deadline anyway, since
            // a request under deadline pressure is unlikely to still be
            // waiting around in a multi-item batch window.
            CancellationToken* cancel = batch.size() == 1 ? batch.front().cancel : nullptr;
            outputs = batch.front().variant->run(merged, cancel);

            if (backend_ == Backend::GPU_SIM) {
                // Simulated device dispatch: a fixed launch/sync overhead
                // plus a bandwidth-modeled transfer cost. No physical GPU is
                // used anywhere in Loomcore; this exists purely so the
                // scheduler and router have two backends with genuinely
                // different cost profiles to route across, matching the
                // project's "simulate heterogeneous backends" requirement.
                double bytes = approxBytes(merged) + approxBytes(outputs);
                double overhead_ms = cfg_.gpu_sim_fixed_overhead_ms + bytes / cfg_.gpu_sim_bytes_per_ms;
                std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(overhead_ms));
            }
        } catch (...) {
            err = std::current_exception();
        }

        double total_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();

        LogEvent flush;
        flush.type = LogEventType::BatchFlushed;
        flush.job_id = batch.size() == 1 ? batch.front().job_id : "(" + std::to_string(batch.size()) + " jobs)";
        flush.node_id = key.node_id;
        flush.backend = backend_;
        flush.precision = key.precision;
        flush.batch_size = batch.size();
        flush.latency_ms = total_ms;
        logger_.log(flush);

        for (size_t i = 0; i < batch.size(); ++i) {
            if (err) {
                batch[i].on_error(err);
                continue;
            }
            std::vector<NamedTensor> item_outputs;
            item_outputs.reserve(outputs.size());
            for (const auto& out : outputs) {
                item_outputs.push_back(batch.size() == 1 ? out : out.sliceBatch(static_cast<int64_t>(i),
                                                                                  static_cast<int64_t>(i) + 1));
            }
            batch[i].on_done(std::move(item_outputs), total_ms);
        }
    }

    Backend backend_;
    SchedulerConfig cfg_;
    MetricsRegistry& metrics_;
    Logger& logger_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool stopping_ = false;
    std::map<PendingKey, std::deque<NodeTask>> pending_;
    std::map<PendingKey, size_t> max_batch_by_key_;
    std::map<PendingKey, int> window_by_key_;
    std::map<PendingKey, std::chrono::steady_clock::time_point> first_enqueue_;
    std::set<PendingKey> in_ready_;
    std::vector<PendingKey> ready_;
    std::vector<std::thread> threads_;
};

// ---------------------------------------------------------------------------
// JobState
// ---------------------------------------------------------------------------
struct JobState {
    std::string job_id;
    TensorMap external_inputs;
    int priority = 0;
    double time_budget_ms = -1.0;
    std::chrono::steady_clock::time_point start_time;
    std::vector<std::string> sinks;

    std::mutex mutex; // guards node_outputs only (remaining_deps/remaining_nodes are lock-free)
    // Values are shared, not deep-copied, across every node dispatch that
    // snapshots this map (see Scheduler::Impl::dispatchNode) — a node with
    // many dependents no longer costs one full tensor-data copy per
    // dependent dispatch, just a shared_ptr copy. See
    // NodeExecutionContext::upstream_outputs in loomcore/model_node.h.
    std::map<std::string, std::shared_ptr<const std::vector<NamedTensor>>> node_outputs;
    std::map<std::string, std::atomic<int>> remaining_deps; // fixed key set after construction
    std::atomic<int> remaining_nodes{0};

    // `failed` is a fast, best-effort, possibly-set-more-than-once hint
    // ("stop doing further work for this job") read by dispatchNode/
    // onNodeFinished; it is NOT what decides who gets to touch `promise`.
    // `settled` is that arbiter: exactly one of the success path
    // (onNodeFinished, when remaining_nodes hits zero) and the failure
    // path (failJob, from any node or the deadline reaper) may win its
    // compare_exchange and call promise.set_value/set_exception — see
    // docs/ARCHITECTURE.md "Scheduler" for the race this closes (a
    // std::future is only allowed to be settled once; a second attempt
    // throws std::future_error, uncaught on a lane worker thread, which
    // is a process-terminating bug, not a recoverable one).
    std::atomic<bool> failed{false};
    std::atomic<bool> settled{false};
    std::promise<JobResult> promise;

    // Only constructed when SchedulerConfig::enable_deadline_cancellation
    // is on and this job has a time budget; shared by every node task this
    // job dispatches so the deadline reaper can abort whichever one
    // happens to be inside ModelVariant::run() when the budget expires.
    std::unique_ptr<CancellationToken> cancel_token;

    // Only populated when enable_admission_control or
    // enable_precision_planning is on and this job has a time budget with
    // enough existing metrics history to plan from (see
    // PrecisionPlan::has_estimate). Computed once at submission, never
    // mutated afterwards, so a stable pointer into it may be handed out
    // via RoutingContext::precision_plan for this job's whole lifetime.
    std::optional<PrecisionPlan> precision_plan;

    // Keeps whatever Runtime-owned GraphSnapshot this job's Graph/nodes
    // came from alive for exactly as long as this job is in flight — see
    // Scheduler::submitJob's `keep_alive` parameter and
    // docs/ARCHITECTURE.md "Hot-reloading a graph".
    std::shared_ptr<void> keep_alive;

    double remainingBudgetMs() const {
        if (time_budget_ms < 0.0) return -1.0;
        double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
        return time_budget_ms - elapsed;
    }
};

// ---------------------------------------------------------------------------
// Scheduler::Impl
// ---------------------------------------------------------------------------
struct Scheduler::Impl {
    const Graph& graph;
    const std::map<std::string, ModelNode>& nodes;
    MetricsRegistry& metrics;
    std::shared_ptr<IRoutingPolicy> router;
    SchedulerConfig cfg;
    Logger& logger;

    BackendLane cpu_lane;
    BackendLane gpu_lane;

    std::atomic<bool> shutting_down{false};

    // --- Deadline reaper (only running when cfg.enable_deadline_cancellation) ---
    std::mutex reaper_mutex;
    std::condition_variable reaper_cv;
    bool reaper_stop = false;
    std::vector<std::weak_ptr<JobState>> reaper_watch_list;
    std::thread reaper_thread;

    Impl(const Graph& g, const std::map<std::string, ModelNode>& n, MetricsRegistry& m,
         std::shared_ptr<IRoutingPolicy> r, SchedulerConfig c, Logger& l)
        : graph(g),
          nodes(n),
          metrics(m),
          router(std::move(r)),
          cfg(c),
          logger(l),
          cpu_lane(Backend::CPU,
                   cfg.cpu_threads > 0 ? cfg.cpu_threads
                                       : static_cast<int>(std::max(1u, std::thread::hardware_concurrency() / 2)),
                   cfg, metrics, logger),
          gpu_lane(Backend::GPU_SIM, cfg.gpu_sim_threads, cfg, metrics, logger) {
        if (cfg.enable_deadline_cancellation) {
            reaper_thread = std::thread([this] { reaperLoop(); });
        }
    }

    ~Impl() {
        if (reaper_thread.joinable()) {
            {
                std::lock_guard<std::mutex> lock(reaper_mutex);
                reaper_stop = true;
            }
            reaper_cv.notify_all();
            reaper_thread.join();
        }
    }

    BackendLane& laneFor(Backend b) { return b == Backend::CPU ? cpu_lane : gpu_lane; }

    void reaperLoop() {
        std::unique_lock<std::mutex> lock(reaper_mutex);
        while (!reaper_stop) {
            reaper_cv.wait_for(lock, std::chrono::duration<double, std::milli>(cfg.deadline_reaper_poll_ms),
                                [&] { return reaper_stop; });
            if (reaper_stop) return;

            std::vector<std::shared_ptr<JobState>> to_check;
            for (auto it = reaper_watch_list.begin(); it != reaper_watch_list.end();) {
                auto sp = it->lock();
                if (!sp || sp->settled.load()) {
                    it = reaper_watch_list.erase(it);
                    continue;
                }
                to_check.push_back(std::move(sp));
                ++it;
            }
            lock.unlock();

            for (auto& job : to_check) {
                if (job->remainingBudgetMs() <= 0.0) {
                    if (job->cancel_token) job->cancel_token->requestCancel();
                    failJob(job, std::make_exception_ptr(DeadlineExceededError(
                                     "job '" + job->job_id + "' exceeded its time budget of " +
                                     std::to_string(job->time_budget_ms) + "ms")));
                }
            }
            lock.lock();
        }
    }

    void dispatchNode(std::shared_ptr<JobState> job, const std::string& node_id) {
        if (job->failed.load()) return;
        try {
            const NodeConfig& cfg_node = graph.node(node_id);
            const ModelNode& mnode = nodes.at(node_id);

            RoutingContext ctx;
            ctx.job_id = job->job_id;
            ctx.node = &cfg_node;
            ctx.metrics = &metrics;
            ctx.cpu_queue_depth = metrics.queueDepth("CPU");
            ctx.gpu_queue_depth = metrics.queueDepth("GPU_SIM");
            ctx.time_budget_remaining_ms = job->remainingBudgetMs();
            ctx.precision_plan = job->precision_plan.has_value() ? &*job->precision_plan : nullptr;

            std::vector<NamedTensor> confidence_source_storage;
            if (cfg_node.confidence && !cfg_node.confidence_source_node.empty()) {
                std::lock_guard<std::mutex> lock(job->mutex);
                auto it = job->node_outputs.find(cfg_node.confidence_source_node);
                if (it != job->node_outputs.end() && it->second) {
                    confidence_source_storage = *it->second;
                    ctx.upstream_confidence_source = &confidence_source_storage;
                }
            }

            std::optional<RoutingDecision> decision = router ? router->decide(ctx) : std::nullopt;
            if (decision.has_value() &&
                (decision->skip || decision->precision.has_value() || decision->backend.has_value())) {
                LogEvent ev;
                ev.type = LogEventType::RoutingDecision;
                ev.job_id = job->job_id;
                ev.node_id = node_id;
                ev.message = decision->policy_name + ": " + decision->reason;
                logger.log(ev);
            }

            if (decision.has_value() && decision->skip) {
                LogEvent ev;
                ev.type = LogEventType::NodeSkipped;
                ev.job_id = job->job_id;
                ev.node_id = node_id;
                logger.log(ev);
                onNodeFinished(job, node_id, {});
                return;
            }

            Precision precision = (decision && decision->precision) ? *decision->precision
                                   : cfg_node.hasVariant(Precision::FP32) ? Precision::FP32
                                                                          : cfg_node.variants.front().precision;
            Backend backend = (decision && decision->backend) ? *decision->backend : cfg_node.backend;

            std::shared_ptr<ModelVariant> variant = mnode.variant(precision);
            if (!variant) {
                variant = mnode.variants.begin()->second;
                precision = variant->precision();
            }

            NodeExecutionContext exec_ctx;
            exec_ctx.job_id = job->job_id;
            exec_ctx.graph_inputs = &job->external_inputs;
            std::map<std::string, std::shared_ptr<const std::vector<NamedTensor>>> upstream_copy;
            {
                std::lock_guard<std::mutex> lock(job->mutex);
                upstream_copy = job->node_outputs; // cheap: copies shared_ptrs, not tensor data
            }
            exec_ctx.upstream_outputs = &upstream_copy;

            std::vector<NamedTensor> inputs = cfg_node.binder(exec_ctx);
            inputs = reorderToExpected(std::move(inputs), variant->inputNames());
            if (inputs.size() != variant->inputNames().size()) {
                throw LoomcoreError("node '" + node_id + "': binder produced " + std::to_string(inputs.size()) +
                                     " tensors but the model expects " + std::to_string(variant->inputNames().size()));
            }

            metrics.incrementInFlight(node_id);

            NodeTask task;
            task.job_id = job->job_id;
            task.node_id = node_id;
            task.precision = precision;
            task.priority = job->priority;
            task.enqueue_time = std::chrono::steady_clock::now();
            if (job->time_budget_ms >= 0.0) {
                task.deadline = job->start_time + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                                       std::chrono::duration<double, std::milli>(job->time_budget_ms));
            }
            task.inputs = std::move(inputs);
            task.variant = variant;
            task.cancel = job->cancel_token.get();
            task.on_done = [this, job, node_id, backend, precision](std::vector<NamedTensor> outputs, double latency_ms) {
                metrics.recordLatency(node_id, latency_ms);
                metrics.recordLatency(precisionMetricsKey(node_id, precision), latency_ms);
                metrics.recordOutcome(node_id, true);
                metrics.decrementInFlight(node_id);
                LogEvent ev;
                ev.type = LogEventType::NodeCompleted;
                ev.job_id = job->job_id;
                ev.node_id = node_id;
                ev.backend = backend;
                ev.precision = precision;
                ev.latency_ms = latency_ms;
                logger.log(ev);
                onNodeFinished(job, node_id, std::move(outputs));
            };
            task.on_error = [this, job, node_id](std::exception_ptr e) {
                metrics.recordOutcome(node_id, false);
                metrics.decrementInFlight(node_id);
                failJob(job, e);
            };

            LogEvent scheduled;
            scheduled.type = LogEventType::NodeScheduled;
            scheduled.job_id = job->job_id;
            scheduled.node_id = node_id;
            scheduled.backend = backend;
            scheduled.precision = precision;
            logger.log(scheduled);

            laneFor(backend).submit(std::move(task), cfg_node.max_batch_size, cfg_node.batch_window_ms);
        } catch (...) {
            failJob(job, std::current_exception());
        }
    }

    void onNodeFinished(std::shared_ptr<JobState> job, const std::string& node_id, std::vector<NamedTensor> outputs) {
        if (job->failed.load()) return;
        {
            std::lock_guard<std::mutex> lock(job->mutex);
            job->node_outputs[node_id] = std::make_shared<const std::vector<NamedTensor>>(std::move(outputs));
        }
        for (const auto& dependent : graph.dependents(node_id)) {
            if (--job->remaining_deps.at(dependent) == 0) dispatchNode(job, dependent);
        }
        if (--job->remaining_nodes == 0) {
            // See JobState::settled: this is the "success" side of the
            // single arbitration point deciding who gets to touch
            // job->promise. If a concurrent failJob() (from a sibling
            // node, or the deadline reaper) already won it, back out
            // without calling set_value — the promise is already settled
            // with an exception, and a second settle attempt is undefined
            // behavior (std::future_error, uncaught here since this runs
            // on a lane worker thread with no caller frame to catch it).
            bool expected = false;
            if (!job->settled.compare_exchange_strong(expected, true)) return;
            JobResult result;
            {
                std::lock_guard<std::mutex> lock(job->mutex);
                for (const auto& sink : job->sinks) {
                    auto it = job->node_outputs.find(sink);
                    if (it != job->node_outputs.end() && it->second) result[sink] = *it->second;
                }
            }
            LogEvent ev;
            ev.type = LogEventType::JobCompleted;
            ev.job_id = job->job_id;
            logger.log(ev);
            job->promise.set_value(std::move(result));
        }
    }

    void failJob(std::shared_ptr<JobState> job, std::exception_ptr e) {
        // Set unconditionally (harmless if called more than once): every
        // dispatchNode/onNodeFinished call for this job checks this first
        // to stop doing further work as soon as any failure is known,
        // regardless of which failure — if any — ultimately wins the
        // settle race below.
        job->failed.store(true);

        bool expected = false;
        if (!job->settled.compare_exchange_strong(expected, true)) return; // already settled (success or another failure)
        LogEvent ev;
        ev.type = LogEventType::Error;
        ev.job_id = job->job_id;
        try {
            std::rethrow_exception(e);
        } catch (const std::exception& ex) {
            ev.message = ex.what();
        } catch (...) {
            ev.message = "unknown error";
        }
        logger.log(ev);
        job->promise.set_exception(e);
    }
};

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------
namespace {
const Graph& validated(const Graph& g) {
    g.validate(); // throws GraphError before any lane thread is spun up
    return g;
}

std::future<JobResult> alreadyFailedFuture(std::exception_ptr e) {
    std::promise<JobResult> p;
    p.set_exception(std::move(e));
    return p.get_future();
}
} // namespace

Scheduler::Scheduler(const Graph& graph, const std::map<std::string, ModelNode>& nodes, MetricsRegistry& metrics,
                      std::shared_ptr<IRoutingPolicy> router, SchedulerConfig cfg, Logger* logger)
    : impl_(std::make_unique<Impl>(validated(graph), nodes, metrics, std::move(router), cfg,
                                    logger ? *logger : Logger::instance())) {}

Scheduler::~Scheduler() { shutdown(); }

void Scheduler::shutdown() { impl_->shutting_down.store(true); }

std::future<JobResult> Scheduler::submitJob(TensorMap external_inputs, int priority, double time_budget_ms,
                                             std::shared_ptr<void> keep_alive) {
    if (impl_->shutting_down.load()) {
        return alreadyFailedFuture(
            std::make_exception_ptr(LoomcoreError("Scheduler::submitJob: scheduler is shutting down")));
    }

    std::optional<PrecisionPlan> plan;
    if (time_budget_ms >= 0.0 && (impl_->cfg.enable_admission_control || impl_->cfg.enable_precision_planning)) {
        plan = planPrecisionForBudget(impl_->graph, impl_->nodes, impl_->metrics, time_budget_ms);
        if (impl_->cfg.enable_admission_control && plan->has_estimate && !plan->feasible) {
            LogEvent ev;
            ev.type = LogEventType::JobRejected;
            ev.message = "PrecisionPlanner: critical path " + std::to_string(plan->critical_path_fp32_ms) +
                         "ms (best achievable " + std::to_string(plan->critical_path_planned_ms) +
                         "ms) exceeds budget " + std::to_string(time_budget_ms) + "ms even after downgrades";
            impl_->logger.log(ev);
            return alreadyFailedFuture(std::make_exception_ptr(JobRejectedError(
                "job rejected by admission control: " + ev.message)));
        }
    }

    auto job = std::make_shared<JobState>();
    job->job_id = newJobId();
    job->external_inputs = std::move(external_inputs);
    job->priority = priority;
    job->time_budget_ms = time_budget_ms;
    job->start_time = std::chrono::steady_clock::now();
    job->sinks = impl_->graph.sinkNodes();
    job->remaining_nodes = static_cast<int>(impl_->graph.nodes().size());
    job->precision_plan = std::move(plan);
    job->keep_alive = std::move(keep_alive);
    for (const auto& n : impl_->graph.nodes()) {
        job->remaining_deps.try_emplace(n.id, static_cast<int>(n.depends_on.size()));
    }

    if (impl_->cfg.enable_deadline_cancellation && time_budget_ms >= 0.0) {
        job->cancel_token = std::make_unique<CancellationToken>();
        std::lock_guard<std::mutex> lock(impl_->reaper_mutex);
        impl_->reaper_watch_list.push_back(job);
    }

    LogEvent submitted;
    submitted.type = LogEventType::JobSubmitted;
    submitted.job_id = job->job_id;
    impl_->logger.log(submitted);

    auto future = job->promise.get_future();
    std::vector<std::string> roots;
    for (const auto& n : impl_->graph.nodes()) {
        if (n.depends_on.empty()) roots.push_back(n.id);
    }
    for (const auto& id : roots) impl_->dispatchNode(job, id);
    return future;
}

JobResult Scheduler::runSync(TensorMap external_inputs, int priority, double time_budget_ms,
                              std::shared_ptr<void> keep_alive) {
    return submitJob(std::move(external_inputs), priority, time_budget_ms, std::move(keep_alive)).get();
}

} // namespace loomcore
