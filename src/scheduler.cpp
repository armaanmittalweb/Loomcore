#include "loomcore/scheduler.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
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

struct PendingKey {
    std::string node_id;
    Precision precision;
    bool operator<(const PendingKey& o) const {
        if (node_id != o.node_id) return node_id < o.node_id;
        return precision < o.precision;
    }
};

struct NodeTask {
    std::string job_id;
    std::string node_id;
    Precision precision;
    int priority = 0;
    std::chrono::steady_clock::time_point enqueue_time;
    std::vector<NamedTensor> inputs;
    std::shared_ptr<ModelVariant> variant;
    std::function<void(std::vector<NamedTensor>, double)> on_done;
    std::function<void(std::exception_ptr)> on_error;
};

} // namespace

// ---------------------------------------------------------------------------
// BackendLane — one simulated backend's worker pool + per-(node,precision)
// batching queues. See docs/ARCHITECTURE.md "Scheduler" for the design and
// the deliberate simplicity tradeoffs (linear scan over ready keys, a
// busy-poll batch window) called out inline below.
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
        PendingKey key{task.node_id, task.precision};
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
    double effectivePriority(const PendingKey& key, std::chrono::steady_clock::time_point now) const {
        // Caller holds mutex_.
        const auto& dq = pending_.at(key);
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

            // Pick the ready key with the highest effective (aged) priority.
            // A linear scan is intentional: at DAG sizes in the tens of
            // distinct (node, precision) pairs this is simpler than an
            // indexed heap and costs nothing measurable.
            auto now = clock::now();
            size_t best_i = 0;
            double best_score = -1e300;
            for (size_t i = 0; i < ready_.size(); ++i) {
                double score = effectivePriority(ready_[i], now);
                if (score > best_score) {
                    best_score = score;
                    best_i = i;
                }
            }
            PendingKey key = ready_[best_i];
            auto& dq = pending_.at(key);
            size_t max_batch = max_batch_by_key_.at(key);
            int window_ms = window_by_key_.at(key);
            auto elapsed = std::chrono::duration<double, std::milli>(now - first_enqueue_.at(key)).count();

            if (dq.size() < max_batch && window_ms > 0 && elapsed < static_cast<double>(window_ms)) {
                // Still within the dynamic-batching window and not yet full:
                // release the lock and sleep a short slice so other worker
                // threads in this lane can keep servicing other ready keys,
                // then loop back and re-evaluate. This is a deliberately
                // simple busy-poll rather than a per-key timer.
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(std::min(2, std::max(1, window_ms))));
                continue;
            }

            // Flush now: remove the key from the ready set and take up to
            // max_batch queued tasks.
            in_ready_.erase(key);
            ready_.erase(ready_.begin() + static_cast<long>(best_i));
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
            outputs = batch.front().variant->run(merged);

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
    std::map<std::string, std::vector<NamedTensor>> node_outputs;
    std::map<std::string, std::atomic<int>> remaining_deps; // fixed key set after construction
    std::atomic<int> remaining_nodes{0};
    std::atomic<bool> failed{false};
    std::promise<JobResult> promise;

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
          gpu_lane(Backend::GPU_SIM, cfg.gpu_sim_threads, cfg, metrics, logger) {}

    BackendLane& laneFor(Backend b) { return b == Backend::CPU ? cpu_lane : gpu_lane; }

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

            std::vector<NamedTensor> confidence_source;
            if (cfg_node.confidence && !cfg_node.confidence_source_node.empty()) {
                std::lock_guard<std::mutex> lock(job->mutex);
                auto it = job->node_outputs.find(cfg_node.confidence_source_node);
                if (it != job->node_outputs.end()) {
                    confidence_source = it->second;
                    ctx.upstream_confidence_source = &confidence_source;
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
            std::map<std::string, std::vector<NamedTensor>> upstream_copy;
            {
                std::lock_guard<std::mutex> lock(job->mutex);
                upstream_copy = job->node_outputs;
            }
            exec_ctx.upstream_outputs = &upstream_copy;

            std::vector<NamedTensor> inputs = cfg_node.binder(exec_ctx);
            inputs = reorderToExpected(std::move(inputs), variant->inputNames());
            if (inputs.size() != variant->inputNames().size()) {
                throw LoomcoreError("node '" + node_id + "': binder produced " + std::to_string(inputs.size()) +
                                     " tensors but the model expects " + std::to_string(variant->inputNames().size()));
            }

            NodeTask task;
            task.job_id = job->job_id;
            task.node_id = node_id;
            task.precision = precision;
            task.priority = job->priority;
            task.enqueue_time = std::chrono::steady_clock::now();
            task.inputs = std::move(inputs);
            task.variant = variant;
            task.on_done = [this, job, node_id, backend, precision](std::vector<NamedTensor> outputs, double latency_ms) {
                metrics.recordLatency(node_id, latency_ms);
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
            task.on_error = [this, job](std::exception_ptr e) { failJob(job, e); };

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
            job->node_outputs[node_id] = std::move(outputs);
        }
        for (const auto& dependent : graph.dependents(node_id)) {
            if (--job->remaining_deps.at(dependent) == 0) dispatchNode(job, dependent);
        }
        if (--job->remaining_nodes == 0) {
            JobResult result;
            {
                std::lock_guard<std::mutex> lock(job->mutex);
                for (const auto& sink : job->sinks) {
                    auto it = job->node_outputs.find(sink);
                    if (it != job->node_outputs.end()) result[sink] = it->second;
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
        bool expected = false;
        if (!job->failed.compare_exchange_strong(expected, true)) return; // already failed
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
} // namespace

Scheduler::Scheduler(const Graph& graph, const std::map<std::string, ModelNode>& nodes, MetricsRegistry& metrics,
                      std::shared_ptr<IRoutingPolicy> router, SchedulerConfig cfg, Logger* logger)
    : impl_(std::make_unique<Impl>(validated(graph), nodes, metrics, std::move(router), cfg,
                                    logger ? *logger : Logger::instance())) {}

Scheduler::~Scheduler() { shutdown(); }

void Scheduler::shutdown() {
    // BackendLane's own destructor drains/stops its threads; nothing else
    // to do here. Kept as an explicit method (rather than destructor-only)
    // so callers can shut down deterministically before other teardown.
}

std::future<JobResult> Scheduler::submitJob(TensorMap external_inputs, int priority, double time_budget_ms) {
    auto job = std::make_shared<JobState>();
    job->job_id = newJobId();
    job->external_inputs = std::move(external_inputs);
    job->priority = priority;
    job->time_budget_ms = time_budget_ms;
    job->start_time = std::chrono::steady_clock::now();
    job->sinks = impl_->graph.sinkNodes();
    job->remaining_nodes = static_cast<int>(impl_->graph.nodes().size());
    for (const auto& n : impl_->graph.nodes()) {
        job->remaining_deps.try_emplace(n.id, static_cast<int>(n.depends_on.size()));
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

JobResult Scheduler::runSync(TensorMap external_inputs, int priority, double time_budget_ms) {
    return submitJob(std::move(external_inputs), priority, time_budget_ms).get();
}

} // namespace loomcore
