#include "loomcore/runtime.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>

#include <nlohmann/json.hpp>

namespace loomcore {

namespace {

Backend parseBackend(const std::string& s) {
    if (s == "CPU") return Backend::CPU;
    if (s == "GPU_SIM") return Backend::GPU_SIM;
    throw LoomcoreError("unknown backend '" + s + "' (expected \"CPU\" or \"GPU_SIM\")");
}

Precision parsePrecision(const std::string& s) {
    if (s == "FP32") return Precision::FP32;
    if (s == "INT8") return Precision::INT8;
    throw LoomcoreError("unknown precision '" + s + "' (expected \"FP32\" or \"INT8\")");
}

} // namespace

// One immutable-after-construction bundle of everything a running graph
// needs: the topology, the loaded model variants, the router, and the
// Scheduler built to reference *this specific* graph/nodes pair. Runtime
// hands out shared_ptr<GraphSnapshot> (treated as read-only after
// construction) instead of owning a single mutable Graph/nodes/Scheduler
// triple in place, so reloadGraph() can build a whole new one and swap it
// in atomically. See docs/ARCHITECTURE.md "Hot-reloading a graph" for why
// the earlier design — reassigning impl_->graph/impl_->nodes in place
// while the previous Scheduler's lane threads still held references to
// them — was a use-after-free the moment loadGraph() was called a second
// time on a Runtime already handling traffic.
struct GraphSnapshot {
    Graph graph;
    std::map<std::string, ModelNode> nodes;
    std::shared_ptr<IRoutingPolicy> router;
    std::shared_ptr<Scheduler> scheduler;
};

struct Runtime::Impl {
    // Persists across reloads (deliberately outside GraphSnapshot): a node
    // id's latency/outcome history carries over when the same id exists in
    // the new graph, rather than resetting to cold on every reload.
    MetricsRegistry metrics;

    // Readers (submit/run/graph) take a lock-free atomic snapshot of
    // `current` and never block on `swap_mutex` — see currentSnapshot().
    // `swap_mutex` only serializes concurrent writers (loadGraph /
    // reloadGraph) against each other, so two reloads racing on different
    // threads can't interleave their JSON-parsing/session-loading/logger
    // reconfiguration side effects.
    std::mutex swap_mutex;
    std::shared_ptr<GraphSnapshot> current;

    std::shared_ptr<GraphSnapshot> currentSnapshot() const { return std::atomic_load(&current); }

    // --- Deferred snapshot teardown ("the graveyard") ---
    //
    // A GraphSnapshot's last reference is *not* safe to drop wherever
    // shared_ptr refcounting happens to drop it: JobState::keep_alive
    // (see Scheduler::submitJob) pins the snapshot alive for exactly as
    // long as a job runs, and a job's *last* completion callback runs on
    // one of its own Scheduler's lane worker threads — set up exactly
    // that way so the graph a job started against never disappears out
    // from under it mid-flight. But if that same callback happens to drop
    // the snapshot's last reference (because a reload has already
    // replaced impl_->current and this was the last job still pinning the
    // old one), destroying the GraphSnapshot destroys its Scheduler,
    // whose destructor joins its own BackendLane worker threads —
    // including the very thread this callback is running on. Joining a
    // std::thread from itself is undefined behavior (in practice: an
    // uncaught std::system_error on a thread with no caller frame to
    // catch it, i.e. std::terminate).
    //
    // The fix: JobState never receives the real shared_ptr<GraphSnapshot>
    // directly. It receives a proxy shared_ptr<void> (built by
    // deferredKeepAlive() below) whose custom deleter — which *does* run
    // on whatever thread drops the last reference, lane thread included —
    // does nothing more expensive than pushing the real shared_ptr onto
    // this queue. The actual destruction happens later, when the
    // dedicated graveyard thread drains the queue on its own stack,
    // nowhere near any lane thread.
    std::mutex graveyard_mutex;
    std::condition_variable graveyard_cv;
    std::deque<std::shared_ptr<GraphSnapshot>> graveyard;
    bool graveyard_stop = false;
    std::thread graveyard_thread;

    Impl() : graveyard_thread([this] { graveyardLoop(); }) {}

    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(graveyard_mutex);
            graveyard_stop = true;
        }
        graveyard_cv.notify_all();
        if (graveyard_thread.joinable()) graveyard_thread.join();
    }

    void graveyardLoop() {
        std::unique_lock<std::mutex> lock(graveyard_mutex);
        while (true) {
            graveyard_cv.wait(lock, [&] { return graveyard_stop || !graveyard.empty(); });
            if (graveyard.empty() && graveyard_stop) return;
            std::deque<std::shared_ptr<GraphSnapshot>> batch;
            batch.swap(graveyard);
            lock.unlock();
            batch.clear(); // the actual ~GraphSnapshot()/~Scheduler() calls happen here
            lock.lock();
        }
    }

    std::shared_ptr<void> deferredKeepAlive(std::shared_ptr<GraphSnapshot> snap) {
        GraphSnapshot* raw = snap.get();
        return std::shared_ptr<void>(raw, [this, snap](void*) mutable {
            std::lock_guard<std::mutex> lock(graveyard_mutex);
            graveyard.push_back(std::move(snap));
            graveyard_cv.notify_one();
        });
    }
};

Runtime::Runtime() : impl_(std::make_unique<Impl>()) {
    (void)Environment::shared(); // force initialization up front, at a predictable point
}

Runtime::~Runtime() = default;

namespace {

std::shared_ptr<GraphSnapshot> buildSnapshot(const std::string& config_path,
                                              std::map<std::string, NodeInputBinder> binders,
                                              std::map<std::string, ConfidenceExtractor> confidence_extractors,
                                              std::shared_ptr<IRoutingPolicy> router, RuntimeOptions& options,
                                              MetricsRegistry& metrics) {
    std::ifstream in(config_path);
    if (!in) throw LoomcoreError("Runtime: cannot open config file '" + config_path + "'");
    nlohmann::json doc;
    try {
        in >> doc;
    } catch (const nlohmann::json::parse_error& ex) {
        throw LoomcoreError("Runtime: malformed JSON in '" + config_path + "': " + ex.what());
    }

    if (doc.contains("scheduler")) {
        const auto& s = doc["scheduler"];
        if (s.contains("cpu_threads")) options.scheduler.cpu_threads = s["cpu_threads"].get<int>();
        if (s.contains("gpu_sim_threads")) options.scheduler.gpu_sim_threads = s["gpu_sim_threads"].get<int>();
        if (s.contains("gpu_sim_fixed_overhead_ms"))
            options.scheduler.gpu_sim_fixed_overhead_ms = s["gpu_sim_fixed_overhead_ms"].get<double>();
        if (s.contains("gpu_sim_bytes_per_ms"))
            options.scheduler.gpu_sim_bytes_per_ms = s["gpu_sim_bytes_per_ms"].get<double>();
    }
    if (doc.contains("log_file") && options.log_file.empty()) {
        options.log_file = doc["log_file"].get<std::string>();
    }
    Logger::instance().configure(options.log_file, options.log_to_stdout);

    if (!doc.contains("nodes") || !doc["nodes"].is_array()) {
        throw LoomcoreError("Runtime: config must contain a 'nodes' array");
    }

    Graph graph;
    for (const auto& jn : doc["nodes"]) {
        NodeConfig cfg;
        cfg.id = jn.at("id").get<std::string>();
        if (jn.contains("backend")) cfg.backend = parseBackend(jn["backend"].get<std::string>());
        if (jn.contains("priority")) cfg.priority = jn["priority"].get<int>();
        if (jn.contains("max_batch_size")) cfg.max_batch_size = jn["max_batch_size"].get<size_t>();
        if (jn.contains("batch_window_ms")) cfg.batch_window_ms = jn["batch_window_ms"].get<int>();
        if (jn.contains("quality_weight")) cfg.quality_weight = jn["quality_weight"].get<double>();
        if (jn.contains("depends_on")) {
            for (const auto& d : jn["depends_on"]) cfg.depends_on.push_back(d.get<std::string>());
        }
        if (!jn.contains("variants") || !jn["variants"].is_array() || jn["variants"].empty()) {
            throw LoomcoreError("Runtime: node '" + cfg.id + "' must declare a non-empty 'variants' array");
        }
        for (const auto& jv : jn["variants"]) {
            VariantConfig vc;
            vc.precision = parsePrecision(jv.at("precision").get<std::string>());
            vc.model_path = jv.at("model_path").get<std::string>();
            cfg.variants.push_back(vc);
        }

        auto bind_it = binders.find(cfg.id);
        if (bind_it == binders.end()) {
            throw LoomcoreError("Runtime: no NodeInputBinder registered for node '" + cfg.id + "'");
        }
        cfg.binder = bind_it->second;

        if (jn.contains("confidence_source")) {
            cfg.confidence_source_node = jn["confidence_source"].get<std::string>();
            auto conf_it = confidence_extractors.find(cfg.id);
            if (conf_it == confidence_extractors.end()) {
                throw LoomcoreError("Runtime: node '" + cfg.id +
                                     "' declares 'confidence_source' but no ConfidenceExtractor was registered for it");
            }
            cfg.confidence = conf_it->second;
        }

        graph.addNode(std::move(cfg));
    }
    graph.validate();

    std::map<std::string, ModelNode> nodes;
    for (const auto& cfg : graph.nodes()) {
        ModelNode mnode;
        mnode.config = cfg;
        for (const auto& v : cfg.variants) {
            mnode.variants[v.precision] = std::make_shared<ModelVariant>(
                Environment::shared(), v.model_path, v.precision, options.intra_op_threads_per_variant);
        }
        nodes[cfg.id] = std::move(mnode);
    }

    // Two-phase construction: graph/nodes are moved into their final,
    // stable heap location inside `snap` *before* the Scheduler is built,
    // since Scheduler stores references to them (see loomcore/scheduler.h)
    // that must remain valid for the Scheduler's entire lifetime — which,
    // via GraphSnapshot's own shared_ptr lifetime, may outlast this
    // function by as long as the slowest job still pinning it runs.
    auto snap = std::make_shared<GraphSnapshot>();
    snap->graph = std::move(graph);
    snap->nodes = std::move(nodes);
    snap->router = router ? router : std::make_shared<CompositeRouter>();
    snap->scheduler = std::make_shared<Scheduler>(snap->graph, snap->nodes, metrics, snap->router, options.scheduler);
    return snap;
}

} // namespace

void Runtime::loadGraph(const std::string& config_path, std::map<std::string, NodeInputBinder> binders,
                         std::map<std::string, ConfidenceExtractor> confidence_extractors,
                         std::shared_ptr<IRoutingPolicy> router, RuntimeOptions options) {
    auto snap = buildSnapshot(config_path, std::move(binders), std::move(confidence_extractors), std::move(router),
                               options, impl_->metrics);
    std::lock_guard<std::mutex> lock(impl_->swap_mutex);
    std::atomic_store(&impl_->current, snap);
}

void Runtime::reloadGraph(const std::string& config_path, std::map<std::string, NodeInputBinder> binders,
                           std::map<std::string, ConfidenceExtractor> confidence_extractors,
                           std::shared_ptr<IRoutingPolicy> router, RuntimeOptions options) {
    if (!impl_->currentSnapshot()) {
        throw LoomcoreError("Runtime::reloadGraph: loadGraph() must be called at least once first");
    }
    // Everything expensive — parsing JSON, loading every ONNX session for
    // the new graph — happens *before* this function ever touches
    // impl_->current, so the old graph keeps serving every job for the
    // whole (potentially hundreds-of-milliseconds) duration of building
    // the new one; there is no window where traffic is paused.
    auto snap = buildSnapshot(config_path, std::move(binders), std::move(confidence_extractors), std::move(router),
                               options, impl_->metrics);
    std::lock_guard<std::mutex> lock(impl_->swap_mutex);
    std::atomic_store(&impl_->current, snap);
    // The old GraphSnapshot is destroyed right here if this call dropped
    // its last reference, or later — the instant the last job still
    // pinning it (via Scheduler::submitJob's `keep_alive`, passed by
    // submit() below) settles — if not. Either way this call never blocks
    // on that destruction, and no in-flight job is disturbed by it: it
    // keeps running against the exact Graph/ModelNode/Scheduler it started
    // with.
}

std::future<JobResult> Runtime::submit(TensorMap inputs, int priority, double time_budget_ms) {
    auto snap = impl_->currentSnapshot();
    if (!snap) throw LoomcoreError("Runtime::submit: loadGraph() has not been called");
    auto* scheduler = snap->scheduler.get();
    // Pass a deferred-teardown proxy (see Impl::deferredKeepAlive), not
    // `snap` itself: this job's *last* completion callback may run on one
    // of `scheduler`'s own lane threads, and if that happens to be the
    // last reference to `snap`, destroying it inline would destroy
    // `scheduler` and join those very threads from themselves.
    return scheduler->submitJob(std::move(inputs), priority, time_budget_ms, impl_->deferredKeepAlive(snap));
}

JobResult Runtime::run(TensorMap inputs, int priority, double time_budget_ms) {
    return submit(std::move(inputs), priority, time_budget_ms).get();
}

MetricsRegistry::Stats Runtime::nodeStats(const std::string& node_id) const { return impl_->metrics.stats(node_id); }

std::vector<std::string> Runtime::recentLogs(size_t n) const { return Logger::instance().recentLines(n); }

const Graph& Runtime::graph() const {
    auto snap = impl_->currentSnapshot();
    if (!snap) throw LoomcoreError("Runtime::graph: loadGraph() has not been called");
    return snap->graph;
}

} // namespace loomcore
