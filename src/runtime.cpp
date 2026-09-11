#include "loomcore/runtime.h"

#include <fstream>
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

struct Runtime::Impl {
    Graph graph;
    std::map<std::string, ModelNode> nodes;
    MetricsRegistry metrics;
    std::shared_ptr<IRoutingPolicy> router;
    std::unique_ptr<Scheduler> scheduler;
};

Runtime::Runtime() : impl_(std::make_unique<Impl>()) {
    (void)Environment::shared(); // force initialization up front, at a predictable point
}

Runtime::~Runtime() = default;

void Runtime::loadGraph(const std::string& config_path, std::map<std::string, NodeInputBinder> binders,
                         std::map<std::string, ConfidenceExtractor> confidence_extractors,
                         std::shared_ptr<IRoutingPolicy> router, RuntimeOptions options) {
    std::ifstream in(config_path);
    if (!in) throw LoomcoreError("Runtime::loadGraph: cannot open config file '" + config_path + "'");
    nlohmann::json doc;
    try {
        in >> doc;
    } catch (const nlohmann::json::parse_error& ex) {
        throw LoomcoreError("Runtime::loadGraph: malformed JSON in '" + config_path + "': " + ex.what());
    }

    Logger::instance().configure(options.log_file, /*also_stdout=*/true);

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
        Logger::instance().configure(options.log_file, true);
    }

    if (!doc.contains("nodes") || !doc["nodes"].is_array()) {
        throw LoomcoreError("Runtime::loadGraph: config must contain a 'nodes' array");
    }

    Graph graph;
    for (const auto& jn : doc["nodes"]) {
        NodeConfig cfg;
        cfg.id = jn.at("id").get<std::string>();
        if (jn.contains("backend")) cfg.backend = parseBackend(jn["backend"].get<std::string>());
        if (jn.contains("priority")) cfg.priority = jn["priority"].get<int>();
        if (jn.contains("max_batch_size")) cfg.max_batch_size = jn["max_batch_size"].get<size_t>();
        if (jn.contains("batch_window_ms")) cfg.batch_window_ms = jn["batch_window_ms"].get<int>();
        if (jn.contains("depends_on")) {
            for (const auto& d : jn["depends_on"]) cfg.depends_on.push_back(d.get<std::string>());
        }
        if (!jn.contains("variants") || !jn["variants"].is_array() || jn["variants"].empty()) {
            throw LoomcoreError("Runtime::loadGraph: node '" + cfg.id + "' must declare a non-empty 'variants' array");
        }
        for (const auto& jv : jn["variants"]) {
            VariantConfig vc;
            vc.precision = parsePrecision(jv.at("precision").get<std::string>());
            vc.model_path = jv.at("model_path").get<std::string>();
            cfg.variants.push_back(vc);
        }

        auto bind_it = binders.find(cfg.id);
        if (bind_it == binders.end()) {
            throw LoomcoreError("Runtime::loadGraph: no NodeInputBinder registered for node '" + cfg.id + "'");
        }
        cfg.binder = bind_it->second;

        if (jn.contains("confidence_source")) {
            cfg.confidence_source_node = jn["confidence_source"].get<std::string>();
            auto conf_it = confidence_extractors.find(cfg.id);
            if (conf_it == confidence_extractors.end()) {
                throw LoomcoreError("Runtime::loadGraph: node '" + cfg.id +
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

    impl_->graph = std::move(graph);
    impl_->nodes = std::move(nodes);
    impl_->router = router ? router : std::make_shared<CompositeRouter>();
    impl_->scheduler = std::make_unique<Scheduler>(impl_->graph, impl_->nodes, impl_->metrics, impl_->router,
                                                    options.scheduler);
}

std::future<JobResult> Runtime::submit(TensorMap inputs, int priority, double time_budget_ms) {
    if (!impl_->scheduler) throw LoomcoreError("Runtime::submit: loadGraph() has not been called");
    return impl_->scheduler->submitJob(std::move(inputs), priority, time_budget_ms);
}

JobResult Runtime::run(TensorMap inputs, int priority, double time_budget_ms) {
    return submit(std::move(inputs), priority, time_budget_ms).get();
}

MetricsRegistry::Stats Runtime::nodeStats(const std::string& node_id) const { return impl_->metrics.stats(node_id); }

std::vector<std::string> Runtime::recentLogs(size_t n) const { return Logger::instance().recentLines(n); }

const Graph& Runtime::graph() const { return impl_->graph; }

} // namespace loomcore
