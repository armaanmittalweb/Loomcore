// Loomcore — DAG node configuration and the loaded model wrapper.
#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "loomcore/environment.h"
#include "loomcore/export.h"
#include "loomcore/types.h"

namespace loomcore {

// Snapshot of everything a node's input binder is allowed to see: the
// external inputs the whole graph run was started with, plus the already-
// computed outputs of every upstream node this node (transitively or
// directly) depends on. Binders are pure functions of this context so they
// stay trivially testable in isolation from the scheduler.
//
// `upstream_outputs`' values are shared, not copied: the scheduler snapshots
// this map once per node dispatch (see docs/ARCHITECTURE.md "Scheduler"),
// and every job sharing a completed upstream node shares the same
// heap-allocated `vector<NamedTensor>` rather than each dispatch deep-copying
// every upstream tensor it might never read. Binders only ever see it
// through the read-only accessors below.
struct LOOMCORE_API NodeExecutionContext {
    std::string job_id;
    const TensorMap* graph_inputs = nullptr;
    const std::map<std::string, std::shared_ptr<const std::vector<NamedTensor>>>* upstream_outputs = nullptr;

    // Convenience accessor: the single named output tensor produced by
    // upstream node `node_id`. Throws LoomcoreError if either is missing.
    const NamedTensor& upstreamTensor(const std::string& node_id, const std::string& tensor_name) const;
    // First (and typically only) output tensor of an upstream node.
    const NamedTensor& upstreamFirst(const std::string& node_id) const;
};

// Produces the ordered list of input tensors a node's ONNX session should
// be run with, given the current job's external inputs and upstream
// results. This is the one place DAG "wiring semantics" live: e.g. turning
// an upstream classifier's logits into a token-id tensor for a downstream
// text encoder. Config files describe topology; binders describe data
// flow, and are therefore registered in code (see Runtime::loadGraph).
using NodeInputBinder = std::function<std::vector<NamedTensor>(const NodeExecutionContext&)>;

// Optional: extracts a scalar "confidence"-like signal from a node's own
// output tensors, so a routing policy can gate a downstream node's
// execution on it without the router needing to know that node's tensor
// layout. See loomcore::ConfidenceGatePolicy.
using ConfidenceExtractor = std::function<std::optional<float>(const std::vector<NamedTensor>&)>;

struct VariantConfig {
    Precision precision = Precision::FP32;
    std::string model_path;
};

struct LOOMCORE_API NodeConfig {
    std::string id;
    std::vector<std::string> depends_on;

    Backend backend = Backend::CPU;
    int priority = 0;               // higher runs first, all else equal
    size_t max_batch_size = 1;      // 1 disables batching
    int batch_window_ms = 0;        // 0 = dispatch as soon as the lane is free

    std::vector<VariantConfig> variants; // at least one; keyed by precision
    NodeInputBinder binder;              // required before Runtime::loadGraph completes

    // Optional confidence gate: if `confidence` is set, `confidence_source_node`
    // names the upstream node (must be listed in depends_on) whose outputs are
    // handed to it. See ConfidenceGatePolicy in loomcore/router.h.
    ConfidenceExtractor confidence;
    std::string confidence_source_node;

    // How much this node's prediction quality is presumed to matter,
    // relative to other nodes, when PrecisionPlanner (loomcore/planner.h)
    // has to choose which subset of critical-path nodes to downgrade to
    // INT8 to fit a deadline. Higher = more precision-sensitive = more
    // strongly preferred to stay at FP32. Purely a caller-declared hint —
    // Loomcore has no accuracy measurement of its own (see
    // docs/BENCHMARKS.md's "latency only, never accuracy" caveat) — and
    // defaults to 1.0 for every node, under which the DP's chosen set
    // coincides with the simpler "fewest nodes downgraded" greedy answer.
    double quality_weight = 1.0;

    const VariantConfig* variant(Precision p) const;
    bool hasVariant(Precision p) const;
};

// A cooperative cancellation handle for one in-flight (or about-to-run)
// ModelVariant::run() call. Binding one to a call lets another thread — the
// scheduler's deadline reaper (see Scheduler / RuntimeOptions) — request
// that ONNX Runtime abort the run at its next op boundary once a job's
// time budget has passed, rather than merely deciding *not to route to*
// a node ahead of time (that's what LatencyBudgetPolicy does). This is a
// thin, header-clean wrapper around Ort::RunOptions::SetTerminate so that
// no public Loomcore header needs to include the ONNX Runtime C++ API
// (see loomcore/environment.h for why that boundary matters).
//
// Thread-safe: `requestCancel()` is meant to be called from a thread other
// than the one executing `run()`.
class LOOMCORE_API CancellationToken {
public:
    CancellationToken();
    ~CancellationToken();
    CancellationToken(const CancellationToken&) = delete;
    CancellationToken& operator=(const CancellationToken&) = delete;

    // Requests that any run currently (or subsequently) bound to this token
    // abort as soon as ONNX Runtime next checks for termination. Idempotent.
    void requestCancel();
    bool cancelled() const;

    // Opaque `Ort::RunOptions*`, cast back internally by model_node.cpp.
    void* nativeHandle() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// One loaded ONNX Runtime session for a single (node, precision) pair,
// plus the introspected input/output metadata used to validate binder
// output and to build ONNX Runtime tensors without the caller needing the
// ORT C++ API.
class LOOMCORE_API ModelVariant {
public:
    ModelVariant(Environment& env, const std::string& model_path, Precision precision, int intra_op_threads);
    ~ModelVariant();
    ModelVariant(ModelVariant&&) noexcept;
    ModelVariant& operator=(ModelVariant&&) noexcept;
    ModelVariant(const ModelVariant&) = delete;
    ModelVariant& operator=(const ModelVariant&) = delete;

    Precision precision() const { return precision_; }
    const std::vector<std::string>& inputNames() const { return input_names_; }
    const std::vector<std::string>& outputNames() const { return output_names_; }

    // Runs a batch. `inputs` must be ordered to match inputNames() (the
    // scheduler enforces this via the binder's declared order at load
    // time). Returns outputs in outputNames() order. If `cancel` is
    // non-null, the run is bound to it for its duration so a concurrent
    // `cancel->requestCancel()` aborts the underlying ONNX Runtime call;
    // an aborted run throws LoomcoreError.
    std::vector<NamedTensor> run(const std::vector<NamedTensor>& inputs, CancellationToken* cancel = nullptr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Precision precision_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

// A logical DAG node: its config plus one loaded ModelVariant per declared
// precision.
struct LOOMCORE_API ModelNode {
    NodeConfig config;
    std::map<Precision, std::shared_ptr<ModelVariant>> variants;

    std::shared_ptr<ModelVariant> variant(Precision p) const;
};

} // namespace loomcore
