/* Loomcore — C ABI.
 *
 * Everywhere else in this repo, "the ORT dependency is fully encapsulated"
 * is the boundary Loomcore is careful about (see loomcore/environment.h).
 * This header is about a different boundary: loomcore::Runtime's public
 * API is C++ — std::string, std::function, std::shared_ptr — which means
 * a consumer must be built with a compatible compiler *and* C++ ABI
 * (CMakeLists.txt's /wd4251 suppression on Windows is explicit that this
 * only holds because every in-repo consumer links the exact
 * loomcore_core.dll built alongside it). A shared library whose only
 * consumable surface requires that is a vendored header, not a library.
 *
 * This is a second, independent, `extern "C"` surface over the same
 * loomcore_core: opaque handles, POD structs, integer status codes
 * instead of C++ exceptions. It covers Runtime's lifecycle, loading a
 * graph with C-callable binders, running one job synchronously, and
 * reading back node latency stats — not the full C++ API (no async
 * submit()/future, no custom IRoutingPolicy, no Python-style upstream-
 * outputs-as-a-map) — see bindings/c/smoke_test.c for a real, pure-C
 * program built and run against nothing but this header and the compiled
 * .dll/.so, and docs/CLAIMS.md #13 for how to run it yourself.
 */
#ifndef LOOMCORE_API_H
#define LOOMCORE_API_H

#include <stddef.h>
#include <stdint.h>

#include "loomcore/export.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Bump whenever a breaking change is made to this header's ABI (struct
 * layout, function signatures, enum values). A consumer can check this at
 * runtime against the version it was compiled against. */
LOOMCORE_API uint32_t loomcore_c_abi_version(void);

typedef enum LoomcoreCDType {
    LOOMCORE_C_FLOAT32 = 0,
    LOOMCORE_C_INT64 = 1,
    LOOMCORE_C_INT32 = 2
} LoomcoreCDType;

typedef enum LoomcoreCStatus {
    LOOMCORE_C_OK = 0,
    LOOMCORE_C_ERROR = 1,          /* see loomcore_c_last_error() */
    LOOMCORE_C_JOB_REJECTED = 2,   /* admission control refused the job — see JobRejectedError */
    LOOMCORE_C_DEADLINE_EXCEEDED = 3 /* the deadline reaper cancelled the job — see DeadlineExceededError */
} LoomcoreCStatus;

/* A borrowed view, never an owning handle: `shape`/`data` point into
 * memory the CALLER (for inputs) or Loomcore (for results, see
 * LoomcoreResultSet below) owns. `element_count` is the number of
 * elements (not bytes) `data` points to, consistent across every dtype. */
typedef struct LoomcoreCTensor {
    const char* name;
    const int64_t* shape;
    size_t shape_len;
    LoomcoreCDType dtype;
    const void* data;
    size_t element_count;
} LoomcoreCTensor;

/* One tensor plus which upstream node produced it (NULL for a graph
 * input) — this is how binder callbacks see both the job's external
 * inputs and every upstream node's outputs: as one flat array each,
 * rather than the C++ API's map<node_id, vector<NamedTensor>>, which has
 * no natural C representation. A binder filters by `owner_node_id`
 * itself (strcmp) to find the specific upstream tensors it needs — see
 * loomcore/model_node.h's NodeExecutionContext for the C++ equivalent
 * this flattens. */
typedef struct LoomcoreNamedCTensor {
    const char* owner_node_id; /* NULL => graph input, not an upstream output */
    LoomcoreCTensor tensor;
} LoomcoreNamedCTensor;

/* Loomcore's C++ NodeInputBinder as a C function pointer: given the job's
 * graph inputs and every upstream node's outputs (both flattened, see
 * above), write up to `out_capacity` output tensors into `out_tensors`
 * and set `*out_count`, returning 0. `out_tensors[i].data`/`.shape` only
 * need to stay valid until this call returns — Loomcore copies them
 * immediately into its own owned storage before dispatching the node's
 * ONNX Runtime session, exactly the same synchronous-handoff contract
 * `LoomcoreCTensor` uses everywhere else in this header. Return nonzero
 * to fail the node (and thus the job) with a message set via
 * loomcore_c_set_binder_error(). */
typedef int (*LoomcoreBinderFn)(const LoomcoreNamedCTensor* graph_inputs, size_t n_graph_inputs,
                                 const LoomcoreNamedCTensor* upstream_tensors, size_t n_upstream_tensors,
                                 void* user_data, LoomcoreCTensor* out_tensors, size_t out_capacity,
                                 size_t* out_count);

/* Call from inside a LoomcoreBinderFn that is about to return nonzero, to
 * attach a message loomcore_c_last_error() will report after the failed
 * job's run() call returns LOOMCORE_C_ERROR. Optional — a generic message
 * is used if this isn't called. */
LOOMCORE_API void loomcore_c_set_binder_error(const char* message);

/* The most recent error message on *this thread* (Loomcore's own C++
 * exceptions are translated to this at the C boundary; nothing throws
 * across it). Valid until the next loomcore_c_* call on this thread. */
LOOMCORE_API const char* loomcore_c_last_error(void);

typedef struct LoomcoreRuntime LoomcoreRuntime; /* opaque */

LOOMCORE_API LoomcoreRuntime* loomcore_runtime_create(void);
LOOMCORE_API void loomcore_runtime_destroy(LoomcoreRuntime* rt);

/* Must be called once per node id declared in the graph config, before
 * loomcore_runtime_load_graph — mirrors the C++ API's requirement that
 * every node has a registered NodeInputBinder (see runtime.h). Returns 0
 * on success. */
LOOMCORE_API int loomcore_runtime_set_binder(LoomcoreRuntime* rt, const char* node_id, LoomcoreBinderFn fn,
                                                void* user_data);

/* Loads the graph (same JSON schema as the C++ API — see
 * docs/ARCHITECTURE.md "Graph config schema"). Nodes with no confidence
 * source declared in the config need no further setup; confidence-gated
 * nodes are not yet exposed through this C surface (a scope this header's
 * own comment states plainly — see the top of this file). */
LOOMCORE_API LoomcoreCStatus loomcore_runtime_load_graph(LoomcoreRuntime* rt, const char* config_path);

/* Synchronous run. `inputs`/`n_inputs` are the job's external inputs;
 * `time_budget_ms < 0` means unbounded. On LOOMCORE_C_OK, `*out_result`
 * is a new LoomcoreResultSet the caller must eventually pass to
 * loomcore_result_destroy(); on any other status it is left untouched and
 * loomcore_c_last_error() has details. */
typedef struct LoomcoreResultSet LoomcoreResultSet; /* opaque */

LOOMCORE_API LoomcoreCStatus loomcore_runtime_run(LoomcoreRuntime* rt, const LoomcoreCTensor* inputs,
                                                     size_t n_inputs, double time_budget_ms,
                                                     LoomcoreResultSet** out_result);

LOOMCORE_API size_t loomcore_result_count(const LoomcoreResultSet* rs);
/* Fills `*out_node_id` and `*out_tensor` for sink-tensor `index` (see
 * docs/ARCHITECTURE.md: a job's result is every sink node's outputs, in
 * no particular cross-node order but stable within one ResultSet). The
 * tensor's `data`/`shape` remain valid until loomcore_result_destroy(). */
LOOMCORE_API int loomcore_result_get(const LoomcoreResultSet* rs, size_t index, const char** out_node_id,
                                        LoomcoreCTensor* out_tensor);
LOOMCORE_API void loomcore_result_destroy(LoomcoreResultSet* rs);

/* Rolling latency stats for one node (see MetricsRegistry::Stats in the
 * C++ API). Returns 0 on success (the node exists), nonzero otherwise. */
LOOMCORE_API int loomcore_node_stats(LoomcoreRuntime* rt, const char* node_id, double* out_p50_ms,
                                        double* out_p95_ms, double* out_mean_ms, size_t* out_count);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LOOMCORE_API_H */
