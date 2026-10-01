#include "loomcore/c_api.h"

#include "loomcore/runtime.h"

#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace loomcore;

namespace {

constexpr uint32_t kAbiVersion = 1;
constexpr size_t kMaxBinderOutputs = 8; // see c_api.h's header comment on this header's intentional scope

thread_local std::string g_last_error;
thread_local std::string g_binder_error; // set by loomcore_c_set_binder_error, consumed once per binder call

void setLastError(const std::string& msg) { g_last_error = msg; }

DType fromCDType(LoomcoreCDType d) {
    switch (d) {
        case LOOMCORE_C_FLOAT32: return DType::Float32;
        case LOOMCORE_C_INT64: return DType::Int64;
        case LOOMCORE_C_INT32: return DType::Int32;
    }
    return DType::Float32;
}

LoomcoreCDType toCDType(DType d) {
    switch (d) {
        case DType::Float32: return LOOMCORE_C_FLOAT32;
        case DType::Int64: return LOOMCORE_C_INT64;
        case DType::Int32: return LOOMCORE_C_INT32;
    }
    return LOOMCORE_C_FLOAT32;
}

// Borrowed view: `t` must outlive the returned LoomcoreCTensor.
LoomcoreCTensor toCTensor(const NamedTensor& t) {
    LoomcoreCTensor ct{};
    ct.name = t.name.c_str();
    ct.shape = t.shape.data();
    ct.shape_len = t.shape.size();
    ct.dtype = toCDType(t.dtype);
    switch (t.dtype) {
        case DType::Float32:
            ct.data = t.f32.data();
            ct.element_count = t.f32.size();
            break;
        case DType::Int64:
            ct.data = t.i64.data();
            ct.element_count = t.i64.size();
            break;
        case DType::Int32:
            ct.data = t.i32.data();
            ct.element_count = t.i32.size();
            break;
    }
    return ct;
}

// Copies out of `ct` immediately (the synchronous-handoff contract
// documented in c_api.h): the caller's buffer need not outlive this call.
NamedTensor fromCTensor(const LoomcoreCTensor& ct) {
    std::string name = ct.name ? ct.name : "";
    std::vector<int64_t> shape(ct.shape, ct.shape + ct.shape_len);
    switch (fromCDType(ct.dtype)) {
        case DType::Float32: {
            const float* p = static_cast<const float*>(ct.data);
            return NamedTensor::makeFloat(std::move(name), std::move(shape), std::vector<float>(p, p + ct.element_count));
        }
        case DType::Int64: {
            const int64_t* p = static_cast<const int64_t*>(ct.data);
            return NamedTensor::makeInt64(std::move(name), std::move(shape),
                                           std::vector<int64_t>(p, p + ct.element_count));
        }
        case DType::Int32: {
            const int32_t* p = static_cast<const int32_t*>(ct.data);
            return NamedTensor::makeInt32(std::move(name), std::move(shape),
                                           std::vector<int32_t>(p, p + ct.element_count));
        }
    }
    return NamedTensor{};
}

struct RegisteredBinder {
    LoomcoreBinderFn fn = nullptr;
    void* user_data = nullptr;
};

} // namespace

struct LoomcoreRuntime {
    Runtime runtime;
    std::map<std::string, RegisteredBinder> binders;
};

struct LoomcoreResultSet {
    // Flattened (node_id, tensor) pairs — one entry per sink-node output
    // tensor, owning its own NamedTensor so the LoomcoreCTensor views
    // handed back by loomcore_result_get stay valid until destroy().
    std::vector<std::pair<std::string, NamedTensor>> entries;
};

extern "C" {

uint32_t loomcore_c_abi_version(void) { return kAbiVersion; }

void loomcore_c_set_binder_error(const char* message) { g_binder_error = message ? message : ""; }

const char* loomcore_c_last_error(void) { return g_last_error.c_str(); }

LoomcoreRuntime* loomcore_runtime_create(void) {
    try {
        return new LoomcoreRuntime();
    } catch (const std::exception& ex) {
        setLastError(ex.what());
        return nullptr;
    } catch (...) {
        setLastError("unknown error constructing Runtime");
        return nullptr;
    }
}

void loomcore_runtime_destroy(LoomcoreRuntime* rt) { delete rt; }

int loomcore_runtime_set_binder(LoomcoreRuntime* rt, const char* node_id, LoomcoreBinderFn fn, void* user_data) {
    if (!rt || !node_id || !fn) {
        setLastError("loomcore_runtime_set_binder: null argument");
        return 1;
    }
    rt->binders[node_id] = RegisteredBinder{fn, user_data};
    return 0;
}

LoomcoreCStatus loomcore_runtime_load_graph(LoomcoreRuntime* rt, const char* config_path) {
    if (!rt || !config_path) {
        setLastError("loomcore_runtime_load_graph: null argument");
        return LOOMCORE_C_ERROR;
    }
    try {
        std::map<std::string, NodeInputBinder> cpp_binders;
        for (const auto& [node_id, reg] : rt->binders) {
            LoomcoreBinderFn fn = reg.fn;
            void* user_data = reg.user_data;
            std::string node_id_copy = node_id;
            cpp_binders[node_id] = [fn, user_data, node_id_copy](const NodeExecutionContext& ctx) -> std::vector<NamedTensor> {
                // Flatten graph_inputs and upstream_outputs into the two
                // flat C arrays LoomcoreBinderFn expects — see c_api.h.
                std::vector<LoomcoreNamedCTensor> graph_inputs_c;
                if (ctx.graph_inputs) {
                    graph_inputs_c.reserve(ctx.graph_inputs->size());
                    for (const auto& [k, v] : *ctx.graph_inputs) {
                        LoomcoreNamedCTensor nct{};
                        nct.owner_node_id = nullptr;
                        nct.tensor = toCTensor(v);
                        graph_inputs_c.push_back(nct);
                    }
                }
                std::vector<LoomcoreNamedCTensor> upstream_c;
                std::vector<std::string> owner_ids; // keep c_str() pointers stable across the loop below
                if (ctx.upstream_outputs) {
                    size_t total = 0;
                    for (const auto& [k, vec] : *ctx.upstream_outputs) total += vec ? vec->size() : 0;
                    upstream_c.reserve(total);
                    owner_ids.reserve(ctx.upstream_outputs->size());
                    for (const auto& [k, vec] : *ctx.upstream_outputs) {
                        if (!vec) continue;
                        owner_ids.push_back(k);
                        const char* owner = owner_ids.back().c_str();
                        for (const auto& t : *vec) {
                            LoomcoreNamedCTensor nct{};
                            nct.owner_node_id = owner;
                            nct.tensor = toCTensor(t);
                            upstream_c.push_back(nct);
                        }
                    }
                }

                LoomcoreCTensor out_buf[kMaxBinderOutputs];
                size_t out_count = 0;
                g_binder_error.clear();
                int rc = fn(graph_inputs_c.empty() ? nullptr : graph_inputs_c.data(), graph_inputs_c.size(),
                            upstream_c.empty() ? nullptr : upstream_c.data(), upstream_c.size(), user_data, out_buf,
                            kMaxBinderOutputs, &out_count);
                if (rc != 0) {
                    throw LoomcoreError("C binder for node '" + node_id_copy +
                                        "' failed: " + (g_binder_error.empty() ? "(no message set)" : g_binder_error));
                }
                std::vector<NamedTensor> result;
                result.reserve(out_count);
                for (size_t i = 0; i < out_count; ++i) result.push_back(fromCTensor(out_buf[i]));
                return result;
            };
        }
        rt->runtime.loadGraph(config_path, std::move(cpp_binders));
        return LOOMCORE_C_OK;
    } catch (const std::exception& ex) {
        setLastError(ex.what());
        return LOOMCORE_C_ERROR;
    } catch (...) {
        setLastError("unknown error in loomcore_runtime_load_graph");
        return LOOMCORE_C_ERROR;
    }
}

LoomcoreCStatus loomcore_runtime_run(LoomcoreRuntime* rt, const LoomcoreCTensor* inputs, size_t n_inputs,
                                     double time_budget_ms, LoomcoreResultSet** out_result) {
    if (!rt || !out_result) {
        setLastError("loomcore_runtime_run: null argument");
        return LOOMCORE_C_ERROR;
    }
    try {
        TensorMap tensor_map;
        for (size_t i = 0; i < n_inputs; ++i) {
            NamedTensor t = fromCTensor(inputs[i]);
            tensor_map[t.name] = std::move(t);
        }
        JobResult result = rt->runtime.run(std::move(tensor_map), /*priority=*/0, time_budget_ms);

        auto rs = std::make_unique<LoomcoreResultSet>();
        for (auto& [node_id, tensors] : result) {
            for (auto& t : tensors) rs->entries.emplace_back(node_id, std::move(t));
        }
        *out_result = rs.release();
        return LOOMCORE_C_OK;
    } catch (const JobRejectedError& ex) {
        setLastError(ex.what());
        return LOOMCORE_C_JOB_REJECTED;
    } catch (const DeadlineExceededError& ex) {
        setLastError(ex.what());
        return LOOMCORE_C_DEADLINE_EXCEEDED;
    } catch (const std::exception& ex) {
        setLastError(ex.what());
        return LOOMCORE_C_ERROR;
    } catch (...) {
        setLastError("unknown error in loomcore_runtime_run");
        return LOOMCORE_C_ERROR;
    }
}

size_t loomcore_result_count(const LoomcoreResultSet* rs) { return rs ? rs->entries.size() : 0; }

int loomcore_result_get(const LoomcoreResultSet* rs, size_t index, const char** out_node_id,
                         LoomcoreCTensor* out_tensor) {
    if (!rs || index >= rs->entries.size() || !out_node_id || !out_tensor) return 1;
    const auto& [node_id, tensor] = rs->entries[index];
    *out_node_id = node_id.c_str();
    *out_tensor = toCTensor(tensor);
    return 0;
}

void loomcore_result_destroy(LoomcoreResultSet* rs) { delete rs; }

int loomcore_node_stats(LoomcoreRuntime* rt, const char* node_id, double* out_p50_ms, double* out_p95_ms,
                         double* out_mean_ms, size_t* out_count) {
    if (!rt || !node_id) return 1;
    try {
        auto stats = rt->runtime.nodeStats(node_id);
        if (out_p50_ms) *out_p50_ms = stats.p50_ms;
        if (out_p95_ms) *out_p95_ms = stats.p95_ms;
        if (out_mean_ms) *out_mean_ms = stats.mean_ms;
        if (out_count) *out_count = stats.count;
        return 0;
    } catch (...) {
        return 1;
    }
}

} // extern "C"
