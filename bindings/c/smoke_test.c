/* Loomcore C API smoke test — see docs/CLAIMS.md #13.
 *
 * Deliberately plain C (compiled with a C compiler, not C++ — see
 * bindings/c/CMakeLists.txt's LANGUAGES C), including no C++ Loomcore
 * header, to prove the C ABI (include/loomcore/c_api.h) is a real,
 * separately-consumable surface over loomcore_core and not merely a
 * header that happens to also compile under extern "C".
 *
 * Builds the same tiny two-node DAG tests/test_scheduler.cpp uses
 * (identity -> double over the hermetic fixture ONNX graphs in assets/,
 * no downloaded reference models needed), runs one job through the C
 * API, and checks the numeric result by hand.
 */
#include "loomcore/c_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* node "a": y = identity(x). Passes the graph input named "x" straight
 * through as this node's only output. */
static int identity_binder(const LoomcoreNamedCTensor* graph_inputs, size_t n_graph_inputs,
                            const LoomcoreNamedCTensor* upstream_tensors, size_t n_upstream_tensors, void* user_data,
                            LoomcoreCTensor* out_tensors, size_t out_capacity, size_t* out_count) {
    (void)upstream_tensors;
    (void)n_upstream_tensors;
    (void)user_data;
    for (size_t i = 0; i < n_graph_inputs; ++i) {
        if (graph_inputs[i].owner_node_id == NULL && strcmp(graph_inputs[i].tensor.name, "x") == 0) {
            if (out_capacity < 1) return 1;
            out_tensors[0] = graph_inputs[i].tensor;
            *out_count = 1;
            return 0;
        }
    }
    loomcore_c_set_binder_error("no graph input named 'x'");
    return 1;
}

/* node "b": y = double(a's output). Passes node "a"'s output straight
 * through as this node's only input; test_double.onnx does the actual
 * doubling (y = x + x). */
static int double_binder(const LoomcoreNamedCTensor* graph_inputs, size_t n_graph_inputs,
                          const LoomcoreNamedCTensor* upstream_tensors, size_t n_upstream_tensors, void* user_data,
                          LoomcoreCTensor* out_tensors, size_t out_capacity, size_t* out_count) {
    (void)graph_inputs;
    (void)n_graph_inputs;
    (void)user_data;
    for (size_t i = 0; i < n_upstream_tensors; ++i) {
        if (upstream_tensors[i].owner_node_id != NULL && strcmp(upstream_tensors[i].owner_node_id, "a") == 0) {
            if (out_capacity < 1) return 1;
            out_tensors[0] = upstream_tensors[i].tensor;
            *out_count = 1;
            return 0;
        }
    }
    loomcore_c_set_binder_error("no upstream tensor from node 'a'");
    return 1;
}

int main(int argc, char** argv) {
    const char* config_path = argc > 1 ? argv[1] : "bindings/c/smoke_test_config.json";

    printf("Loomcore C API smoke test (ABI version %u)\n", loomcore_c_abi_version());

    LoomcoreRuntime* rt = loomcore_runtime_create();
    if (!rt) {
        fprintf(stderr, "loomcore_runtime_create failed: %s\n", loomcore_c_last_error());
        return 1;
    }

    if (loomcore_runtime_set_binder(rt, "a", identity_binder, NULL) != 0 ||
        loomcore_runtime_set_binder(rt, "b", double_binder, NULL) != 0) {
        fprintf(stderr, "loomcore_runtime_set_binder failed: %s\n", loomcore_c_last_error());
        loomcore_runtime_destroy(rt);
        return 1;
    }

    if (loomcore_runtime_load_graph(rt, config_path) != LOOMCORE_C_OK) {
        fprintf(stderr, "loomcore_runtime_load_graph failed: %s\n", loomcore_c_last_error());
        loomcore_runtime_destroy(rt);
        return 1;
    }

    float x_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    int64_t x_shape[2] = {1, 4};
    LoomcoreCTensor input;
    input.name = "x";
    input.shape = x_shape;
    input.shape_len = 2;
    input.dtype = LOOMCORE_C_FLOAT32;
    input.data = x_data;
    input.element_count = 4;

    LoomcoreResultSet* result = NULL;
    LoomcoreCStatus status = loomcore_runtime_run(rt, &input, 1, /*time_budget_ms=*/-1.0, &result);
    if (status != LOOMCORE_C_OK) {
        fprintf(stderr, "loomcore_runtime_run failed (status=%d): %s\n", (int)status, loomcore_c_last_error());
        loomcore_runtime_destroy(rt);
        return 1;
    }

    size_t n = loomcore_result_count(result);
    printf("job produced %zu sink tensor(s)\n", n);

    int ok = (n == 1);
    if (ok) {
        const char* node_id = NULL;
        LoomcoreCTensor out;
        if (loomcore_result_get(result, 0, &node_id, &out) != 0) {
            ok = 0;
        } else {
            printf("sink node '%s': [", node_id);
            const float expected[4] = {2.0f, 4.0f, 6.0f, 8.0f};
            const float* data = (const float*)out.data;
            for (size_t i = 0; i < out.element_count; ++i) {
                printf("%.1f%s", data[i], i + 1 < out.element_count ? ", " : "");
                if (out.element_count != 4 || data[i] != expected[i]) ok = 0;
            }
            printf("]\n");
        }
    }

    double p50 = 0, p95 = 0, mean = 0;
    size_t count = 0;
    if (loomcore_node_stats(rt, "a", &p50, &p95, &mean, &count) == 0) {
        printf("node 'a' stats: p50=%.4fms p95=%.4fms mean=%.4fms n=%zu\n", p50, p95, mean, count);
    }

    loomcore_result_destroy(result);
    loomcore_runtime_destroy(rt);

    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
