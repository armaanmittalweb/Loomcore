// Loomcore Python bindings — a thin pybind11 wrapper around the C++
// runtime. Tensors cross the boundary as numpy arrays; DAG node binders
// and confidence extractors are plain Python callables, adapted to
// loomcore::NodeInputBinder / loomcore::ConfidenceExtractor here; a
// routing policy can be one of the built-in C++ policies OR a Python
// class subclassing `loomcore.RoutingPolicy` (a pybind11 trampoline),
// which is Milestone 4's "basic router logic" made callable from Python
// while still running inside the C++ scheduler on every job.
//
// See bindings/python/README.md for the Python-facing API and
// docs/ARCHITECTURE.md "Python bindings" for the GIL-handling details
// that make it safe to call back into Python from scheduler worker
// threads.
#include "loomcore/runtime.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>

namespace py = pybind11;
using namespace loomcore;

namespace {

// ---------------------------------------------------------------------------
// NamedTensor <-> numpy
// ---------------------------------------------------------------------------

NamedTensor tensorFromNumpy(const std::string& name, py::array arr) {
    std::vector<int64_t> shape(static_cast<size_t>(arr.ndim()));
    for (py::ssize_t i = 0; i < arr.ndim(); ++i) shape[static_cast<size_t>(i)] = static_cast<int64_t>(arr.shape(i));
    size_t count = static_cast<size_t>(arr.size());
    char kind = arr.dtype().kind();

    if (kind == 'f') {
        py::array_t<float, py::array::c_style | py::array::forcecast> typed(arr);
        std::vector<float> data(typed.data(), typed.data() + count);
        return NamedTensor::makeFloat(name, shape, std::move(data));
    }
    if (kind == 'i' || kind == 'u') {
        if (arr.dtype().itemsize() == 4 && kind == 'i') {
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> typed(arr);
            std::vector<int32_t> data(typed.data(), typed.data() + count);
            return NamedTensor::makeInt32(name, shape, std::move(data));
        }
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> typed(arr);
        std::vector<int64_t> data(typed.data(), typed.data() + count);
        return NamedTensor::makeInt64(name, shape, std::move(data));
    }
    throw std::runtime_error("loomcore: unsupported numpy dtype for tensor '" + name +
                              "' (expected a float or integer array)");
}

py::array tensorToNumpy(const NamedTensor& t) {
    std::vector<py::ssize_t> shape(t.shape.begin(), t.shape.end());
    switch (t.dtype) {
        case DType::Float32: {
            py::array_t<float> arr(shape);
            std::copy(t.f32.begin(), t.f32.end(), arr.mutable_data());
            return arr;
        }
        case DType::Int64: {
            py::array_t<int64_t> arr(shape);
            std::copy(t.i64.begin(), t.i64.end(), arr.mutable_data());
            return arr;
        }
        case DType::Int32: {
            py::array_t<int32_t> arr(shape);
            std::copy(t.i32.begin(), t.i32.end(), arr.mutable_data());
            return arr;
        }
    }
    throw std::runtime_error("loomcore: tensor with unknown dtype cannot be converted to numpy");
}

TensorMap dictToTensorMap(const py::dict& d) {
    TensorMap tm;
    for (auto item : d) {
        std::string key = item.first.cast<std::string>();
        tm[key] = tensorFromNumpy(key, item.second.cast<py::array>());
    }
    return tm;
}

py::dict jobResultToDict(const JobResult& r) {
    py::dict d;
    for (const auto& [node_id, tensors] : r) {
        py::list lst;
        for (const auto& t : tensors) lst.append(tensorToNumpy(t));
        d[py::str(node_id)] = lst;
    }
    return d;
}

// ---------------------------------------------------------------------------
// Python-callable adapters for NodeInputBinder / ConfidenceExtractor
// ---------------------------------------------------------------------------

// The Python binder signature is:
//   fn(graph_inputs: Dict[str, np.ndarray], upstream_outputs: Dict[str, List[np.ndarray]])
//       -> List[Tuple[str, np.ndarray]]
// Returning (name, array) pairs (rather than bare arrays) lets the C++
// side match them to the model's real input names exactly the way a
// C++-authored NodeInputBinder does (see loomcore::reorderToExpected in
// scheduler.cpp) regardless of the order the Python author builds them in.
NodeInputBinder makePyBinder(py::function fn) {
    return [fn](const NodeExecutionContext& ctx) -> std::vector<NamedTensor> {
        py::gil_scoped_acquire gil;

        py::dict graph_inputs;
        if (ctx.graph_inputs != nullptr) {
            for (const auto& [k, v] : *ctx.graph_inputs) graph_inputs[py::str(k)] = tensorToNumpy(v);
        }
        py::dict upstream;
        if (ctx.upstream_outputs != nullptr) {
            for (const auto& [k, vec] : *ctx.upstream_outputs) {
                py::list lst;
                for (const auto& t : vec) lst.append(tensorToNumpy(t));
                upstream[py::str(k)] = lst;
            }
        }

        py::list result = fn(graph_inputs, upstream).cast<py::list>();
        std::vector<NamedTensor> out;
        out.reserve(result.size());
        for (auto item : result) {
            py::tuple pair = item.cast<py::tuple>();
            if (pair.size() != 2) {
                throw std::runtime_error("loomcore: a Python node binder must return a list of (name, ndarray) pairs");
            }
            out.push_back(tensorFromNumpy(pair[0].cast<std::string>(), pair[1].cast<py::array>()));
        }
        return out;
    };
}

// Python signature: fn(upstream_outputs: List[np.ndarray]) -> Optional[float]
ConfidenceExtractor makePyConfidence(py::function fn) {
    return [fn](const std::vector<NamedTensor>& upstream) -> std::optional<float> {
        py::gil_scoped_acquire gil;
        py::list lst;
        for (const auto& t : upstream) lst.append(tensorToNumpy(t));
        py::object result = fn(lst);
        if (result.is_none()) return std::nullopt;
        return result.cast<float>();
    };
}

// ---------------------------------------------------------------------------
// A reduced, value-typed view of RoutingContext safe to hand to Python
// (the real RoutingContext holds raw pointers into scheduler-owned state
// that has no business being exposed across the language boundary).
// ---------------------------------------------------------------------------
struct PyRoutingContext {
    std::string job_id;
    std::string node_id;
    size_t cpu_queue_depth = 0;
    size_t gpu_queue_depth = 0;
    double time_budget_remaining_ms = -1.0;
    // list[np.ndarray] mirroring the node's confidence_source upstream
    // output when the node config set one, else None. Exposed directly
    // (rather than only through the separate confidence_extractors
    // mechanism) so a Python-authored RoutingPolicy can compute its own
    // gating logic inline, the way examples/run_example.py's
    // SkipIfConfident does.
    py::object upstream_confidence_source = py::none();
};

// Trampoline so a Python class can subclass loomcore.RoutingPolicy and
// implement decide() (name() is optional to override). Every call
// acquires the GIL, since this runs on whichever scheduler lane thread is
// dispatching the node in question — never the thread that called
// Runtime.run() from Python.
class PyRoutingPolicy : public IRoutingPolicy {
public:
    using IRoutingPolicy::IRoutingPolicy;

    std::string name() const override {
        py::gil_scoped_acquire gil;
        py::function override_fn = py::get_override(this, "name");
        return override_fn ? override_fn().cast<std::string>() : std::string("PythonRoutingPolicy");
    }

    std::optional<RoutingDecision> decide(const RoutingContext& ctx) const override {
        py::gil_scoped_acquire gil;

        PyRoutingContext view;
        view.job_id = ctx.job_id;
        view.node_id = ctx.node ? ctx.node->id : std::string();
        view.cpu_queue_depth = ctx.cpu_queue_depth;
        view.gpu_queue_depth = ctx.gpu_queue_depth;
        view.time_budget_remaining_ms = ctx.time_budget_remaining_ms;
        if (ctx.upstream_confidence_source != nullptr) {
            py::list lst;
            for (const auto& t : *ctx.upstream_confidence_source) lst.append(tensorToNumpy(t));
            view.upstream_confidence_source = std::move(lst);
        }

        py::function override_fn = py::get_override(this, "decide");
        if (!override_fn) {
            throw std::runtime_error("loomcore.RoutingPolicy subclasses must implement decide(self, ctx)");
        }
        py::object result = override_fn(view);
        if (result.is_none()) return std::nullopt;
        auto decision = result.cast<RoutingDecision>();
        decision.policy_name = name();
        return decision;
    }
};

} // namespace

PYBIND11_MODULE(_loomcore, m) {
    m.doc() = "Loomcore: multi-model execution orchestrator for edge inference (C++ core, Python bindings)";

    py::enum_<Backend>(m, "Backend")
        .value("CPU", Backend::CPU)
        .value("GPU_SIM", Backend::GPU_SIM);

    py::enum_<Precision>(m, "Precision")
        .value("FP32", Precision::FP32)
        .value("INT8", Precision::INT8);

    py::class_<PyRoutingContext>(m, "RoutingContext")
        .def_readonly("job_id", &PyRoutingContext::job_id)
        .def_readonly("node_id", &PyRoutingContext::node_id)
        .def_readonly("cpu_queue_depth", &PyRoutingContext::cpu_queue_depth)
        .def_readonly("gpu_queue_depth", &PyRoutingContext::gpu_queue_depth)
        .def_readonly("time_budget_remaining_ms", &PyRoutingContext::time_budget_remaining_ms)
        .def_readonly("upstream_confidence_source", &PyRoutingContext::upstream_confidence_source);

    py::class_<RoutingDecision>(m, "RoutingDecision")
        .def(py::init([](std::optional<Precision> precision, std::optional<Backend> backend, bool skip,
                          std::string reason) {
                 RoutingDecision d;
                 d.precision = precision;
                 d.backend = backend;
                 d.skip = skip;
                 d.reason = std::move(reason);
                 d.policy_name = "python";
                 return d;
             }),
             py::arg("precision") = py::none(), py::arg("backend") = py::none(), py::arg("skip") = false,
             py::arg("reason") = std::string())
        .def_readwrite("precision", &RoutingDecision::precision)
        .def_readwrite("backend", &RoutingDecision::backend)
        .def_readwrite("skip", &RoutingDecision::skip)
        .def_readwrite("reason", &RoutingDecision::reason);

    // Deliberately no .def("decide", ...) / .def("name", ...) here: nothing
    // calls those from Python directly. The C++ scheduler invokes them
    // through the ordinary virtual-dispatch vtable (shared_ptr<IRoutingPolicy>),
    // which the PyRoutingPolicy trampoline already routes into Python's
    // override of decide()/name() — see its definition above. Binding them
    // as callable Python methods would additionally require exposing the
    // real (pointer-heavy) RoutingContext type, which is exactly what
    // PyRoutingContext exists to avoid.
    py::class_<IRoutingPolicy, PyRoutingPolicy, std::shared_ptr<IRoutingPolicy>>(m, "RoutingPolicy")
        .def(py::init<>());

    py::class_<LatencyBudgetPolicy, IRoutingPolicy, std::shared_ptr<LatencyBudgetPolicy>>(m, "LatencyBudgetPolicy")
        .def(py::init<double>(), py::arg("threshold_ms"));

    py::class_<LoadAwareBackendPolicy, IRoutingPolicy, std::shared_ptr<LoadAwareBackendPolicy>>(
        m, "LoadAwareBackendPolicy")
        .def(py::init<>());

    py::class_<CompositeRouter, IRoutingPolicy, std::shared_ptr<CompositeRouter>>(m, "CompositeRouter")
        .def(py::init<>())
        // keep_alive<1, 2>: a Python policy object stored only via the
        // C++ shared_ptr (e.g. `router.add(MyPolicy())` with no surviving
        // Python reference) would otherwise have its Python wrapper
        // garbage-collected as soon as add() returns — the C++ object
        // stays alive (kept by the shared_ptr) but PyRoutingPolicy's
        // get_override() then finds no Python object to call into. Tying
        // the policy's lifetime to the CompositeRouter's (arg 1 = self)
        // fixes it.
        .def("add", &CompositeRouter::add, py::arg("policy"), py::keep_alive<1, 2>());

    py::class_<Runtime>(m, "Runtime")
        .def(py::init<>())
        .def(
            "load_graph",
            [](Runtime& self, const std::string& config_path, py::dict binders, py::object confidence_extractors,
               py::object router) {
                std::map<std::string, NodeInputBinder> cpp_binders;
                for (auto item : binders) {
                    cpp_binders[item.first.cast<std::string>()] = makePyBinder(item.second.cast<py::function>());
                }
                std::map<std::string, ConfidenceExtractor> cpp_confidence;
                if (!confidence_extractors.is_none()) {
                    for (auto item : confidence_extractors.cast<py::dict>()) {
                        cpp_confidence[item.first.cast<std::string>()] =
                            makePyConfidence(item.second.cast<py::function>());
                    }
                }
                std::shared_ptr<IRoutingPolicy> cpp_router;
                if (!router.is_none()) cpp_router = router.cast<std::shared_ptr<IRoutingPolicy>>();

                py::gil_scoped_release release; // loading walks the graph and loads ONNX sessions; no Python needed
                self.loadGraph(config_path, cpp_binders, cpp_confidence, cpp_router);
            },
            py::arg("config_path"), py::arg("binders"), py::arg("confidence_extractors") = py::none(),
            py::arg("router") = py::none(),
            // keep_alive<1, 5>: self=1, ... , router=5th argument — same
            // rationale as CompositeRouter::add's keep_alive (see there).
            py::keep_alive<1, 5>(),
            "Load a DAG from a JSON config (see examples/graph_config.json). `binders` maps node id -> "
            "callable(graph_inputs: dict, upstream_outputs: dict) -> list[(name, ndarray)].")
        .def(
            "run",
            [](Runtime& self, py::dict inputs, int priority, double time_budget_ms) {
                TensorMap tm = dictToTensorMap(inputs);
                JobResult result;
                {
                    py::gil_scoped_release release; // let other Python threads run while the C++ job executes
                    result = self.run(std::move(tm), priority, time_budget_ms);
                }
                return jobResultToDict(result);
            },
            py::arg("inputs"), py::arg("priority") = 0, py::arg("time_budget_ms") = -1.0,
            "Runs one graph job to completion and returns {sink_node_id: [ndarray, ...]}.")
        .def(
            "node_stats",
            [](Runtime& self, const std::string& node_id) {
                auto s = self.nodeStats(node_id);
                py::dict d;
                d["p50_ms"] = s.p50_ms;
                d["p95_ms"] = s.p95_ms;
                d["mean_ms"] = s.mean_ms;
                d["count"] = s.count;
                return d;
            },
            py::arg("node_id"))
        .def(
            "recent_logs", [](Runtime& self, size_t n) { return self.recentLogs(n); }, py::arg("n") = 100,
            "Returns up to the last n structured log lines (JSON strings) — parse with Python's json module.");
}
