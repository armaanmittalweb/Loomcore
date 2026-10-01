#include "loomcore/model_node.h"

#include <onnxruntime_cxx_api.h>

#include <atomic>
#include <cstring>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace loomcore {

// ---------------------------------------------------------------------------
// NodeExecutionContext
// ---------------------------------------------------------------------------

const NamedTensor& NodeExecutionContext::upstreamTensor(const std::string& node_id,
                                                          const std::string& tensor_name) const {
    if (upstream_outputs == nullptr) {
        throw LoomcoreError("NodeExecutionContext: no upstream outputs available");
    }
    auto it = upstream_outputs->find(node_id);
    if (it == upstream_outputs->end() || !it->second) {
        throw LoomcoreError("NodeExecutionContext: upstream node '" + node_id + "' has no recorded output");
    }
    for (const auto& t : *it->second) {
        if (t.name == tensor_name) return t;
    }
    throw LoomcoreError("NodeExecutionContext: upstream node '" + node_id + "' has no tensor named '" +
                         tensor_name + "'");
}

const NamedTensor& NodeExecutionContext::upstreamFirst(const std::string& node_id) const {
    if (upstream_outputs == nullptr) {
        throw LoomcoreError("NodeExecutionContext: no upstream outputs available");
    }
    auto it = upstream_outputs->find(node_id);
    if (it == upstream_outputs->end() || !it->second || it->second->empty()) {
        throw LoomcoreError("NodeExecutionContext: upstream node '" + node_id + "' produced no outputs");
    }
    return it->second->front();
}

// ---------------------------------------------------------------------------
// NodeConfig
// ---------------------------------------------------------------------------

const VariantConfig* NodeConfig::variant(Precision p) const {
    for (const auto& v : variants) {
        if (v.precision == p) return &v;
    }
    return nullptr;
}

bool NodeConfig::hasVariant(Precision p) const { return variant(p) != nullptr; }

// ---------------------------------------------------------------------------
// Path + tensor <-> Ort::Value conversion helpers
// ---------------------------------------------------------------------------

namespace {

#if defined(_WIN32)
std::wstring toOrtPath(const std::string& utf8) {
    if (utf8.empty()) return std::wstring();
    int needed = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), static_cast<int>(utf8.size()), nullptr, 0);
    std::wstring wide(static_cast<size_t>(needed), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), static_cast<int>(utf8.size()), wide.data(), needed);
    return wide;
}
using OrtPathString = std::wstring;
#else
using OrtPathString = std::string;
std::string toOrtPath(const std::string& utf8) { return utf8; }
#endif

Ort::Value toOrtValue(const Ort::MemoryInfo& mem_info, const NamedTensor& t) {
    switch (t.dtype) {
        case DType::Float32:
            return Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(t.f32.data()), t.f32.size(),
                                                     t.shape.data(), t.shape.size());
        case DType::Int64:
            return Ort::Value::CreateTensor<int64_t>(mem_info, const_cast<int64_t*>(t.i64.data()), t.i64.size(),
                                                       t.shape.data(), t.shape.size());
        case DType::Int32:
            return Ort::Value::CreateTensor<int32_t>(mem_info, const_cast<int32_t*>(t.i32.data()), t.i32.size(),
                                                       t.shape.data(), t.shape.size());
    }
    throw LoomcoreError("toOrtValue: unsupported dtype for tensor '" + t.name + "'");
}

NamedTensor fromOrtValue(const std::string& name, Ort::Value& v) {
    auto info = v.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = info.GetShape();
    // A dynamic axis reported as -1 by the model graph is meaningless on a
    // concrete output; ORT always resolves it to the real extent, which
    // GetShape() already returns for a materialized output value.
    size_t count = info.GetElementCount();
    ONNXTensorElementDataType elem_type = info.GetElementType();
    switch (elem_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            const float* data = v.GetTensorData<float>();
            return NamedTensor::makeFloat(name, shape, std::vector<float>(data, data + count));
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            const int64_t* data = v.GetTensorData<int64_t>();
            return NamedTensor::makeInt64(name, shape, std::vector<int64_t>(data, data + count));
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
            const int32_t* data = v.GetTensorData<int32_t>();
            return NamedTensor::makeInt32(name, shape, std::vector<int32_t>(data, data + count));
        }
        default:
            throw LoomcoreError("fromOrtValue: unsupported ONNX output element type for tensor '" + name + "'");
    }
}

} // namespace

// ---------------------------------------------------------------------------
// CancellationToken
// ---------------------------------------------------------------------------

// Wraps one Ort::RunOptions: SetTerminate()/UnsetTerminate() are documented
// as safe to call from a thread other than the one inside Run(), which is
// exactly the usage pattern here (a scheduler-owned deadline reaper calling
// requestCancel() while a lane worker is inside ModelVariant::run()).
struct CancellationToken::Impl {
    Ort::RunOptions run_options;
    std::atomic<bool> cancelled{false};
};

CancellationToken::CancellationToken() : impl_(std::make_unique<Impl>()) {}
CancellationToken::~CancellationToken() = default;

void CancellationToken::requestCancel() {
    impl_->cancelled.store(true, std::memory_order_release);
    impl_->run_options.SetTerminate();
}

bool CancellationToken::cancelled() const { return impl_->cancelled.load(std::memory_order_acquire); }

void* CancellationToken::nativeHandle() const { return &impl_->run_options; }

// ---------------------------------------------------------------------------
// ModelVariant
// ---------------------------------------------------------------------------

struct ModelVariant::Impl {
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo mem_info{nullptr};

    Impl() : mem_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}
};

ModelVariant::ModelVariant(Environment& env, const std::string& model_path, Precision precision,
                            int intra_op_threads)
    : impl_(std::make_unique<Impl>()), precision_(precision) {
    impl_->session_options.SetIntraOpNumThreads(intra_op_threads);
    impl_->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    auto* ort_env = reinterpret_cast<Ort::Env*>(env.nativeHandle());
    OrtPathString path = toOrtPath(model_path);
    try {
        impl_->session = std::make_unique<Ort::Session>(*ort_env, path.c_str(), impl_->session_options);
    } catch (const Ort::Exception& ex) {
        throw LoomcoreError("ModelVariant: failed to load '" + model_path + "': " + ex.what());
    }

    Ort::AllocatorWithDefaultOptions allocator;
    const size_t n_in = impl_->session->GetInputCount();
    const size_t n_out = impl_->session->GetOutputCount();
    input_names_.reserve(n_in);
    output_names_.reserve(n_out);
    for (size_t i = 0; i < n_in; ++i) {
        auto name = impl_->session->GetInputNameAllocated(i, allocator);
        input_names_.emplace_back(name.get());
    }
    for (size_t i = 0; i < n_out; ++i) {
        auto name = impl_->session->GetOutputNameAllocated(i, allocator);
        output_names_.emplace_back(name.get());
    }
}

ModelVariant::~ModelVariant() = default;
ModelVariant::ModelVariant(ModelVariant&&) noexcept = default;
ModelVariant& ModelVariant::operator=(ModelVariant&&) noexcept = default;

std::vector<NamedTensor> ModelVariant::run(const std::vector<NamedTensor>& inputs, CancellationToken* cancel) const {
    if (inputs.size() != input_names_.size()) {
        throw LoomcoreError("ModelVariant::run: expected " + std::to_string(input_names_.size()) +
                             " inputs, got " + std::to_string(inputs.size()));
    }
    std::vector<const char*> in_names, out_names;
    in_names.reserve(input_names_.size());
    out_names.reserve(output_names_.size());
    for (const auto& n : input_names_) in_names.push_back(n.c_str());
    for (const auto& n : output_names_) out_names.push_back(n.c_str());

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.reserve(inputs.size());
    for (const auto& t : inputs) ort_inputs.push_back(toOrtValue(impl_->mem_info, t));

    // If a cancellation token is bound, run with *its* Ort::RunOptions
    // (an already-cancelled token has typically already called
    // SetTerminate(), so ORT will abort this run essentially immediately)
    // so a concurrent requestCancel() from another thread can abort this
    // call in flight. Otherwise use a fresh, never-cancelled RunOptions —
    // identical to the pre-cancellation-support behavior.
    Ort::RunOptions local_opts;
    Ort::RunOptions* opts = cancel ? reinterpret_cast<Ort::RunOptions*>(cancel->nativeHandle()) : &local_opts;

    std::vector<Ort::Value> ort_outputs;
    try {
        ort_outputs = impl_->session->Run(*opts, in_names.data(), ort_inputs.data(), ort_inputs.size(),
                                           out_names.data(), out_names.size());
    } catch (const Ort::Exception& ex) {
        if (cancel && cancel->cancelled()) {
            throw LoomcoreError("ModelVariant::run: cancelled (deadline exceeded): " + std::string(ex.what()));
        }
        throw;
    }

    std::vector<NamedTensor> results;
    results.reserve(ort_outputs.size());
    for (size_t i = 0; i < ort_outputs.size(); ++i) {
        results.push_back(fromOrtValue(output_names_[i], ort_outputs[i]));
    }
    return results;
}

// ---------------------------------------------------------------------------
// ModelNode
// ---------------------------------------------------------------------------

std::shared_ptr<ModelVariant> ModelNode::variant(Precision p) const {
    auto it = variants.find(p);
    return it == variants.end() ? nullptr : it->second;
}

} // namespace loomcore
