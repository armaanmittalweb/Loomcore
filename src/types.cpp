#include "loomcore/types.h"

#include <algorithm>
#include <numeric>

namespace loomcore {

const char* toString(Backend b) {
    switch (b) {
        case Backend::CPU: return "CPU";
        case Backend::GPU_SIM: return "GPU_SIM";
    }
    return "UNKNOWN_BACKEND";
}

const char* toString(Precision p) {
    switch (p) {
        case Precision::FP32: return "FP32";
        case Precision::INT8: return "INT8";
    }
    return "UNKNOWN_PRECISION";
}

const char* toString(DType d) {
    switch (d) {
        case DType::Float32: return "float32";
        case DType::Int64: return "int64";
        case DType::Int32: return "int32";
    }
    return "unknown_dtype";
}

NamedTensor NamedTensor::makeFloat(std::string name, std::vector<int64_t> shape, std::vector<float> data) {
    NamedTensor t;
    t.name = std::move(name);
    t.shape = std::move(shape);
    t.dtype = DType::Float32;
    t.f32 = std::move(data);
    return t;
}

NamedTensor NamedTensor::makeInt64(std::string name, std::vector<int64_t> shape, std::vector<int64_t> data) {
    NamedTensor t;
    t.name = std::move(name);
    t.shape = std::move(shape);
    t.dtype = DType::Int64;
    t.i64 = std::move(data);
    return t;
}

NamedTensor NamedTensor::makeInt32(std::string name, std::vector<int64_t> shape, std::vector<int32_t> data) {
    NamedTensor t;
    t.name = std::move(name);
    t.shape = std::move(shape);
    t.dtype = DType::Int32;
    t.i32 = std::move(data);
    return t;
}

size_t NamedTensor::elementCount() const {
    if (shape.empty()) return 1;
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d < 0 ? 0 : d);
    return n;
}

int64_t NamedTensor::batchDim() const {
    if (shape.empty()) {
        throw LoomcoreError("NamedTensor::batchDim: tensor '" + name + "' has no shape");
    }
    return shape[0];
}

namespace {
size_t innerStride(const std::vector<int64_t>& shape) {
    // Number of elements in one "row" along axis 0.
    size_t stride = 1;
    for (size_t i = 1; i < shape.size(); ++i) stride *= static_cast<size_t>(shape[i]);
    return stride;
}
} // namespace

NamedTensor NamedTensor::sliceBatch(int64_t begin, int64_t end) const {
    if (shape.empty() || end < begin || end > shape[0] || begin < 0) {
        throw LoomcoreError("NamedTensor::sliceBatch: invalid range for tensor '" + name + "'");
    }
    NamedTensor out;
    out.name = name;
    out.shape = shape;
    out.shape[0] = end - begin;
    out.dtype = dtype;
    const size_t stride = innerStride(shape);
    const size_t off = static_cast<size_t>(begin) * stride;
    const size_t count = static_cast<size_t>(end - begin) * stride;
    switch (dtype) {
        case DType::Float32:
            out.f32.assign(f32.begin() + off, f32.begin() + off + count);
            break;
        case DType::Int64:
            out.i64.assign(i64.begin() + off, i64.begin() + off + count);
            break;
        case DType::Int32:
            out.i32.assign(i32.begin() + off, i32.begin() + off + count);
            break;
    }
    return out;
}

NamedTensor concatBatch(const std::vector<NamedTensor>& parts) {
    if (parts.empty()) {
        throw LoomcoreError("concatBatch: no parts to concatenate");
    }
    NamedTensor out;
    out.name = parts.front().name;
    out.dtype = parts.front().dtype;
    out.shape = parts.front().shape;
    if (out.shape.empty()) {
        throw LoomcoreError("concatBatch: tensor '" + out.name + "' has no shape");
    }
    int64_t totalBatch = 0;
    for (const auto& p : parts) {
        if (p.dtype != out.dtype) {
            throw LoomcoreError("concatBatch: dtype mismatch for tensor '" + out.name + "'");
        }
        if (p.shape.size() != out.shape.size() ||
            !std::equal(p.shape.begin() + 1, p.shape.end(), out.shape.begin() + 1)) {
            throw LoomcoreError("concatBatch: non-batch dimension mismatch for tensor '" + out.name + "'");
        }
        totalBatch += p.shape[0];
    }
    out.shape[0] = totalBatch;
    const size_t stride = innerStride(out.shape);
    switch (out.dtype) {
        case DType::Float32:
            out.f32.reserve(static_cast<size_t>(totalBatch) * stride);
            for (const auto& p : parts) out.f32.insert(out.f32.end(), p.f32.begin(), p.f32.end());
            break;
        case DType::Int64:
            out.i64.reserve(static_cast<size_t>(totalBatch) * stride);
            for (const auto& p : parts) out.i64.insert(out.i64.end(), p.i64.begin(), p.i64.end());
            break;
        case DType::Int32:
            out.i32.reserve(static_cast<size_t>(totalBatch) * stride);
            for (const auto& p : parts) out.i32.insert(out.i32.end(), p.i32.begin(), p.i32.end());
            break;
    }
    return out;
}

std::pair<int64_t, float> argmax(const NamedTensor& t) {
    if (t.dtype != DType::Float32 || t.f32.empty()) {
        throw LoomcoreError("argmax: expected a non-empty float32 tensor");
    }
    int64_t bestIdx = 0;
    float bestVal = t.f32[0];
    for (size_t i = 1; i < t.f32.size(); ++i) {
        if (t.f32[i] > bestVal) {
            bestVal = t.f32[i];
            bestIdx = static_cast<int64_t>(i);
        }
    }
    return {bestIdx, bestVal};
}

} // namespace loomcore
