// Loomcore — core value types shared across the runtime.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "loomcore/export.h"

namespace loomcore {

// Logical compute lane a node is dispatched to. There is no physical GPU in
// the reference environment for this project: GPU_SIM is a *simulated*
// heterogeneous backend — a separate worker pool with different batching
// aggressiveness and an injected transfer-latency cost model — so the
// scheduler and router have two genuinely different cost profiles to reason
// about, the way they would with a real CPU/GPU split. See
// docs/ARCHITECTURE.md "Simulated backends" for exactly what is and isn't
// simulated.
enum class Backend { CPU, GPU_SIM };

// Numeric precision of a model variant. A logical DAG node may register
// both an FP32 and an INT8 variant of the same model; the router picks
// between them at run time (see loomcore/router.h).
enum class Precision { FP32, INT8 };

// Element type carried by a NamedTensor. Intentionally small: it covers
// exactly what the two reference models need (float image tensors for
// MobileNetV2, int64 token id / mask tensors for BERT-tiny) rather than
// mirroring the full ONNX type system.
enum class DType { Float32, Int64, Int32 };

LOOMCORE_API const char* toString(Backend b);
LOOMCORE_API const char* toString(Precision p);
LOOMCORE_API const char* toString(DType d);

// Defined so these scoped enums have a sensible default ostream
// representation (used by anything that streams them, doctest's
// CHECK()-failure printer included — without this, comparing enum class
// values in a doctest assertion fails to compile).
inline std::ostream& operator<<(std::ostream& os, Backend b) { return os << toString(b); }
inline std::ostream& operator<<(std::ostream& os, Precision p) { return os << toString(p); }
inline std::ostream& operator<<(std::ostream& os, DType d) { return os << toString(d); }

// A single named, typed, shaped tensor with owned contiguous storage.
// Exactly one of the data vectors is populated, selected by `dtype`.
struct LOOMCORE_API NamedTensor {
    std::string name;
    std::vector<int64_t> shape;
    DType dtype = DType::Float32;

    std::vector<float> f32;
    std::vector<int64_t> i64;
    std::vector<int32_t> i32;

    static NamedTensor makeFloat(std::string name, std::vector<int64_t> shape, std::vector<float> data);
    static NamedTensor makeInt64(std::string name, std::vector<int64_t> shape, std::vector<int64_t> data);
    static NamedTensor makeInt32(std::string name, std::vector<int64_t> shape, std::vector<int32_t> data);

    // Product of `shape`. Empty shape (scalar) yields 1.
    size_t elementCount() const;

    // Number of entries along axis 0 (the batch axis by convention
    // throughout Loomcore). Requires a non-empty shape.
    int64_t batchDim() const;

    // Returns a tensor holding only the [begin, end) slice along axis 0.
    // Used to split a batched model output back into per-request results.
    NamedTensor sliceBatch(int64_t begin, int64_t end) const;
};

// Concatenates `parts` along axis 0 into one batched tensor. All parts must
// share name, dtype, and every non-batch dimension. Used to merge several
// queued requests for the same node into a single inference call.
LOOMCORE_API NamedTensor concatBatch(const std::vector<NamedTensor>& parts);

using TensorMap = std::map<std::string, NamedTensor>;

// argmax over a 1-D (or single-row 2-D, row-major) float tensor. Returns
// {index, value}. Used by the reference pipeline's confidence-gated
// routing policy and by the example's label decoding.
LOOMCORE_API std::pair<int64_t, float> argmax(const NamedTensor& t);

class LOOMCORE_API LoomcoreError : public std::runtime_error {
public:
    explicit LoomcoreError(const std::string& what) : std::runtime_error(what) {}
};

// Surfaced through a job's future — thrown when runSync() is used, or when
// the caller .get()s/.wait()s the future submitJob() returned — when
// SchedulerConfig::enable_admission_control is on and PrecisionPlanner
// judges the requested time budget undeliverable even after downgrading
// every eligible critical-path node. submitJob() itself never throws this
// synchronously: it returns an already-failed future immediately, before
// any node has been dispatched. Catchable separately from a mid-run
// failure (LoomcoreError) so a caller can distinguish "you never even
// started my job" from "my job started and then something went wrong".
class LOOMCORE_API JobRejectedError : public LoomcoreError {
public:
    explicit JobRejectedError(const std::string& what) : LoomcoreError(what) {}
};

// Thrown (via the job's future) when SchedulerConfig::enable_deadline_cancellation
// is on and a job's time budget elapsed before it completed. Distinguishes
// "the deadline reaper had to step in and cancel in-flight work" from an
// ordinary node-level failure.
class LOOMCORE_API DeadlineExceededError : public LoomcoreError {
public:
    explicit DeadlineExceededError(const std::string& what) : LoomcoreError(what) {}
};

} // namespace loomcore
