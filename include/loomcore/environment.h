// Loomcore — process-wide ONNX Runtime environment handle.
//
// Kept as an opaque wrapper (rather than exposing Ort::Env in any public
// header) so that nothing outside src/model_node.cpp and src/runtime.cpp
// needs to see the ONNX Runtime C++ API or link against it directly —
// the ORT dependency is fully encapsulated inside loomcore_core. Consumers
// (examples, the benchmark, Python bindings) only ever see
// loomcore::Environment / loomcore::ModelVariant / loomcore::Runtime.
#pragma once

#include <memory>

#include "loomcore/export.h"

namespace loomcore {

class LOOMCORE_API Environment {
public:
    // One Ort::Env per process is the documented ONNX Runtime usage
    // pattern; this returns that single instance, created on first use and
    // kept alive for the remainder of the process.
    static Environment& shared();

    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;

    // Opaque `Ort::Env*`, cast back internally by model_node.cpp. Not
    // meaningful to code that doesn't link ONNX Runtime itself.
    void* nativeHandle() const;

private:
    Environment();
    ~Environment();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace loomcore
