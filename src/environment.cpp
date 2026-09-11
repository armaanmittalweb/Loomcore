#include "loomcore/environment.h"

#include <onnxruntime_cxx_api.h>

namespace loomcore {

struct Environment::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "loomcore"};
};

Environment::Environment() : impl_(std::make_unique<Impl>()) {}
Environment::~Environment() = default;

Environment& Environment::shared() {
    static Environment instance;
    return instance;
}

void* Environment::nativeHandle() const { return static_cast<void*>(&impl_->env); }

} // namespace loomcore
