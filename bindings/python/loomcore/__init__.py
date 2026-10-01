"""Loomcore Python bindings: a thin wrapper around the compiled `_loomcore`
pybind11 extension (see bindings/python/loomcore_py.cpp). The runtime
itself is C++; this package exists for tooling, testing, and scripting
against it — not as an alternate implementation.

The compiled extension isn't pip-packaged yet (see README.md "Python
bindings" for why and the intended follow-up), so this module locates it
next to whichever CMake build produced it: pass the build directory via
the LOOMCORE_BUILD_DIR environment variable, or just have a `build/`
directory in the repo root (the default `cmake -B build` layout) with the
extension under `build/bin/<config>/`.
"""
import os
import sys


def _find_build_bin_dir():
    configs = ("", "Release", "RelWithDebInfo", "Debug", "MinSizeRel")
    candidates = []
    env = os.environ.get("LOOMCORE_BUILD_DIR")
    if env:
        for config in configs:
            candidates.append(os.path.join(env, "bin", config))

    here = os.path.dirname(os.path.abspath(__file__))
    # here = <repo>/bindings/python/loomcore, so the repo root is three levels up.
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    for build_name in ("build", "out/build"):
        for config in configs:
            candidates.append(os.path.join(repo_root, build_name, "bin", config))

    for c in candidates:
        if os.path.isdir(c) and any(name.startswith("_loomcore") for name in os.listdir(c)):
            return c
    return None


_bin_dir = _find_build_bin_dir()
if _bin_dir:
    # Windows needs the ONNX Runtime / loomcore_core DLLs on the loader's
    # search path in addition to sys.path finding the .pyd itself.
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_bin_dir)
    sys.path.insert(0, _bin_dir)

try:
    from _loomcore import (  # noqa: E402  (path must be adjusted first)
        Backend,
        BulkheadPolicy,
        CircuitBreakerPolicy,
        CompositeRouter,
        ConfidenceGatePolicy,
        DeadlineExceededError,
        JobRejectedError,
        LatencyBudgetPolicy,
        LoadAwareBackendPolicy,
        LoomcoreError,
        PlannedPrecisionPolicy,
        Precision,
        RoutingContext,
        RoutingDecision,
        RoutingPolicy,
        Runtime,
        RuntimeOptions,
        SchedulerConfig,
        WordPieceTokenizer,
        export_perfetto_trace,
    )
except ImportError as exc:  # pragma: no cover - exercised only in a broken env
    raise ImportError(
        "Could not import the compiled _loomcore extension.\n"
        "Build it first: cmake -S . -B build && cmake --build build --config Release\n"
        "(requires -DLOOMCORE_BUILD_PYTHON_BINDINGS=ON, the default), then either set "
        "LOOMCORE_BUILD_DIR=<path to that build dir> or run from the repo root so "
        "build/bin/ is found automatically."
    ) from exc

__all__ = [
    "Backend",
    "Precision",
    "Runtime",
    "RuntimeOptions",
    "SchedulerConfig",
    "RoutingContext",
    "RoutingDecision",
    "RoutingPolicy",
    "LatencyBudgetPolicy",
    "LoadAwareBackendPolicy",
    "ConfidenceGatePolicy",
    "PlannedPrecisionPolicy",
    "CircuitBreakerPolicy",
    "BulkheadPolicy",
    "CompositeRouter",
    "WordPieceTokenizer",
    "export_perfetto_trace",
    "LoomcoreError",
    "JobRejectedError",
    "DeadlineExceededError",
]
