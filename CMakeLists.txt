cmake_minimum_required(VERSION 3.20)
project(Loomcore VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

option(LOOMCORE_BUILD_TESTS "Build the doctest unit tests" ON)
option(LOOMCORE_BUILD_EXAMPLES "Build the example DAG pipeline executable" ON)
option(LOOMCORE_BUILD_BENCHMARKS "Build the FP32 vs INT8 latency benchmark" ON)
option(LOOMCORE_BUILD_PYTHON_BINDINGS "Build the pybind11 Python extension module" ON)
option(LOOMCORE_DOWNLOAD_ONNXRUNTIME "Auto-download a prebuilt ONNX Runtime if ONNXRUNTIME_ROOT_DIR is unset" ON)
set(ONNXRUNTIME_ROOT_DIR "" CACHE PATH "Path to an existing ONNX Runtime install (skips auto-download)")

# ---------------------------------------------------------------------------
# All targets share one output directory (per config, for multi-config
# generators like Visual Studio) so that a shared library sitting next to
# an executable is always found without extra PATH/RPATH juggling on
# Windows, and via $ORIGIN rpath on Linux/macOS.
# ---------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
foreach(_cfg ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER "${_cfg}" _cfg_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper} "${CMAKE_BINARY_DIR}/bin/${_cfg}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_cfg_upper} "${CMAKE_BINARY_DIR}/bin/${_cfg}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_cfg_upper} "${CMAKE_BINARY_DIR}/lib/${_cfg}")
endforeach()

if(UNIX)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_RPATH "$ORIGIN")
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
if(NOT ONNXRUNTIME_ROOT_DIR)
    if(NOT LOOMCORE_DOWNLOAD_ONNXRUNTIME)
        message(FATAL_ERROR "ONNXRUNTIME_ROOT_DIR is not set and LOOMCORE_DOWNLOAD_ONNXRUNTIME is OFF.")
    endif()
    include(cmake/FetchOnnxRuntime.cmake)
    loomcore_fetch_onnxruntime(ONNXRUNTIME_ROOT_DIR)
endif()
message(STATUS "Loomcore: using ONNX Runtime at ${ONNXRUNTIME_ROOT_DIR}")

add_library(onnxruntime SHARED IMPORTED GLOBAL)
target_include_directories(onnxruntime INTERFACE "${ONNXRUNTIME_ROOT_DIR}/include")
if(WIN32)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_ROOT_DIR}/lib/onnxruntime.dll"
        IMPORTED_IMPLIB "${ONNXRUNTIME_ROOT_DIR}/lib/onnxruntime.lib"
    )
elseif(APPLE)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.dylib"
    )
else()
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so"
    )
endif()

include(cmake/FetchHeaderOnlyDeps.cmake)
loomcore_fetch_single_header(NLOHMANN_JSON_INCLUDE_DIR "nlohmann_json"
    "https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp"
    "nlohmann/json.hpp")
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE "${NLOHMANN_JSON_INCLUDE_DIR}")
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)

# ---------------------------------------------------------------------------
# Core shared library
# ---------------------------------------------------------------------------
add_library(loomcore_core SHARED
    src/types.cpp
    src/graph.cpp
    src/environment.cpp
    src/model_node.cpp
    src/metrics.cpp
    src/logger.cpp
    src/router.cpp
    src/scheduler.cpp
    src/runtime.cpp
    src/tokenizer.cpp
)
add_library(loomcore::core ALIAS loomcore_core)

target_include_directories(loomcore_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_definitions(loomcore_core
    PRIVATE LOOMCORE_BUILD_SHARED
    PUBLIC LOOMCORE_USE_SHARED
)
target_link_libraries(loomcore_core
    PRIVATE onnxruntime nlohmann_json::nlohmann_json
)
if(MSVC)
    target_compile_options(loomcore_core PRIVATE /W4 /permissive-)
    # C4251/C4275: "needs to have dll-interface" for STL members/bases of
    # exported classes. Unavoidable when a shared library's public API
    # uses std:: types (the standard, accepted C++-on-Windows tradeoff),
    # and safe here since every consumer in this repo links the exact
    # loomcore_core.dll built alongside it with the same toolchain/CRT.
    # PUBLIC so consumers see the same suppression when including these headers.
    target_compile_options(loomcore_core PUBLIC /wd4251 /wd4275)
else()
    target_compile_options(loomcore_core PRIVATE -Wall -Wextra)
endif()

# The ONNX Runtime shared library is a private link dependency (nothing in
# Loomcore's public headers exposes ORT types — see loomcore/environment.h)
# but loomcore_core.dll/.so still needs it to be *found* by the OS loader
# at run time. One copy into the shared output directory covers every
# consumer target (examples, tests, benchmark, Python extension).
add_custom_command(TARGET loomcore_core POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "$<TARGET_FILE:onnxruntime>" "$<TARGET_FILE_DIR:loomcore_core>"
    COMMENT "Loomcore: staging ONNX Runtime shared library next to loomcore_core"
)

# ---------------------------------------------------------------------------
# Subprojects
# ---------------------------------------------------------------------------
if(LOOMCORE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(LOOMCORE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(LOOMCORE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(LOOMCORE_BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings/python)
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "Loomcore configuration summary")
message(STATUS "  C++ standard          : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Generator             : ${CMAKE_GENERATOR}")
message(STATUS "  ONNX Runtime          : ${ONNXRUNTIME_ROOT_DIR}")
message(STATUS "  Build tests           : ${LOOMCORE_BUILD_TESTS}")
message(STATUS "  Build examples        : ${LOOMCORE_BUILD_EXAMPLES}")
message(STATUS "  Build benchmarks      : ${LOOMCORE_BUILD_BENCHMARKS}")
message(STATUS "  Build Python bindings : ${LOOMCORE_BUILD_PYTHON_BINDINGS}")
message(STATUS "")
