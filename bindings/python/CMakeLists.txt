find_package(Python3 COMPONENTS Interpreter Development QUIET)
if(NOT Python3_FOUND)
    message(WARNING "Loomcore: Python3 development headers not found; skipping Python bindings")
    return()
endif()

# Prefer the pybind11 CMake config that ships with `pip install pybind11`
# (discovered via its own --cmakedir helper); fall back to fetching the
# source if pybind11 isn't installed for this interpreter.
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_dir_result
    ERROR_QUIET
)
if(_pybind11_dir_result EQUAL 0)
    find_package(pybind11 CONFIG QUIET)
endif()
if(NOT pybind11_FOUND)
    message(STATUS "Loomcore: pybind11 not found via pip install; fetching source instead")
    include(FetchContent)
    FetchContent_Declare(pybind11_src GIT_REPOSITORY https://github.com/pybind/pybind11.git GIT_TAG v2.13.6)
    FetchContent_MakeAvailable(pybind11_src)
endif()

pybind11_add_module(_loomcore loomcore_py.cpp)
target_link_libraries(_loomcore PRIVATE loomcore::core)
target_compile_definitions(_loomcore PRIVATE LOOMCORE_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
