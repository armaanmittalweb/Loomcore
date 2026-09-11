// Loomcore — shared-library visibility macros.
//
// Loomcore is built as a shared library on both targeted platforms
// (a .dll under MSVC, a .so under GCC/Clang). LOOMCORE_API expands to
// the correct import/export/visibility attribute depending on which
// side of the library boundary the including translation unit is on.
//
// The core CMake target defines LOOMCORE_BUILD_SHARED (PRIVATE) while
// compiling loomcore itself; consumers (examples, benchmarks, tests,
// the Python extension) get LOOMCORE_USE_SHARED defined for them by
// the loomcore::core target's usage requirements.
#pragma once

#if defined(_WIN32) || defined(_WIN64)
    #if defined(LOOMCORE_BUILD_SHARED)
        #define LOOMCORE_API __declspec(dllexport)
    #elif defined(LOOMCORE_USE_SHARED)
        #define LOOMCORE_API __declspec(dllimport)
    #else
        #define LOOMCORE_API
    #endif
#else
    #if defined(LOOMCORE_BUILD_SHARED) || defined(LOOMCORE_USE_SHARED)
        #define LOOMCORE_API __attribute__((visibility("default")))
    #else
        #define LOOMCORE_API
    #endif
#endif
