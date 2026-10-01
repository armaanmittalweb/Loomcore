loomcore_fetch_single_header(DOCTEST_INCLUDE_DIR "doctest"
    "https://raw.githubusercontent.com/doctest/doctest/v2.4.11/doctest/doctest.h"
    "doctest/doctest.h")
add_library(doctest INTERFACE)
target_include_directories(doctest INTERFACE "${DOCTEST_INCLUDE_DIR}")

add_executable(loomcore_tests
    test_main.cpp
    test_types.cpp
    test_graph.cpp
    test_router.cpp
    test_tokenizer.cpp
    test_scheduler.cpp
    test_scheduler_deadlines.cpp
    test_planner.cpp
    test_runtime_reload.cpp
    test_perfetto_export.cpp
)
target_link_libraries(loomcore_tests PRIVATE loomcore::core doctest nlohmann_json::nlohmann_json)
target_compile_definitions(loomcore_tests PRIVATE
    LOOMCORE_TEST_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../assets"
)

add_test(NAME loomcore_tests COMMAND loomcore_tests)
