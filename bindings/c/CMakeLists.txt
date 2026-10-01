# Loomcore C API smoke test — see docs/CLAIMS.md #13. Deliberately its own
# CMakeLists (rather than folding into examples/) since it needs the C
# language enabled and exists purely to prove include/loomcore/c_api.h is
# consumable from a pure-C translation unit against the compiled shared
# library, not to demonstrate application code.
enable_language(C)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/smoke_test_config.json.in"
    "${CMAKE_CURRENT_BINARY_DIR}/smoke_test_config.json"
    @ONLY
)

add_executable(loomcore_c_smoke_test smoke_test.c)
target_link_libraries(loomcore_c_smoke_test PRIVATE loomcore::core)
set_target_properties(loomcore_c_smoke_test PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
target_compile_definitions(loomcore_c_smoke_test PRIVATE
    LOOMCORE_C_SMOKE_TEST_CONFIG="${CMAKE_CURRENT_BINARY_DIR}/smoke_test_config.json"
)

add_test(NAME loomcore_c_smoke_test
         COMMAND loomcore_c_smoke_test "${CMAKE_CURRENT_BINARY_DIR}/smoke_test_config.json")
