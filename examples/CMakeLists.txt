# stb_image / stb_image_resize2 are fetched only for this example target —
# decoding a real JPEG/PNG and resizing it to MobileNetV2's 224x224 input
# is "real-world edge inference input handling" the example wants to show,
# but it has nothing to do with Loomcore's own orchestration API, so it
# stays out of loomcore_core entirely.
loomcore_fetch_single_header(STB_IMAGE_INCLUDE_DIR "stb"
    "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h"
    "stb_image.h")
loomcore_fetch_single_header(STB_RESIZE_INCLUDE_DIR "stb"
    "https://raw.githubusercontent.com/nothings/stb/master/stb_image_resize2.h"
    "stb_image_resize2.h")

add_executable(loomcore_example run_example.cpp)
target_link_libraries(loomcore_example PRIVATE loomcore::core)
target_include_directories(loomcore_example PRIVATE "${STB_IMAGE_INCLUDE_DIR}" "${STB_RESIZE_INCLUDE_DIR}")
target_compile_definitions(loomcore_example PRIVATE
    LOOMCORE_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    LOOMCORE_PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
)
if(MSVC)
    target_compile_options(loomcore_example PRIVATE /W4)
endif()

# Proves the RCU graph hot-swap (docs/ARCHITECTURE.md "Hot-reloading a
# graph") under real concurrent load — see reload_demo.cpp's own header
# comment. Needs the real reference models, same as loomcore_example.
add_executable(loomcore_reload_demo reload_demo.cpp)
target_link_libraries(loomcore_reload_demo PRIVATE loomcore::core)
target_compile_definitions(loomcore_reload_demo PRIVATE LOOMCORE_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
if(MSVC)
    target_compile_options(loomcore_reload_demo PRIVATE /W4)
endif()
