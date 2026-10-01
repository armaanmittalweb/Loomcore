# Loomcore installed-package consumer — see this directory's README.md and
# docs/CLAIMS.md #14. This is a genuinely standalone CMake project: it is
# NOT added as a subdirectory of the main build (grep the top-level
# CMakeLists.txt — it isn't there). It is configured on its own, against
# an *installed* Loomcore (via -DCMAKE_PREFIX_PATH=<install prefix>), to
# prove find_package(Loomcore) is real rather than only "it builds inside
# this repo's own CMake tree".
cmake_minimum_required(VERSION 3.20)
project(LoomcoreConsumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Loomcore REQUIRED)

add_executable(loomcore_consumer main.cpp)
# The exported target is loomcore::loomcore_core (CMake's install(EXPORT)
# namespaces the *real* target name, loomcore_core — it doesn't know about
# the in-repo convenience ALIAS loomcore::core that the rest of this
# repo's own CMakeLists.txt files use internally).
target_link_libraries(loomcore_consumer PRIVATE loomcore::loomcore_core)
