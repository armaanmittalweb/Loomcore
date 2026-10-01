# Loomcore installed-package consumer

Proves `find_package(Loomcore REQUIRED)` works against a real `cmake --install`
tree — not merely "the example builds inside Loomcore's own CMake tree",
which vendoring the source into another project would already give you
for free and which was, before Phase 05, the *only* way to consume
Loomcore. See [docs/CLAIMS.md](../../docs/CLAIMS.md) #14.

This directory is a genuinely separate CMake project — it is **not**
added as a subdirectory of the main build (check the top-level
`CMakeLists.txt`; it isn't there) — configured on its own against
wherever you install Loomcore.

## Reproduce it yourself

From the repo root, with Loomcore already built (`docs/BUILD.md`):

```
cmake --install build --config RelWithDebInfo --prefix /some/install/prefix
```

Then, from this directory, configure and build against *only* that
install tree:

```
cd examples/consumer
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH=/some/install/prefix
cmake --build build --config RelWithDebInfo
```

(Linux/macOS: drop `-G "Visual Studio 17 2022" -A x64`.)

On success, `find_package(Loomcore REQUIRED)` resolved, `loomcore_consumer`
linked against the exported `loomcore::loomcore_core` target (see this
directory's `CMakeLists.txt` for why that's the target name, not
`loomcore::core` — CMake's `install(EXPORT)` namespaces the library's
*real* target name, not the in-repo convenience `ALIAS` the rest of this
repo's own CMakeLists.txt files use), and built with zero references to
Loomcore's source tree.

## Running it

Windows has no rpath equivalent, so `loomcore_core.dll` and
`onnxruntime.dll` (installed to `<prefix>/bin`) need to be next to the
executable or on `PATH` — the same DLL-staging requirement every other
executable in this repo already has (see the top-level `CMakeLists.txt`'s
shared output directory comment). On Linux, `CMAKE_INSTALL_RPATH` (also
set in the top-level `CMakeLists.txt`) makes this unnecessary.

```
# Windows, from this directory, after the build above:
copy /some/install/prefix\bin\*.dll build\RelWithDebInfo\
build\RelWithDebInfo\loomcore_consumer.exe
```

Expected output:

```
Loomcore consumed via find_package(Loomcore): Runtime constructed OK.
```

`main.cpp` deliberately does the minimum that actually exercises the
packaging claim: include a header found via the installed package,
link via `find_package`, and construct a real `loomcore::Runtime` (which
forces ONNX Runtime environment initialization, so a missing or
misplaced `onnxruntime.dll`/`.so` at the install prefix fails here, not
silently). It never calls `loadGraph()` — proving the *package* is
consumable doesn't need the reference ONNX models present.
