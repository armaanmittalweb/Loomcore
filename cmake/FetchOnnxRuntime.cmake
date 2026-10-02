# Downloads a prebuilt ONNX Runtime C++/C distribution for the host
# platform and defines an IMPORTED target `onnxruntime::onnxruntime`.
#
# Building ONNX Runtime from source is a multi-hour, multi-gigabyte
# undertaking that is completely orthogonal to what this project
# demonstrates, so — like every other prebuilt-binary C++ dependency in
# the ecosystem (libtorch, CUDA, etc.) — Loomcore fetches Microsoft's
# official release archive instead. Set ONNXRUNTIME_ROOT_DIR to point at
# an existing install (e.g. one already unpacked, or one from your
# package manager) to skip this entirely.

function(loomcore_fetch_onnxruntime OUT_ROOT_DIR)
    set(_ort_version "1.30.0")

    if(WIN32)
        set(_ort_pkg "onnxruntime-win-x64-${_ort_version}")
        set(_ort_ext "zip")
    elseif(APPLE)
        set(_ort_pkg "onnxruntime-osx-arm64-${_ort_version}")
        set(_ort_ext "tgz")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
        set(_ort_pkg "onnxruntime-linux-aarch64-${_ort_version}")
        set(_ort_ext "tgz")
    else()
        set(_ort_pkg "onnxruntime-linux-x64-${_ort_version}")
        set(_ort_ext "tgz")
    endif()

    set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${_ort_version}/${_ort_pkg}.${_ort_ext}")
    set(_cache_dir "${CMAKE_BINARY_DIR}/_deps/onnxruntime")
    set(_marker "${_cache_dir}/${_ort_pkg}/.loomcore_extracted")

    if(NOT EXISTS "${_marker}")
        set(_downloads_dir "${CMAKE_BINARY_DIR}/_deps/downloads")
        file(MAKE_DIRECTORY "${_downloads_dir}")
        file(MAKE_DIRECTORY "${_cache_dir}")
        set(_archive "${_downloads_dir}/${_ort_pkg}.${_ort_ext}")

        message(STATUS "Loomcore: downloading ONNX Runtime ${_ort_version} (${_ort_pkg}) — cached after the first run")
        file(DOWNLOAD "${_ort_url}" "${_archive}" SHOW_PROGRESS STATUS _dl_status)
        list(GET _dl_status 0 _dl_code)
        if(NOT _dl_code EQUAL 0)
            list(GET _dl_status 1 _dl_msg)
            file(REMOVE "${_archive}")
            message(FATAL_ERROR
                "Loomcore: failed to download ONNX Runtime from ${_ort_url} (${_dl_msg}).\n"
                "Set -DONNXRUNTIME_ROOT_DIR=<path to an existing ONNX Runtime install> to skip auto-download.")
        endif()

        message(STATUS "Loomcore: extracting ${_ort_pkg}.${_ort_ext}")
        file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${_cache_dir}")
        file(WRITE "${_marker}" "ok")
    endif()

    set(${OUT_ROOT_DIR} "${_cache_dir}/${_ort_pkg}" PARENT_SCOPE)
endfunction()
