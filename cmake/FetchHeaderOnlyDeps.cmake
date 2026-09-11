# Fetches the two single-header dependencies Loomcore uses
# (nlohmann/json for config + log serialization, doctest for tests) by
# downloading just the header rather than cloning the full repositories —
# both projects publish their whole library as one file specifically for
# this kind of consumption.

function(loomcore_fetch_single_header OUT_INCLUDE_DIR NAME URL RELATIVE_HEADER_PATH)
    set(_dir "${CMAKE_BINARY_DIR}/_deps/${NAME}")
    set(_header "${_dir}/${RELATIVE_HEADER_PATH}")
    if(NOT EXISTS "${_header}")
        get_filename_component(_header_subdir "${_header}" DIRECTORY)
        file(MAKE_DIRECTORY "${_header_subdir}")
        message(STATUS "Loomcore: downloading ${NAME} header")
        file(DOWNLOAD "${URL}" "${_header}" STATUS _status)
        list(GET _status 0 _code)
        if(NOT _code EQUAL 0)
            list(GET _status 1 _msg)
            file(REMOVE "${_header}")
            message(FATAL_ERROR "Loomcore: failed to download ${NAME} from ${URL} (${_msg})")
        endif()
    endif()
    set(${OUT_INCLUDE_DIR} "${_dir}" PARENT_SCOPE)
endfunction()
