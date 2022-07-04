cmake_minimum_required(VERSION 3.18)
project(mastermind)

execute_process(
        COMMAND git rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
        COMMAND git log -1 --format=%cd --date=iso
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_DATE
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

configure_file(
        "${PROJECT_SOURCE_DIR}/mastermind_config.h.in"
        "${PROJECT_SOURCE_DIR}/mastermind_config.h"
)

set(MASTERMIND_SOURCES_LIST
        ${PROJECT_SOURCE_DIR}/main.cpp
        ${PROJECT_SOURCE_DIR}/score.cpp
        ${PROJECT_SOURCE_DIR}/utils.cpp
        new_algo.cpp new_algo.h)

add_executable(mastermind ${MASTERMIND_SOURCES_LIST})
target_include_directories(mastermind PUBLIC SYSTEM .)
set_target_properties(mastermind PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        )
target_compile_options(mastermind PRIVATE
        -march=native
        -fno-omit-frame-pointer
        )

# https://github.com/anrieff/libcpuid to get CPU info like number of processors, L1 cache size, etc.
# On Ubuntu: sudo apt install cpuidtool libcpuid15 libcpuid-dev
find_library(LIBCPUID_LIBRARY NAMES libcpuid.so)
find_path(LIBCPUID_INCLUDE_DIR NAMES libcpuid)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBCPUID
        FOUND_VAR LIBCPUID_FOUND
        REQUIRED_VARS LIBCPUID_LIBRARY LIBCPUID_INCLUDE_DIR
        )
if (LIBCPUID_FOUND AND NOT TARGET LIBCPUID)
    add_library(LIBCPUID UNKNOWN IMPORTED)
    set_target_properties(LIBCPUID PROPERTIES
            IMPORTED_LOCATION "${LIBCPUID_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${LIBCPUID_INCLUDE_DIR}/libcpuid"
            )
    target_link_libraries(mastermind PUBLIC LIBCPUID)
endif ()

include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set_source_files_properties(${MASTERMIND_SOURCES_LIST} PROPERTIES LANGUAGE CUDA)
    set_target_properties(mastermind PROPERTIES
            CXX_STANDARD 17
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "86")
    target_compile_options(mastermind PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --generate-line-info
            --use_fast_math
            --relocatable-device-code=true
            --default-stream per-thread
            -fno-omit-frame-pointer
            --extended-lambda
            -dc
            --std=c++17
            --forward-unknown-to-host-compiler -march=native
            >)
endif ()
