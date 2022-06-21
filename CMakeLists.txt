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

add_executable(mastermind-cpu
        main.cpp
        score.cpp
        utils.hpp
        utils.cpp
        )
target_include_directories(mastermind-cpu PUBLIC .)
set_target_properties(mastermind-cpu PROPERTIES CXX_STANDARD 20)
target_compile_options(mastermind-cpu PRIVATE -march=native)


include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_executable(mastermind-cuda
            main.cpp
            score.cpp
            utils.cpp
            cuda_gpu_interface.cu
            )
    target_compile_definitions(mastermind-cuda PUBLIC MASTERMIND_CUDA)
    target_include_directories(mastermind-cuda PUBLIC .)
    set_target_properties(mastermind-cuda PROPERTIES
            CXX_STANDARD 20
            CMAKE_CUDA_STANDARD 17
            CMAKE_CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "86")
    target_compile_options(mastermind-cuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
    target_compile_options(mastermind-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --generate-line-info
            --use_fast_math
            --relocatable-device-code=true
            --std=c++17
            --forward-unknown-to-host-compiler -march=native
            >)

    # https://developer.nvidia.com/blog/building-cuda-4applications-cmake/
endif ()
