cmake_minimum_required(VERSION 3.17)
project(mastermind CXX CUDA)
#project(mastermind CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

set(CMAKE_CUDA_STANDARD 20)

include_directories(.)

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

add_executable(mastermind
        main.cpp
        score.cpp
        utils.hpp
        utils.cpp
        cuda_gpu_interface.cu
        )

set_target_properties(mastermind PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "86")

target_compile_options(mastermind PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        --relocatable-device-code=true
        >)

# https://developer.nvidia.com/blog/building-cuda-applications-cmake/
