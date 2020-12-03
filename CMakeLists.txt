cmake_minimum_required(VERSION 3.17)
project(mastermind)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

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

add_executable(mastermind_metal
        gpu_interface_wrapper.hpp
        gpu_interface_wrapper.mm
        main.cpp
        score.cpp
        utils.hpp
        utils.cpp)
