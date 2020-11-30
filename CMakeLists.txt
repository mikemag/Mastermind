cmake_minimum_required(VERSION 3.17)
project(mastermind)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

include_directories(.)

add_executable(mastermind_metal
        gpu_interface_wrapper.hpp
        gpu_interface_wrapper.mm
        main.cpp
        score.cpp
        utils.hpp
        utils.cpp)
