cmake_minimum_required(VERSION 3.14)
project(bench_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED)

add_executable(bench_cpp bench_cpp.cpp)
target_link_libraries(bench_cpp PRIVATE openvino::runtime)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
