cmake_minimum_required(VERSION 3.14)
project(openvino_yolo_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# OpenVINO 的 pip 包自带完整的 C++ SDK（include/ + libs/ + cmake/），
# 不必另外下载 C++ 发行版：把 OpenVINO_DIR 指到 site-packages/openvino/cmake 即可。
# 本程序刻意不依赖 OpenCV C++（opencv-python 的 wheel 不提供头文件/cmake config），
# 前处理的双线性 resize、letterbox、NMS 全部手写。
find_package(OpenVINO REQUIRED)

# 本机（OpenCore 上的 macOS）的 clang 不会自动搜索 libc++ 头文件目录，
# 表现为 #include <set> 报 'file not found'（numpy 源码编译也是同样原因失败的）。
# 显式把 SDK 里的 c++/v1 加进系统搜索路径即可。
if(APPLE)
  execute_process(COMMAND xcrun --show-sdk-path
                  OUTPUT_VARIABLE _sdk OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_sdk)
    set(CMAKE_OSX_SYSROOT "${_sdk}")
    include_directories(SYSTEM "${_sdk}/usr/include/c++/v1")
    link_directories("${_sdk}/usr/lib")
  endif()
endif()

add_executable(yolo_infer_cpp yolo_infer_cpp.cpp)
target_link_libraries(yolo_infer_cpp PRIVATE openvino::runtime)

# 纯推理计时器（只测模型，不含前后处理），可选
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench_cpp.cpp)
  add_executable(bench_cpp bench_cpp.cpp)
  target_link_libraries(bench_cpp PRIVATE openvino::runtime)
endif()
