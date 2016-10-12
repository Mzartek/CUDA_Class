cmake_minimum_required(VERSION 3.0.2)

project(CUDA_Class)

find_package(CUDA REQUIRED)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/CMakeModules")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  add_definitions(-std=c++14 -Wall)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_definitions(-std=c++14 -Wall)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  add_definitions(/W4)
endif()

set(src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(sources
    ${src_dir}/main.cu)

cuda_add_executable(CUDA_Class ${sources})