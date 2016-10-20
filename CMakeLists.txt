cmake_minimum_required(VERSION 2.8)

project(CUDA_Class)

find_package(CUDA REQUIRED)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/CMakeModules")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  list(APPEND CMAKE_CXX_FLAGS "-std=c++0x -Wall")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  list(APPEND CMAKE_CXX_FLAGS "-std=c++0x -Wall")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  list(APPEND CMAKE_CXX_FLAGS "/W4")
endif()

set(src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(sources
    ${src_dir}/main.cu)

cuda_add_executable(CUDA_Class ${sources})