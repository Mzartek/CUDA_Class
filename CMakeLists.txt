cmake_minimum_required(VERSION 2.8)

project(CUDA_Class)

find_package(CUDA REQUIRED)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/CMakeModules")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Wno-deprecated-gpu-targets)

set(src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(sources
    ${src_dir}/Helpers.h
    ${src_dir}/Particle.h
    ${src_dir}/Vec.h
    ${src_dir}/main.cu
    ${src_dir}/tp.h
    ${src_dir}/tp.cu)

cuda_add_executable(CUDA_Class ${sources})