# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.16)
project(bev_deploy LANGUAGES C CXX CUDA)

# --- General Configuration ---
set(TARGET "x86_64" CACHE STRING "Target architecture (e.g., x86_64, aarch64)")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "-fPIC -O0 -g")

set(CMAKE_CXX_FLAGS "-fPIC -O0 -g -Wno-deprecated-declarations")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 --expt-relaxed-constexpr --compiler-options -fPIC -O2 -Xptxas=-v")


# --- Target-Specific Configuration ---
if(${TARGET} MATCHES "aarch64")
    # aarch64 cross-compilation setup
    set(CMAKE_C_COMPILER   /usr/bin/aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
    set(CMAKE_FIND_ROOT_PATH "/usr/aarch64-linux-gnu")
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.8/")
    set(CUDA_HOST_COMPILER "/usr/bin/aarch64-linux-gnu-g++" CACHE STRING "Host compiler" FORCE)

    set(TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu)
    set(TENSORRT_LIBRARY_DIRS /usr/lib/aarch64-linux-gnu)
    set(AARCH64_CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}targets/aarch64-linux/include)
    set(AARCH64_CUDA_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}targets/aarch64-linux/lib)

    link_directories(
        /drive/drive-linux/lib-target
        ${AARCH64_CUDA_LIBRARY_DIRS}
        ${AARCH64_CUDA_LIBRARY_DIRS}/stubs
        ${TENSORRT_LIBRARY_DIRS}
    )

    set(ARCH 87)
    set(PLATFORM_LIBS nvcudla cudla cudart)
    find_package(CUDA REQUIRED)

elseif(${TARGET} MATCHES "x86_64")
    # x86_64 native build setup
    if(DEFINED ENV{TRT_ROOT})
        set(TRT_ROOT "$ENV{TRT_ROOT}")
        message(STATUS "Using TensorRT from environment variable: ${TRT_ROOT}")
    else()
        message(FATAL_ERROR "TRT_ROOT environment variable is not set. Please point it to your TensorRT installation.")
    endif()

    find_package(CUDA 11.8 REQUIRED)

    set(ARCH 86)
    set(TENSORRT_INCLUDE_DIRS ${TRT_ROOT}/include)
    set(TENSORRT_LIBRARY_DIRS ${TRT_ROOT}/lib)
    link_directories(${TENSORRT_LIBRARY_DIRS})

else()
    message(FATAL_ERROR "TARGET [${TARGET}] is not a recognized value ('aarch64' or 'x86_64').")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_INCLUDE_DIRS}")


# --- Library Definitions ---

# CUDA Kernels (Static Library)
set(plugins_cu_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_scale_deform_attn/ms_deform_attn_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/rotate/rotateKernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/select_and_pad/select_and_pad_kernel.cu
)
add_library(plugins_cu STATIC ${plugins_cu_srcs})

set_target_properties(plugins_cu PROPERTIES CUDA_ARCHITECTURES "${ARCH}")
set_target_properties(plugins_cu PROPERTIES CUDA_STANDARD 14)
set_target_properties(plugins_cu PROPERTIES POSITION_INDEPENDENT_CODE ON)


# C++ Plugin Logic (Shared Library)
set(plugins_cpp_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_scale_deform_attn/ms_deform_attn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rotate/rotate_plugin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/select_and_pad/select_and_pad.cpp
)
add_library(plugins SHARED ${plugins_cpp_srcs})


# --- Include Directories ---
target_include_directories(plugins_cu PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernel
    ${TENSORRT_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
target_include_directories(plugins PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernel
    ${TENSORRT_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
# For aarch64, add the target-specific include path as well
if(${TARGET} MATCHES "aarch64")
    target_include_directories(plugins_cu PRIVATE ${AARCH64_CUDA_INCLUDE_DIRS})
    target_include_directories(plugins PRIVATE ${AARCH64_CUDA_INCLUDE_DIRS})
endif()


# --- Linking ---
target_link_libraries(plugins PRIVATE
    plugins_cu
    nvinfer
)

# Link platform-specific libraries
if(PLATFORM_LIBS)
    target_link_libraries(plugins PRIVATE ${PLATFORM_LIBS})
endif()

# Link cuBLAS
find_package(CUDAToolkit QUIET)
if (CUDAToolkit_FOUND AND TARGET CUDA::cublas)
    # Modern CMake: link imported target
    target_link_libraries(plugins PRIVATE CUDA::cublas)
elseif(EXISTS "${CUDA_CUBLAS_LIBRARIES}")
    # Legacy CMake: link library variable
    target_link_libraries(plugins PRIVATE ${CUDA_CUBLAS_LIBRARIES})
else()
    message(WARNING "Could not find cuBLAS. Linking may fail.")
endif()