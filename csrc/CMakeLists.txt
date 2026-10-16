# =============================================================================
# C++/CUDA scheduler core for llm-inference-engine
#
# Builds a small static library (the Continuous Batching core + a reference CPU
# engine) and, when pybind11 is available, a Python extension ``infer_engine._C``.
#
#   cmake -S csrc -B csrc/build -DLLM_ENGINE_BUILD_CUDA=OFF
#   cmake --build csrc/build -j
#
# Inside the TensorRT container:
#   cmake -S csrc -B csrc/build -DLLM_ENGINE_BUILD_CUDA=ON
# =============================================================================
cmake_minimum_required(VERSION 3.18)
project(infer_engine LANGUAGES CXX)

option(LLM_ENGINE_BUILD_CUDA "Compile CUDA/TensorRT engine path" OFF)
option(LLM_ENGINE_BUILD_PYTHON "Build the pybind11 extension" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Core static library -------------------------------------------------- #
add_library(infer_engine_core STATIC
  src/engine.cpp
  src/batching.cpp
)
target_include_directories(infer_engine_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(infer_engine_core PRIVATE -O3)

if(LLM_ENGINE_BUILD_CUDA)
  message(STATUS "CUDA/TensorRT path enabled")
  find_package(CUDAToolkit REQUIRED)
  # TensorRT headers/libs are expected from the base image (TRT_ROOT).
  set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT install root")
  find_library(NVINFER_LIB nvinfer HINTS ${TENSORRT_ROOT}/lib)
  target_compile_definitions(infer_engine_core PUBLIC WITH_CUDA=1)
  target_link_libraries(infer_engine_core PUBLIC CUDA::cudart)
  if(NVINFER_LIB)
    target_link_libraries(infer_engine_core PUBLIC ${NVINFER_LIB})
  endif()
endif()

# ---- Python extension ----------------------------------------------------- #
if(LLM_ENGINE_BUILD_PYTHON)
  # Locate pybind11 either via the installed package or a CMake config.
  find_package(pybind11 CONFIG QUIET)
  if(pybind11_FOUND)
    pybind11_add_module(_C
      src/bindings.cpp
    )
    target_link_libraries(_C PRIVATE infer_engine_core)
  else()
    message(STATUS "pybind11 not found; skipping Python extension (core lib still built).")
  endif()
endif()
