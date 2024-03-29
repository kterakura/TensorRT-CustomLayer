cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(depthwiseConvolutionPlugin LANGUAGES CXX CUDA)

# tensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
  find_library(TENSORRT_LIBRARY_PARSER nvparsers
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_PARSER})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
INCLUDE(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR
    "Cannot find TensorRT library.")
endif()

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# trt_l2norm_plugin library
add_library(depthwiseConvolutionPlugin SHARED
  depthwiseConvolutionPlugin.cpp
  depthwiseConvolutionPluginKernel.cu
)
target_link_libraries(depthwiseConvolutionPlugin
  ${TENSORRT_LIBRARY}
)
target_compile_features(depthwiseConvolutionPlugin PUBLIC cxx_std_11)
# TODO: I did not found a proper way to autoselect cuda compute capability
# with cmake 3.10. Set 7.2 for xavier. Otherwise breaks fp16 functions.
target_compile_options(depthwiseConvolutionPlugin
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -gencode arch=compute_75,code=sm_75>
)