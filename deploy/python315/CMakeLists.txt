cmake_minimum_required(VERSION 3.31)
project(TensorRTPython315 LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE BINDING_SOURCES "${TRT_SOURCE}/python/src/*.cpp")
pybind11_add_module(tensorrt MODULE NO_EXTRAS ${BINDING_SOURCES})
target_compile_features(tensorrt PRIVATE cxx_std_20)
target_compile_definitions(tensorrt PRIVATE tensorrt_EXPORTS=1 TENSORRT_MODULE=tensorrt)
target_include_directories(tensorrt PRIVATE
    "${TRT_SOURCE}/include" "${TRT_SOURCE}/python/include"
    "${TRT_SOURCE}/python/docstrings" "${TRT_SOURCE}/parsers/onnx"
    "${CUDA_INCLUDE}")
target_link_libraries(tensorrt PRIVATE
    "${TRT_LIBS}/libnvinfer.so.11"
    "${TRT_LIBS}/libnvinfer_plugin.so.11"
    "${TRT_LIBS}/libnvonnxparser.so.11")
set_target_properties(tensorrt PROPERTIES PREFIX "" SUFFIX ".so"
    BUILD_RPATH "$ORIGIN/../tensorrt_libs"
    INSTALL_RPATH "$ORIGIN/../tensorrt_libs")
