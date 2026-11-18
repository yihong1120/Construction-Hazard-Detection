#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <nvrtc.h>
#include "preprocess_source.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error;
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Logger : nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::fprintf(stderr, "[TRT] %s\n", message);
    }
};
Logger logger;
struct Engine {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    void* input = nullptr;
    void* output = nullptr;
    void* bytes = nullptr;
    float* lut = nullptr;
    size_t capacity;
    std::string input_name, output_name;
    std::vector<void*> bindings;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    Engine(const void* data, size_t size, const float* values, size_t max_elements) : capacity(max_elements) {
        cv::setNumThreads(1);
        initLibNvInferPlugins(&logger, "");
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime) throw std::runtime_error("createInferRuntime failed");
        engine.reset(runtime->deserializeCudaEngine(data, size));
        if (!engine) throw std::runtime_error("Engine deserialization failed");
        if (engine->getNbIOTensors() != 2) throw std::runtime_error("Expected one input and output");
        for (int i = 0; i < 2; ++i) {
            const auto* name = engine->getIOTensorName(i);
            if (engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
                throw std::runtime_error("Benchmark requires FP32 engine I/O");
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) input_name = name;
            else output_name = name;
        }
        context.reset(engine->createExecutionContext());
        if (!context) throw std::runtime_error("createExecutionContext failed");
        check(cudaMalloc(&input, capacity * sizeof(float)));
        check(cudaMalloc(&output, 16 * 300 * 6 * sizeof(float)));
        check(cudaMalloc(&bytes, capacity));
        check(cudaMalloc(reinterpret_cast<void**>(&lut), 256 * sizeof(float)));
        check(cudaMemcpy(lut, values, 256 * sizeof(float), cudaMemcpyHostToDevice));
        nvrtcProgram program;
        if (nvrtcCreateProgram(&program, kernel_source, "normalize.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS)
            throw std::runtime_error("NVRTC program creation failed");
        const char* options[] = {"--gpu-architecture=sm_89", "--std=c++17"};
        const auto compiled = nvrtcCompileProgram(program, 2, options);
        if (compiled != NVRTC_SUCCESS) {
            size_t length; nvrtcGetProgramLogSize(program, &length);
            std::string log(length, '\0'); nvrtcGetProgramLog(program, log.data());
            nvrtcDestroyProgram(&program); throw std::runtime_error(log);
        }
        size_t ptx_size; nvrtcGetCUBINSize(program, &ptx_size);
        std::vector<char> ptx(ptx_size); nvrtcGetCUBIN(program, ptx.data());
        nvrtcDestroyProgram(&program);
        if (cuModuleLoadData(&module, ptx.data()) != CUDA_SUCCESS ||
            cuModuleGetFunction(&kernel, module, "normalize") != CUDA_SUCCESS)
            throw std::runtime_error("CUDA kernel load failed");
        for (int i = 0; i < 2; ++i)
            bindings.push_back(engine->getTensorIOMode(engine->getIOTensorName(i)) == nvinfer1::TensorIOMode::kINPUT ? input : output);
    }
    ~Engine() {
        if (module) cuModuleUnload(module);
        cudaFree(lut); cudaFree(bytes); cudaFree(output); cudaFree(input);
    }
    void shape(int batch, int height, int width) {
        if (batch < 1 || batch > 16 || height < 1 || width < 1 || size_t(batch) * 3 * height * width > capacity)
            throw std::runtime_error("Input exceeds reserved capacity");
        nvinfer1::Dims dims{};
        dims.nbDims = 4;
        dims.d[0] = batch; dims.d[1] = 3; dims.d[2] = height; dims.d[3] = width;
        if (!context->setInputShape(input_name.c_str(), dims)) throw std::runtime_error("Shape outside engine profile");
        const auto output_dims = context->getTensorShape(output_name.c_str());
        if (output_dims.nbDims != 3 || output_dims.d[0] != batch || output_dims.d[1] != 300 || output_dims.d[2] != 6)
            throw std::runtime_error("Expected YOLO26 NMS-free [batch,300,6] output");
    }
    void run() {
        if (!context->executeV2(bindings.data())) throw std::runtime_error("executeV2 failed");
    }
};
}

extern "C" const char* chd_error() { return error.c_str(); }
extern "C" void* chd_create(const void* data, size_t size, const float* lut, size_t capacity) {
    try { return new Engine(data, size, lut, capacity); }
    catch (const std::exception& e) { error = e.what(); return nullptr; }
}
extern "C" void chd_destroy(void* pointer) { delete static_cast<Engine*>(pointer); }
extern "C" int chd_load(void* pointer, const float* tensor, int batch, int height, int width) {
    try {
        auto& e = *static_cast<Engine*>(pointer); e.shape(batch, height, width);
        check(cudaMemcpy(e.input, tensor, size_t(batch) * 3 * height * width * sizeof(float), cudaMemcpyHostToDevice));
        return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
extern "C" int chd_run(void* pointer) {
    try { static_cast<Engine*>(pointer)->run(); return 0; }
    catch (const std::exception& e) { error = e.what(); return -1; }
}
extern "C" int chd_detect(void* pointer, const uint8_t* source, float* result, int batch, int height, int width) {
    try {
        auto& e = *static_cast<Engine*>(pointer); e.shape(batch, height, width);
        check(cudaMemcpy(e.bytes, source, size_t(batch) * 3 * height * width, cudaMemcpyHostToDevice));
        int pixels = height * width;
        void* args[] = {&e.bytes, &e.input, &e.lut, &pixels, &batch};
        if (cuLaunchKernel(e.kernel, (pixels * batch + 255)/256, 1, 1,
                           256, 1, 1, 0, nullptr, args, nullptr) != CUDA_SUCCESS)
            throw std::runtime_error("CUDA normalization launch failed");
        e.run();
        check(cudaMemcpy(result, e.output, size_t(batch) * 300 * 6 * sizeof(float), cudaMemcpyDeviceToHost));
        return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
extern "C" int chd_input(void* pointer, float* result, size_t elements) {
    try {
        auto& e = *static_cast<Engine*>(pointer);
        if (elements > e.capacity) throw std::runtime_error("Read exceeds capacity");
        check(cudaMemcpy(result, e.input, elements * sizeof(float), cudaMemcpyDeviceToHost)); return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
extern "C" int chd_letterbox(const uint8_t* source, int height, int width,
                             uint8_t* output, int resized_h, int resized_w,
                             int target_h, int target_w, int top, int left) {
    try {
        cv::Mat image(height, width, CV_8UC3, const_cast<uint8_t*>(source));
        cv::Mat resized;
        if (height == resized_h && width == resized_w) resized = image;
        else cv::resize(image, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);
        cv::Mat target(target_h, target_w, CV_8UC3, output);
        cv::copyMakeBorder(resized, target, top, target_h - resized_h - top,
                           left, target_w - resized_w - left, cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
        return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
