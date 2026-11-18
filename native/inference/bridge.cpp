// TensorRT's public API is C++; keep its lifetime and exceptions behind a C ABI.
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "preprocess_source.h"
#ifdef CHD_GPU_MEDIA
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
}
#endif
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
thread_local int last_code = 0;
struct Unsupported : std::runtime_error { using std::runtime_error::runtime_error; };
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Memory {
    void* ptr = nullptr;
    size_t size = 0;
    ~Memory() { if (ptr) cudaFree(ptr); }
    void reserve(size_t needed) {
        if (needed <= size) return;
        // executeV2 completes synchronously and the engine mutex serializes
        // calls. Release old storage first to avoid doubling activation memory
        // during growth. shape() invalidates its cache before any allocation,
        // so a failed allocation cannot execute with stale buffer addresses.
        if (ptr) cudaFree(ptr);
        ptr = nullptr; size = 0;
        check(cudaMalloc(&ptr, needed));
        size = needed;
    }
};
struct PinnedMemory {
    void* ptr=nullptr; size_t size=0;
    ~PinnedMemory() {if(ptr) cudaFreeHost(ptr);}
    void reserve(size_t bytes) {
        if(bytes<=size) return;
        if(ptr) cudaFreeHost(ptr);
        ptr=nullptr;size=0;check(cudaMallocHost(&ptr,bytes));std::memset(ptr,0,bytes);size=bytes;
    }
};
struct Stream {
    cudaStream_t value=nullptr;
    Stream(int device=-1){if(device>=0)check(cudaSetDevice(device));check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
    ~Stream(){if(value){cudaStreamSynchronize(value);cudaStreamDestroy(value);}}
};
struct Event {
    cudaEvent_t value=nullptr;
    Event(){check(cudaEventCreateWithFlags(&value,cudaEventDisableTiming));}
    ~Event(){if(value)cudaEventDestroy(value);}
};
struct Kernel {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    ~Kernel() { if (module) cuModuleUnload(module); }
    void compile() {
        int device, major, minor;
        check(cudaGetDevice(&device));
        if (cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device)!=CUDA_SUCCESS ||
            cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device)!=CUDA_SUCCESS)
            throw std::runtime_error("Cannot query CUDA compute capability");
        const auto architecture = "--gpu-architecture=sm_" + std::to_string(major) + std::to_string(minor);
        nvrtcProgram program;
        if (nvrtcCreateProgram(&program, kernel_source, "normalize.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS)
            throw std::runtime_error("NVRTC program creation failed");
        const char* options[] = {architecture.c_str(), "--std=c++17"};
        const auto status = nvrtcCompileProgram(program, 2, options);
        if (status != NVRTC_SUCCESS) {
            size_t length = 0; nvrtcGetProgramLogSize(program, &length);
            std::string log(length, '\0'); nvrtcGetProgramLog(program, log.data());
            nvrtcDestroyProgram(&program); throw std::runtime_error("NVRTC: " + log);
        }
        size_t length = 0;
        auto result = nvrtcGetCUBINSize(program, &length);
        std::vector<char> cubin(length);
        if (result == NVRTC_SUCCESS) result = nvrtcGetCUBIN(program, cubin.data());
        nvrtcDestroyProgram(&program);
        if (result != NVRTC_SUCCESS || !length) throw std::runtime_error("NVRTC produced no CUDA binary");
        if (cuModuleLoadData(&module, cubin.data()) != CUDA_SUCCESS ||
            cuModuleGetFunction(&function, module, "normalize") != CUDA_SUCCESS)
            throw std::runtime_error("CUDA normalization kernel load failed");
    }
};
#include "gpu_frames.inc"
struct AsyncSlot {
    Memory pixels,input,output;
    std::vector<ImageHandle> owners;
    PinnedMemory host_pixels,host_output;
    Event uploaded,done;
    bool pending=false,reserved=false; uint64_t ticket=0; size_t output_bytes=0;
};
struct AsyncState {
    Stream upload,compute;
    AsyncSlot slots[2];
    uint64_t sequence=0;
    bool failed=false;
    uint64_t host_upload_bytes=0,device_frames=0,submitted=0;
    void sync(){check(cudaStreamSynchronize(upload.value));check(cudaStreamSynchronize(compute.value));}
    bool busy()const{return slots[0].pending||slots[1].pending||slots[0].reserved||slots[1].reserved;}
};
struct Logger : nvinfer1::ILogger {
    std::mutex lock;
    std::string error;
    void log(Severity level, const char* message) noexcept override {
        try {
            if (level <= Severity::kERROR) { std::lock_guard<std::mutex> guard(lock); error = message; }
            if (level <= Severity::kWARNING) std::fprintf(stderr,"[Rust-TensorRT] %s\n",message);
        } catch (...) {}
    }
    std::string details() { std::lock_guard<std::mutex> guard(lock); return error; }
};
struct Engine {
    // Context dies before activation buffers, engine, runtime and logger.
    Logger logger;
    int device, max_batch = 0, image_size;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    Memory input, output, pixels, activation;
    Kernel kernel;
    CUfunction async_gpu_resize=nullptr;
    std::unique_ptr<AsyncState> async;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::string input_name, output_name;
    std::vector<void*> bindings;
    int current_n=0, current_h=0, current_w=0, output_rows=0;

    Engine(const void* plan,size_t length,int requested_batch,int size,int selected_device,bool asynchronous)
      : device(selected_device),image_size(size) {
        if (getInferLibVersion()!=NV_TENSORRT_VERSION)
            throw std::runtime_error("TensorRT runtime/header version mismatch: runtime=" +
                std::to_string(getInferLibVersion()) + " headers=" + std::to_string(NV_TENSORRT_VERSION));
        check(cudaSetDevice(device));
        cv::setNumThreads(1);
        initLibNvInferPlugins(&logger,"");
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime) throw std::runtime_error("Cannot create TensorRT runtime");
        engine.reset(runtime->deserializeCudaEngine(plan,length));
        if (!engine) throw std::runtime_error("Cannot deserialize TensorRT engine: " + logger.details());
        if (engine->getNbIOTensors()!=2) throw Unsupported("Rust requires one input and one NMS-free detection output");
        for (int i=0;i<2;++i) {
            const char* name=engine->getIOTensorName(i);
            const auto dtype=engine->getTensorDataType(name);
            if (dtype!=nvinfer1::DataType::kFLOAT)
                throw Unsupported("Rust requires FP32 engine I/O (internal INT8 is supported)");
            if (engine->getTensorLocation(name)!=nvinfer1::TensorLocation::kDEVICE)
                throw Unsupported("Rust requires CUDA device I/O");
            if (engine->getTensorFormat(name)!=nvinfer1::TensorFormat::kLINEAR)
                throw Unsupported("Rust requires linear engine I/O");
            if (engine->getTensorIOMode(name)==nvinfer1::TensorIOMode::kINPUT) {
                input_name=name;
            } else { output_name=name; }
        }
        auto minimum=engine->getProfileShape(input_name.c_str(),0,nvinfer1::OptProfileSelector::kMIN);
        auto maximum=engine->getProfileShape(input_name.c_str(),0,nvinfer1::OptProfileSelector::kMAX);
        const auto fixed=engine->getTensorShape(input_name.c_str());
        if (fixed.nbDims==4 && fixed.d[0]>0) minimum=maximum=fixed;
        if (minimum.nbDims!=4 || maximum.nbDims!=4 || minimum.d[0]!=1 ||
            maximum.d[0]<1 || maximum.d[1]!=3 || size<minimum.d[2] || size>maximum.d[2] ||
            size<minimum.d[3] || size>maximum.d[3])
            throw Unsupported("Engine must support NCHW BGR input at requested size and batch 1");
        max_batch=static_cast<int>(std::min<int64_t>(requested_batch,maximum.d[0]));
        context.reset(engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
        if (!context) throw std::runtime_error("Cannot create TensorRT context: " + logger.details());
        kernel.compile();
        if(cuModuleGetFunction(&async_gpu_resize,kernel.module,"gpu_letterbox")!=CUDA_SUCCESS)throw std::runtime_error("Missing GPU letterbox kernel");
        if(asynchronous) async=std::make_unique<AsyncState>();
        shape(1,size,size);
    }
    ~Engine() {
        // PyO3 may release the final reference on another thread.
        cudaSetDevice(device);
        if(async){cudaStreamSynchronize(async->upload.value);cudaStreamSynchronize(async->compute.value);}
    }
    void shape(int n,int h,int w) {
        check(cudaSetDevice(device));
        if (n<1 || n>max_batch || h<1 || w<1 || h>image_size || w>image_size ||
            uint64_t(n)*3*uint64_t(h)*uint64_t(w)>uint64_t(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Input exceeds native engine capacity");
        if (n==current_n && h==current_h && w==current_w) return;
        if(async && async->busy()) throw std::invalid_argument("Collect pending batches before changing shape");
        current_n=0;
        nvinfer1::Dims dims{};dims.nbDims=4;dims.d[0]=n;dims.d[1]=3;dims.d[2]=h;dims.d[3]=w;
        if (!context->setInputShape(input_name.c_str(),dims))
            throw std::invalid_argument("Input is outside TensorRT optimization profile");
        auto out=context->getTensorShape(output_name.c_str());
        if (out.nbDims!=3 || out.d[0]!=n || out.d[1]<1 || out.d[1]>3000 || out.d[2]!=6)
            throw Unsupported("Rust requires NMS-free [batch, boxes, 6] detections");
        output_rows=out.d[1];
        const auto bytes=size_t(n)*3*h*w;
        if(async){
            for(auto& slot:async->slots){
                slot.input.reserve(bytes*4);slot.pixels.reserve(bytes);
                slot.output_bytes=size_t(n)*output_rows*6*4;
                slot.output.reserve(slot.output_bytes);
                slot.host_pixels.reserve(bytes);slot.host_output.reserve(slot.output_bytes);
            }
        }else{
            input.reserve(bytes*4);pixels.reserve(bytes);
            output.reserve(size_t(n)*output_rows*6*4);
        }
        const auto memory=context->updateDeviceMemorySizeForShapes();
        activation.reserve(memory);
        context->setDeviceMemoryV2(activation.ptr,activation.size);
        bindings.clear();
        for (int i=0;i<2;++i)
            bindings.push_back(engine->getTensorIOMode(engine->getIOTensorName(i))==nvinfer1::TensorIOMode::kINPUT?input.ptr:output.ptr);
        current_n=n;current_h=h;current_w=w;
    }
    AsyncSlot* reserve_host(int n,int h,int w){
        if(!async) throw Unsupported("Asynchronous execution is not enabled");
        if(async->failed) throw std::runtime_error("Asynchronous engine failed; close and reload it");
        shape(n,h,w);
        for(auto& slot:async->slots) if(!slot.pending&&!slot.reserved){
            if(async->sequence==UINT64_MAX) throw std::runtime_error("Ticket sequence exhausted");
            slot.ticket=++async->sequence;slot.reserved=true;return &slot;
        }
        throw std::invalid_argument("Both asynchronous slots are occupied");
    }
    AsyncSlot* reservation(uint64_t ticket){
        if(async)for(auto& slot:async->slots)if(slot.reserved&&slot.ticket==ticket)return &slot;
        throw std::invalid_argument("Unknown host reservation");
    }
    uint64_t submit(const uint8_t* source,int n,int h,int w){
        auto* slot=reserve_host(n,h,w);
        std::memcpy(slot->host_pixels.ptr,source,size_t(n)*h*w*3);
        return enqueue_host(slot,n,h,w);
    }
    uint64_t enqueue_host(AsyncSlot* slot,int n,int h,int w){
        slot->reserved=false;slot->pending=true;
        try{
            const auto bytes=size_t(n)*h*w*3;
            async->host_upload_bytes+=bytes;
            check(cudaMemcpyAsync(slot->pixels.ptr,slot->host_pixels.ptr,bytes,cudaMemcpyHostToDevice,async->upload.value));
            check(cudaEventRecord(slot->uploaded.value,async->upload.value));
            check(cudaStreamWaitEvent(async->compute.value,slot->uploaded.value,0));
            int count=h*w;
            void* args[]={&slot->pixels.ptr,&slot->input.ptr,&count,&n};
            if(cuLaunchKernel(kernel.function,(count*n+255)/256,1,1,256,1,1,0,
                reinterpret_cast<CUstream>(async->compute.value),args,nullptr)!=CUDA_SUCCESS)
                throw std::runtime_error("Asynchronous normalization launch failed");
            if(!context->setTensorAddress(input_name.c_str(),slot->input.ptr) ||
               !context->setTensorAddress(output_name.c_str(),slot->output.ptr) ||
               !context->enqueueV3(async->compute.value))
                throw std::runtime_error("TensorRT enqueue failed: "+logger.details());
            check(cudaMemcpyAsync(slot->host_output.ptr,slot->output.ptr,slot->output_bytes,cudaMemcpyDeviceToHost,async->compute.value));
            check(cudaEventRecord(slot->done.value,async->compute.value));
            async->submitted+=1;return slot->ticket;
        }catch(...){
            cudaStreamSynchronize(async->upload.value);cudaStreamSynchronize(async->compute.value);
            async->failed=true;throw;
        }
    }
    // CPU BGR is already colour-correct. Only exact odd-integer reductions
    // enter this path; the caller retains the ordinary CPU path for all others.
    uint64_t submit_bgr(const uint8_t* const* sources,const int* dims,int n,int h,int w){
        if(!async) throw Unsupported("GPU preprocessing requires asynchronous execution");
        if(async->failed) throw std::runtime_error("Asynchronous engine failed; reload it");
        if(!sources||!dims) throw std::invalid_argument("Missing BGR batch");
        shape(n,h,w);
        size_t bytes=0;
        for(int i=0;i<n;++i){
            const int* d=dims+i*6; // source h,w; resized h,w; top,left
            if(!sources[i]||d[0]<=0||d[1]<=0||d[2]<=0||d[3]<=0||
               uint64_t(d[0])*d[1]>64*1024*1024 || d[4]<0||d[5]<0||
               d[2]>h-d[4]||d[3]>w-d[5] || d[0]%d[2] || d[1]%d[3] ||
               d[0]/d[2]!=d[1]/d[3] || (d[1]/d[3])%2!=1)
                throw std::invalid_argument("BGR scale is not an exact odd integer reduction");
            bytes+=size_t(d[0])*d[1]*3;
        }
        AsyncSlot* slot=nullptr;
        for(auto& candidate:async->slots) if(!candidate.pending&&!candidate.reserved){slot=&candidate;break;}
        if(!slot) throw std::invalid_argument("Both asynchronous slots are pending");
        if(async->sequence==UINT64_MAX) throw std::runtime_error("Ticket sequence exhausted");
        slot->pending=true;slot->ticket=++async->sequence;
        try{
            slot->pixels.reserve(bytes);slot->host_pixels.reserve(bytes);
            size_t offset=0;
            for(int i=0;i<n;++i){const int* d=dims+i*6;const auto length=size_t(d[0])*d[1]*3;
                std::memcpy(static_cast<uint8_t*>(slot->host_pixels.ptr)+offset,sources[i],length);offset+=length;}
            async->host_upload_bytes+=bytes;
            check(cudaMemcpyAsync(slot->pixels.ptr,slot->host_pixels.ptr,bytes,cudaMemcpyHostToDevice,async->upload.value));
            check(cudaEventRecord(slot->uploaded.value,async->upload.value));
            check(cudaStreamWaitEvent(async->compute.value,slot->uploaded.value,0));
            offset=0;
            for(int i=0;i<n;++i){const int* d=dims+i*6;
                void* source=static_cast<uint8_t*>(slot->pixels.ptr)+offset;
                void* target=static_cast<float*>(slot->input.ptr)+size_t(i)*3*h*w;
                int width=d[1],height=d[0],rh=d[2],rw=d[3],top=d[4],left=d[5];
                void* args[]={&source,&target,&width,&height,&w,&h,&rw,&rh,&left,&top};
                if(cuLaunchKernel(async_gpu_resize,(h*w+255)/256,1,1,256,1,1,0,
                    reinterpret_cast<CUstream>(async->compute.value),args,nullptr)!=CUDA_SUCCESS)
                    throw std::runtime_error("Exact GPU preprocessing failed");
                offset+=size_t(height)*width*3;
            }
            if(!context->setTensorAddress(input_name.c_str(),slot->input.ptr)||
               !context->setTensorAddress(output_name.c_str(),slot->output.ptr)||
               !context->enqueueV3(async->compute.value))
                throw std::runtime_error("TensorRT enqueue failed: "+logger.details());
            check(cudaMemcpyAsync(slot->host_output.ptr,slot->output.ptr,slot->output_bytes,cudaMemcpyDeviceToHost,async->compute.value));
            check(cudaEventRecord(slot->done.value,async->compute.value));
            async->submitted+=1;return slot->ticket;
        }catch(...){async->sync();async->failed=true;throw;}
    }
    void collect(uint64_t ticket,float* target){
        if(!async) throw Unsupported("Asynchronous execution is not enabled");
        if(async->failed) throw std::runtime_error("Asynchronous engine failed; close and reload it");
        for(auto& slot:async->slots) if(slot.pending && slot.ticket==ticket){
            check(cudaSetDevice(device));check(cudaEventSynchronize(slot.done.value));
            std::memcpy(target,slot.host_output.ptr,slot.output_bytes);slot.pending=false;slot.owners.clear();return;
        }
        throw std::invalid_argument("Unknown or collected asynchronous ticket");
    }

};
template<typename F> int guarded(F operation) {
    try { operation();last_code=0;return 0; }
    catch (const Unsupported& e) {last_error=e.what();last_code=-2;}
    catch (const std::invalid_argument& e) {last_error=e.what();last_code=-3;}
    catch (const std::exception& e) {last_error=e.what();last_code=-1;}
    catch (...) {last_error="Unknown native inference error";last_code=-1;}
    return last_code;
}
}

extern "C" const char* ci_error() {return last_error.c_str();}
extern "C" int ci_error_code() {return last_code;}
extern "C" int ci_header_version() {return NV_TENSORRT_VERSION;}
extern "C" void* ci_create(const void* data,size_t length,int batch,int size,int device,int asynchronous) {
    Engine* result=nullptr;
    guarded([&] {result=new Engine(data,length,batch,size,device,asynchronous!=0);});return result;
}
extern "C" void ci_destroy(void* engine) {delete static_cast<Engine*>(engine);}
extern "C" int ci_max_batch(void* engine) {return static_cast<Engine*>(engine)->max_batch;}
extern "C" size_t ci_activation_bytes(void* engine) {return static_cast<Engine*>(engine)->activation.size;}
extern "C" int ci_rows(void* pointer,int n,int h,int w,int* rows) {
    return guarded([&] {auto& e=*static_cast<Engine*>(pointer);e.shape(n,h,w);*rows=e.output_rows;});
}
extern "C" int ci_predict(void* pointer,const uint8_t* source,float* destination,int n,int h,int w) {
    return guarded([&] {
        auto& e=*static_cast<Engine*>(pointer);e.shape(n,h,w);
        if(e.async){const auto ticket=e.submit(source,n,h,w);e.collect(ticket,destination);return;}
        check(cudaMemcpy(e.pixels.ptr,source,size_t(n)*3*h*w,cudaMemcpyHostToDevice));
        int count=h*w;
        void* args[]={&e.pixels.ptr,&e.input.ptr,&count,&n};
        if (cuLaunchKernel(e.kernel.function,(count*n+255)/256,1,1,256,1,1,0,nullptr,args,nullptr)!=CUDA_SUCCESS)
            throw std::runtime_error("CUDA normalization launch failed");
        if (!e.context->executeV2(e.bindings.data()))
            throw std::runtime_error("TensorRT inference failed: " + e.logger.details());
        size_t elements=size_t(n)*e.output_rows*6;
        check(cudaMemcpy(destination,e.output.ptr,elements*4,cudaMemcpyDeviceToHost));
    });
}
extern "C" int ci_submit(void* pointer,const uint8_t* source,int n,int h,int w,uint64_t* ticket,int* rows){
    return guarded([&]{auto& e=*static_cast<Engine*>(pointer);*ticket=e.submit(source,n,h,w);*rows=e.output_rows;});
}
extern "C" int ci_reserve_host(void* pointer,int n,int h,int w,uint64_t* ticket,uint8_t** target,size_t* length){
    return guarded([&]{auto* slot=static_cast<Engine*>(pointer)->reserve_host(n,h,w);
        *ticket=slot->ticket;*target=static_cast<uint8_t*>(slot->host_pixels.ptr);*length=size_t(n)*h*w*3;});
}
extern "C" int ci_abort_host(void* pointer,uint64_t ticket){
    return guarded([&]{static_cast<Engine*>(pointer)->reservation(ticket)->reserved=false;});
}
extern "C" int ci_submit_host(void* pointer,uint64_t ticket,int* rows){
    return guarded([&]{auto& e=*static_cast<Engine*>(pointer);auto* slot=e.reservation(ticket);
        e.enqueue_host(slot,e.current_n,e.current_h,e.current_w);*rows=e.output_rows;});
}
extern "C" int ci_submit_bgr(void* pointer,const uint8_t* const* sources,const int* dims,int n,int h,int w,uint64_t* ticket,int* rows){
    return guarded([&]{auto& e=*static_cast<Engine*>(pointer);*ticket=e.submit_bgr(sources,dims,n,h,w);*rows=e.output_rows;});
}
// Diagnostic readback is used by the tensor-parity gate, never the live path.
extern "C" int ci_debug_input(void* pointer,uint64_t ticket,float* target,size_t elements){
    return guarded([&]{auto& e=*static_cast<Engine*>(pointer);
        if(!e.async||elements!=size_t(e.current_n)*3*e.current_h*e.current_w)throw std::invalid_argument("Invalid input snapshot");
        for(auto& slot:e.async->slots)if(slot.pending&&slot.ticket==ticket){
            check(cudaSetDevice(e.device));check(cudaEventSynchronize(slot.done.value));
            check(cudaMemcpy(target,slot.input.ptr,elements*sizeof(float),cudaMemcpyDeviceToHost));return;
        }
        throw std::invalid_argument("Unknown input snapshot ticket");
    });
}
extern "C" int ci_collect(void* pointer,uint64_t ticket,float* target){
    return guarded([&]{static_cast<Engine*>(pointer)->collect(ticket,target);});
}
extern "C" size_t ci_pinned_bytes(void* pointer){
    auto& e=*static_cast<Engine*>(pointer);size_t result=0;
    if(e.async)for(auto& slot:e.async->slots) result+=slot.host_pixels.size+slot.host_output.size;
    return result;
}
extern "C" int ci_letterbox(const uint8_t* source,int h,int w,uint8_t* target,int rh,int rw,int th,int tw,int top,int left) {
    return guarded([&] {
        cv::Mat input(h,w,CV_8UC3,const_cast<uint8_t*>(source));
        cv::Mat output(th,tw,CV_8UC3,target);
        if (top) output.rowRange(0,top).setTo(cv::Scalar(114,114,114));
        if (top+rh<th) output.rowRange(top+rh,th).setTo(cv::Scalar(114,114,114));
        if (left) output(cv::Rect(0,top,left,rh)).setTo(cv::Scalar(114,114,114));
        if (left+rw<tw) output(cv::Rect(left+rw,top,tw-left-rw,rh)).setTo(cv::Scalar(114,114,114));
        cv::Mat resized=output(cv::Rect(left,top,rw,rh));
        if (h==rh && w==rw) input.copyTo(resized);
        else cv::resize(input,resized,cv::Size(rw,rh),0,0,cv::INTER_LINEAR);
    });
}

#include "gpu_api.inc"
#include "nv12_api.inc"

#include "gpu_encoder.inc"

extern "C" void ci_engine_stats(void* p,uint64_t* values){auto& e=*static_cast<Engine*>(p);values[0]=e.async?e.async->host_upload_bytes:0;values[1]=e.async?e.async->device_frames:0;values[2]=e.async?e.async->submitted:0;}
