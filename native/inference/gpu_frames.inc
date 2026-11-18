// Device-owned image storage. Foreign DLPack planes are copied only on device.
struct GpuFactory { Memory lut; Kernel kernel; int device;
    std::atomic<uint64_t> imports{0},downloads{0},overlay_uploads{0};
    GpuFactory(const uint8_t* data,size_t bytes,int d):device(d){
        if(bytes!=size_t(256)*256*256*3)throw std::invalid_argument("Expected complete NV12 color lookup");
        check(cudaSetDevice(d));kernel.compile();lut.reserve(bytes);
        check(cudaMemcpy(lut.ptr,data,bytes,cudaMemcpyHostToDevice));
    }
    ~GpuFactory(){cudaSetDevice(device);}
    CUfunction function(const char* name){CUfunction f=nullptr;if(cuModuleGetFunction(&f,kernel.module,name)!=CUDA_SUCCESS)throw std::runtime_error("Missing GPU frame kernel");return f;}
};
struct GpuImage { std::shared_ptr<GpuFactory> factory; Memory bgr,surface,scratch; void* y=nullptr; void* uv=nullptr; Stream stream;
    int width,height,pitch;bool dirty=false;
    GpuImage(std::shared_ptr<GpuFactory> f,int w,int h):factory(std::move(f)),stream(factory->device),width(w),height(h),pitch((w+255)/256*256){
        if(w<2||h<2||w>16384||h>16384||w%2||h%2||uint64_t(w)*h>64*1024*1024)throw std::invalid_argument("Expected even NV12 image at most 64M pixels and 16K per side");
        check(cudaSetDevice(factory->device));bgr.reserve(size_t(w)*h*3);surface.reserve(size_t(pitch)*h*3/2);y=surface.ptr;uv=static_cast<uint8_t*>(surface.ptr)+size_t(pitch)*h;
    }
    ~GpuImage(){cudaSetDevice(factory->device);cudaStreamSynchronize(stream.value);}
    void launch(const char* name,int count,void** args){
        check(cudaSetDevice(factory->device));
        if(cuLaunchKernel(factory->function(name),(count+255)/256,1,1,256,1,1,0,reinterpret_cast<CUstream>(stream.value),args,nullptr)!=CUDA_SUCCESS)
            throw std::runtime_error("GPU frame kernel launch failed");
    }
    void sync(){check(cudaStreamSynchronize(stream.value));}
    void nv12(){if(!dirty)return;void* args[]={&bgr.ptr,&y,&uv,&width,&height,&pitch};launch("frame_to_nv12",width*height,args);sync();dirty=false;}
};
using FactoryHandle=std::shared_ptr<GpuFactory>;
using ImageHandle=std::shared_ptr<GpuImage>;
