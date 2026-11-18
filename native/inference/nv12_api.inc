// Borrowed device planes are retained by Rust's ticket owners until collect or
// engine destruction. Ready-event waits replace device-wide synchronization.
struct Nv12Planes {
    uintptr_t y,uv,ready,lut;
    int width,height,yp,uvp,colorspace,range,device;
};
extern "C" int ci_submit_nv12(void* pointer,const Nv12Planes* planes,const int* dims,
 int n,int h,int w,uint64_t* ticket,int* rows){return guarded([&]{
    auto& e=*static_cast<Engine*>(pointer);
    if(!e.async)throw Unsupported("NV12 requires asynchronous execution");
    if(e.async->failed)throw std::runtime_error("Asynchronous engine failed; reload it");
    check(cudaSetDevice(e.device));
    if(!planes||!dims||n<1||n>e.max_batch)throw std::invalid_argument("Invalid NV12 batch");
    for(int i=0;i<n;++i){const auto& p=planes[i];const int* d=dims+i*4;
        if(p.device!=e.device||p.width<2||p.height<2||p.width%2||p.height%2||
           p.width>16384||p.height>16384||uint64_t(p.width)*p.height>64*1024*1024||
           p.yp<p.width||p.uvp<p.width||!p.ready||d[0]<1||d[1]<1||d[2]<0||d[3]<0||
           d[0]>h-d[2]||d[1]>w-d[3]||p.height%d[0]||p.width%d[1]||
           p.height/d[0]!=p.width/d[1]||(p.width/d[1])%2!=1||
           !(p.colorspace==1||p.colorspace==2||p.colorspace==5||p.colorspace==6)||p.range<0||p.range>2)
            throw std::invalid_argument("Unsupported NV12 geometry or colour profile");
        for(auto address:{p.y,p.uv,p.lut}){
            CUmemorytype type;int device=-1;
            if(cuPointerGetAttribute(&type,CU_POINTER_ATTRIBUTE_MEMORY_TYPE,address)!=CUDA_SUCCESS||
               cuPointerGetAttribute(&device,CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,address)!=CUDA_SUCCESS||
               type!=CU_MEMORYTYPE_DEVICE||device!=e.device)
                throw std::invalid_argument("NV12 must reside on the selected CUDA device");
        }
    }
    e.shape(n,h,w);AsyncSlot* slot=nullptr;
    for(auto& candidate:e.async->slots)if(!candidate.pending&&!candidate.reserved){slot=&candidate;break;}
    if(!slot)throw std::invalid_argument("Both asynchronous slots are pending");
    if(e.async->sequence==UINT64_MAX)throw std::runtime_error("Ticket sequence exhausted");
    CUfunction kernel;
    if(cuModuleGetFunction(&kernel,e.kernel.module,"nv12_letterbox")!=CUDA_SUCCESS)
        throw std::runtime_error("Missing fused NV12 kernel");
    slot->pending=true;slot->ticket=++e.async->sequence;
    try{
        for(int i=0;i<n;++i){auto p=planes[i];int rh=dims[i*4],rw=dims[i*4+1],top=dims[i*4+2],left=dims[i*4+3];
            check(cudaStreamWaitEvent(e.async->compute.value,reinterpret_cast<cudaEvent_t>(p.ready),0));
            void* y=reinterpret_cast<void*>(p.y);void* uv=reinterpret_cast<void*>(p.uv);
            void* lut=reinterpret_cast<void*>(p.lut);
            void* output=static_cast<float*>(slot->input.ptr)+size_t(i)*3*h*w;
            void* args[]={&y,&uv,&lut,&output,&p.width,&p.height,&p.yp,&p.uvp,&w,&h,&rw,&rh,&left,&top};
            if(cuLaunchKernel(kernel,(h*w+255)/256,1,1,256,1,1,0,reinterpret_cast<CUstream>(e.async->compute.value),args,nullptr)!=CUDA_SUCCESS)
                throw std::runtime_error("Fused NV12 preprocessing launch failed");
        }
        if(!e.context->setTensorAddress(e.input_name.c_str(),slot->input.ptr)||
           !e.context->setTensorAddress(e.output_name.c_str(),slot->output.ptr)||
           !e.context->enqueueV3(e.async->compute.value))
            throw std::runtime_error("NV12 TensorRT enqueue failed: "+e.logger.details());
        check(cudaMemcpyAsync(slot->host_output.ptr,slot->output.ptr,slot->output_bytes,cudaMemcpyDeviceToHost,e.async->compute.value));
        check(cudaEventRecord(slot->done.value,e.async->compute.value));
        e.async->device_frames+=n;e.async->submitted+=1;*ticket=slot->ticket;*rows=e.output_rows;
    }catch(...){cudaStreamSynchronize(e.async->compute.value);e.async->failed=true;throw;}
});}
