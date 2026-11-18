extern "C" int ci_gpu_sync(int device){return guarded([&]{check(cudaSetDevice(device));check(cudaDeviceSynchronize());});}
extern "C" void* ci_factory_create(const uint8_t* data,size_t length,int device){
    FactoryHandle* result=nullptr;guarded([&]{result=new FactoryHandle(std::make_shared<GpuFactory>(data,length,device));});return result;
}
extern "C" void ci_factory_destroy(void* p){delete static_cast<FactoryHandle*>(p);}
extern "C" void* ci_frame_import(void* factory,const uint8_t* y,const uint8_t* uv,int yp,int uvp,int width,int height){
    ImageHandle* result=nullptr;guarded([&]{
        auto f=*static_cast<FactoryHandle*>(factory);check(cudaSetDevice(f->device));
        if(yp<width||uvp<width)throw std::invalid_argument("NV12 pitch is smaller than width");
        for(auto ptr:{y,uv}){
            CUmemorytype type;int device=-1;
            if(cuPointerGetAttribute(&type,CU_POINTER_ATTRIBUTE_MEMORY_TYPE,reinterpret_cast<CUdeviceptr>(ptr))!=CUDA_SUCCESS ||
               cuPointerGetAttribute(&device,CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,reinterpret_cast<CUdeviceptr>(ptr))!=CUDA_SUCCESS ||
               type!=CU_MEMORYTYPE_DEVICE || device!=f->device)
                throw std::invalid_argument("NV12 planes must use the selected CUDA device");
        }
        auto frame=std::make_shared<GpuImage>(f,width,height);
        check(cudaMemcpy2DAsync(frame->y,frame->pitch,y,yp,width,height,cudaMemcpyDeviceToDevice,frame->stream.value));
        check(cudaMemcpy2DAsync(frame->uv,frame->pitch,uv,uvp,width,height/2,cudaMemcpyDeviceToDevice,frame->stream.value));
        void* args[]={&frame->y,&frame->uv,&frame->bgr.ptr,&f->lut.ptr,&width,&height,&frame->pitch};
        frame->launch("nv12_to_bgr",width*height,args);frame->sync();frame->dirty=true;f->imports+=1;result=new ImageHandle(std::move(frame));
    });return result;
}
extern "C" void ci_frame_destroy(void* p){delete static_cast<ImageHandle*>(p);}
extern "C" void* ci_frame_copy(void* p){ImageHandle* result=nullptr;guarded([&]{auto old=*static_cast<ImageHandle*>(p);auto fresh=std::make_shared<GpuImage>(old->factory,old->width,old->height);
    check(cudaMemcpyAsync(fresh->bgr.ptr,old->bgr.ptr,size_t(old->width)*old->height*3,cudaMemcpyDeviceToDevice,fresh->stream.value));fresh->sync();fresh->dirty=true;result=new ImageHandle(std::move(fresh));});return result;}
extern "C" int ci_frame_download(void* p,uint8_t* target){return guarded([&]{auto f=*static_cast<ImageHandle*>(p);check(cudaSetDevice(f->factory->device));check(cudaMemcpy(target,f->bgr.ptr,size_t(f->width)*f->height*3,cudaMemcpyDeviceToHost));f->factory->downloads+=size_t(f->width)*f->height*3;});}
extern "C" int ci_frame_blend(void* p,const uint8_t* rgba,int ow,int oh,int x,int y){return guarded([&]{
    auto f=*static_cast<ImageHandle*>(p);if(f.use_count()!=2)throw std::invalid_argument("GPU image is in use; copy it before drawing");
    if(ow<1||oh<1||uint64_t(ow)*oh>64*1024*1024)throw std::invalid_argument("Invalid overlay size");
    check(cudaSetDevice(f->factory->device));f->scratch.reserve(size_t(ow)*oh*4);check(cudaMemcpy(f->scratch.ptr,rgba,size_t(ow)*oh*4,cudaMemcpyHostToDevice));
    void* args[]={&f->bgr.ptr,&f->scratch.ptr,&f->width,&f->height,&ow,&oh,&x,&y};f->launch("frame_blend",ow*oh,args);f->sync();f->dirty=true;f->factory->overlay_uploads+=size_t(ow)*oh*4;
});}
extern "C" int ci_submit_device(void* p,void* const* images,const int* dims,int n,int h,int w,uint64_t* ticket,int* rows){return guarded([&]{
    auto& e=*static_cast<Engine*>(p);if(!e.async)throw Unsupported("Device frames require asynchronous execution");
    if(e.async->failed)throw std::runtime_error("Asynchronous engine failed; close and reload it");
    e.shape(n,h,w);AsyncSlot* slot=nullptr;for(auto& candidate:e.async->slots)if(!candidate.pending&&!candidate.reserved){slot=&candidate;break;}
    if(!slot)throw std::invalid_argument("Both asynchronous slots are pending");
    std::vector<ImageHandle> owners;owners.reserve(n);
    for(int i=0;i<n;++i){auto f=*static_cast<ImageHandle*>(images[i]);if(f->factory->device!=e.device)throw std::invalid_argument("GPU device mismatch");
        int rh=dims[i*4],rw=dims[i*4+1];
        if(rh<1||rw<1||rh>h||rw>w||f->height%rh||f->width%rw||f->height/rh!=f->width/rw||f->width/rw<1)
            throw Unsupported("GPU resize requires integer isotropic downscaling");
        owners.push_back(std::move(f));}
    if(e.async->sequence==UINT64_MAX)throw std::runtime_error("Ticket sequence exhausted");
    slot->pending=true;slot->ticket=++e.async->sequence;slot->owners=std::move(owners);
    try{for(int i=0;i<n;++i){auto f=slot->owners[i];int rh=dims[i*4],rw=dims[i*4+1],top=dims[i*4+2],left=dims[i*4+3];
        float* output=static_cast<float*>(slot->input.ptr)+size_t(i)*3*h*w;
        void* args[]={&f->bgr.ptr,&output,&f->width,&f->height,&w,&h,&rw,&rh,&left,&top};
        if(cuLaunchKernel(e.async_gpu_resize,(h*w+255)/256,1,1,256,1,1,0,reinterpret_cast<CUstream>(e.async->compute.value),args,nullptr)!=CUDA_SUCCESS)throw std::runtime_error("GPU letterbox launch failed");}
        if(!e.context->setTensorAddress(e.input_name.c_str(),slot->input.ptr)||!e.context->setTensorAddress(e.output_name.c_str(),slot->output.ptr)||!e.context->enqueueV3(e.async->compute.value))throw std::runtime_error("TensorRT device enqueue failed: "+e.logger.details());
        check(cudaMemcpyAsync(slot->host_output.ptr,slot->output.ptr,slot->output_bytes,cudaMemcpyDeviceToHost,e.async->compute.value));check(cudaEventRecord(slot->done.value,e.async->compute.value));
        e.async->device_frames+=n;e.async->submitted+=1;*ticket=slot->ticket;*rows=e.output_rows;
    }catch(...){cudaStreamSynchronize(e.async->compute.value);e.async->failed=true;throw;}
});}

extern "C" void ci_factory_stats(void* p,uint64_t* values){auto f=*static_cast<FactoryHandle*>(p);values[0]=f->imports.load();values[1]=f->downloads.load();values[2]=f->overlay_uploads.load();}
