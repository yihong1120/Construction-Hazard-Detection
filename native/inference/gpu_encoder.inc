#ifdef CHD_GPU_MEDIA
namespace {
void av_check(int status){if(status<0){char message[256];av_strerror(status,message,sizeof(message));throw std::runtime_error(std::string("GPU encoder: ")+message);}}
struct AvFrameDeleter{void operator()(AVFrame* p)const{av_frame_free(&p);}};
struct AvPacketDeleter{void operator()(AVPacket* p)const{av_packet_free(&p);}};
void image_buffer_free(void* opaque,uint8_t*){delete static_cast<ImageHandle*>(opaque);}
struct GpuEncoder {
    int device,width,height;int64_t sequence=0;
    AVBufferRef* device_ctx=nullptr;AVBufferRef* frames_ctx=nullptr;
    AVCodecContext* codec=nullptr;
    std::vector<uint8_t> output;
    bool flushed=false;
    GpuEncoder(int w,int h,int fps,int d):device(d),width(w),height(h){
        if(w<2||h<2||w%2||h%2||fps<1||fps>240)throw std::invalid_argument("Invalid GPU encoder dimensions or FPS");
        try{
            check(cudaSetDevice(device));
            device_ctx=av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);if(!device_ctx)throw std::bad_alloc();
            auto* av_device=reinterpret_cast<AVHWDeviceContext*>(device_ctx->data);
            auto* cuda_device=reinterpret_cast<AVCUDADeviceContext*>(av_device->hwctx);
            if(cuCtxGetCurrent(&cuda_device->cuda_ctx)!=CUDA_SUCCESS || !cuda_device->cuda_ctx)throw std::runtime_error("Missing current CUDA context");
            cuda_device->stream=nullptr;av_check(av_hwdevice_ctx_init(device_ctx));
            int status=0;
            frames_ctx=av_hwframe_ctx_alloc(device_ctx);if(!frames_ctx)throw std::bad_alloc();
            auto* frames=reinterpret_cast<AVHWFramesContext*>(frames_ctx->data);
            frames->format=AV_PIX_FMT_CUDA;frames->sw_format=AV_PIX_FMT_NV12;frames->width=w;frames->height=h;frames->initial_pool_size=0;
            av_check(av_hwframe_ctx_init(frames_ctx));
            const auto* implementation=avcodec_find_encoder_by_name("h264_nvenc");
            if(!implementation)throw Unsupported("FFmpeg was built without h264_nvenc");
            codec=avcodec_alloc_context3(implementation);if(!codec)throw std::bad_alloc();
            codec->width=w;codec->height=h;codec->pix_fmt=AV_PIX_FMT_CUDA;
            codec->time_base=AVRational{1,fps};codec->framerate=AVRational{fps,1};
            codec->colorspace=AVCOL_SPC_SMPTE170M;codec->color_range=AVCOL_RANGE_MPEG;
            codec->gop_size=fps*2;codec->max_b_frames=0;codec->thread_count=1;
            codec->hw_device_ctx=av_buffer_ref(device_ctx);codec->hw_frames_ctx=av_buffer_ref(frames_ctx);
            AVDictionary* options=nullptr;av_dict_set(&options,"preset","p1",0);av_dict_set(&options,"tune","ull",0);
            av_dict_set(&options,"rc","constqp",0);av_dict_set(&options,"qp","23",0);av_dict_set(&options,"zerolatency","1",0);
            status=avcodec_open2(codec,implementation,&options);av_dict_free(&options);av_check(status);
        }catch(...){close();throw;}
    }
    void close(){avcodec_free_context(&codec);av_buffer_unref(&frames_ctx);av_buffer_unref(&device_ctx);}
    ~GpuEncoder(){cudaSetDevice(device);close();}
    void drain(){
        std::unique_ptr<AVPacket,AvPacketDeleter> packet(av_packet_alloc());if(!packet)throw std::bad_alloc();
        while(true){int status=avcodec_receive_packet(codec,packet.get());if(status==AVERROR(EAGAIN)||status==AVERROR_EOF)break;av_check(status);
            output.insert(output.end(),packet->data,packet->data+packet->size);av_packet_unref(packet.get());}
    }
    void encode(const ImageHandle* source){
        output.clear();check(cudaSetDevice(device));
        if(flushed)throw std::invalid_argument("GPU encoder has been flushed");
        if(!source){av_check(avcodec_send_frame(codec,nullptr));flushed=true;drain();return;}
        auto image=*source;
        if(image->width!=width||image->height!=height||image->factory->device!=device)throw std::invalid_argument("GPU encoder image shape or device mismatch");
        if(image->dirty && image.use_count()!=2)throw std::invalid_argument("GPU image is in use; copy it before encoding");
        image->nv12();image->sync();
        std::unique_ptr<AVFrame,AvFrameDeleter> frame(av_frame_alloc());if(!frame)throw std::bad_alloc();
        frame->format=AV_PIX_FMT_CUDA;frame->width=width;frame->height=height;frame->pts=sequence++;
        frame->hw_frames_ctx=av_buffer_ref(frames_ctx);
        auto* owner=new ImageHandle(image);
        frame->buf[0]=av_buffer_create(static_cast<uint8_t*>(image->y),size_t(image->pitch)*height*3/2,image_buffer_free,owner,AV_BUFFER_FLAG_READONLY);
        if(!frame->buf[0]){delete owner;throw std::bad_alloc();}
        frame->data[0]=static_cast<uint8_t*>(image->y);frame->data[1]=static_cast<uint8_t*>(image->uv);
        frame->linesize[0]=frame->linesize[1]=image->pitch;
        av_check(avcodec_send_frame(codec,frame.get()));drain();
    }
};
}
extern "C" void* ci_encoder_create(int w,int h,int fps,int device){GpuEncoder* result=nullptr;guarded([&]{result=new GpuEncoder(w,h,fps,device);});return result;}
extern "C" void ci_encoder_destroy(void* p){delete static_cast<GpuEncoder*>(p);}
extern "C" int ci_encoder_encode(void* p,void* image,const uint8_t** bytes,size_t* size){return guarded([&]{auto& encoder=*static_cast<GpuEncoder*>(p);encoder.encode(static_cast<ImageHandle*>(image));*bytes=encoder.output.data();*size=encoder.output.size();});}
#else
extern "C" void* ci_encoder_create(int,int,int,int){guarded([]{throw Unsupported("Wheel needs the gpu-media feature");});return nullptr;}
extern "C" void ci_encoder_destroy(void*){}
extern "C" int ci_encoder_encode(void*,void*,const uint8_t**,size_t*){return guarded([]{throw Unsupported("Wheel needs the gpu-media feature");});}
#endif
