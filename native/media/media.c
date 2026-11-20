// Narrow libav ABI only. Ownership, scheduling, backpressure and policy are
// Rust.
#include <errno.h>
#include <limits.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/display.h>
#include <libavutil/hwcontext.h>
#ifdef CHD_CUDA
#include <cuda.h>
#include <libavutil/hwcontext_cuda.h>
#endif
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static _Thread_local char error_text[256];
const char *cm_error(void) { return error_text; }
static int fail(const char *stage, int code) {
  char details[128];
  av_strerror(code, details, sizeof(details));
  snprintf(error_text, sizeof(error_text), "%s: %s", stage, details);
  return -1;
}
int cm_init(void) {
  av_log_set_level(AV_LOG_QUIET);
  if ((avformat_version() >> 16) != LIBAVFORMAT_VERSION_MAJOR ||
      (avcodec_version() >> 16) != LIBAVCODEC_VERSION_MAJOR ||
      (avutil_version() >> 16) != LIBAVUTIL_VERSION_MAJOR ||
      (swscale_version() >> 16) != LIBSWSCALE_VERSION_MAJOR)
    return fail("FFmpeg headers/runtime mismatch", AVERROR(EINVAL));
  return avformat_network_init();
}

static int rtsp_source(const char *source) {
  return !strncmp(source, "rtsp://", 7) || !strncmp(source, "rtsps://", 8);
}

static int open_media_input(AVFormatContext **input, const char *source,
                            AVDictionary **options) {
  if (rtsp_source(source)) {
    av_dict_set(options, "rtsp_transport", "tcp", 0);
    av_dict_set(options, "timeout", "5000000", 0);
  }
  int result = avformat_open_input(input, source, NULL, options);
  av_dict_free(options);
  return result;
}

static const AVCodecParameters *first_video(const AVFormatContext *input) {
  for (unsigned int i = 0; i < input->nb_streams; i++) {
    const AVCodecParameters *parameters = input->streams[i]->codecpar;
    if (parameters->codec_type == AVMEDIA_TYPE_VIDEO)
      return parameters;
  }
  return NULL;
}

int cm_probe_video_dimensions(const char *source, int (*interrupt)(void *),
                              void *opaque, int *width, int *height) {
  AVFormatContext *input = avformat_alloc_context();
  if (!input)
    return fail("source probe", AVERROR(ENOMEM));
  input->interrupt_callback = (AVIOInterruptCB){interrupt, opaque};
  AVDictionary *options = NULL;
  int result = open_media_input(&input, source, &options);
  if (result < 0)
    goto close;
  const AVCodecParameters *video = first_video(input);
  if (!video || video->width <= 0 || video->height <= 0) {
    result = avformat_find_stream_info(input, NULL);
    if (result < 0)
      goto close;
    video = first_video(input);
  }
  if (video && video->width > 0 && video->height > 0) {
    *width = video->width;
    *height = video->height;
  } else {
    result = AVERROR_INVALIDDATA;
  }
close:
  avformat_close_input(&input);
  return result < 0 ? fail("source probe", result) : 0;
}

typedef struct {
  AVFormatContext *input;
  AVCodecContext *codec;
  AVPacket *packet;
  AVFrame *frame;
  struct SwsContext *scale;
  int stream;
  int flushed;
  int gpu;
  int primary_retained;
  int packet_clock_rate;
  void (*packet_tap)(void *, void *);
  void *packet_opaque;
} Decoder;
void cm_decoder_free(void *pointer) {
  Decoder *d = pointer;
  if (!d)
    return;
  sws_freeContext(d->scale);
  av_frame_free(&d->frame);
  av_packet_free(&d->packet);
  avcodec_free_context(&d->codec);
  avformat_close_input(&d->input);
#ifdef CHD_CUDA
  if (d->primary_retained) {
    CUdevice device;
    if (cuDeviceGet(&device, d->gpu) == CUDA_SUCCESS) cuDevicePrimaryCtxRelease(device);
  }
#endif
  free(d);
}
static enum AVPixelFormat cuda_format(AVCodecContext *codec,
                                     const enum AVPixelFormat *formats) {
  (void)codec;
  for (; *formats != AV_PIX_FMT_NONE; ++formats)
    if (*formats == AV_PIX_FMT_CUDA)
      return *formats;
  return AV_PIX_FMT_NONE;
}
static int open_codec(Decoder *d, const AVCodec *codec, int gpu) {
  int result;
  d->codec->thread_count = 1;
  d->codec->thread_type = FF_THREAD_SLICE;
  d->codec->max_pixels = 64 * 1024 * 1024;
  d->gpu = gpu;
  if (gpu >= 0) {
#ifdef CHD_CUDA
    char device[24];
    snprintf(device, sizeof(device), "%d", gpu);
    CUdevice ordinal; CUcontext primary, previous;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&ordinal, gpu) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&primary, ordinal) != CUDA_SUCCESS) {
      result = AVERROR_EXTERNAL; return result;
    }
    d->primary_retained = 1;
    if (cuCtxPushCurrent(primary) != CUDA_SUCCESS) {
      result = AVERROR_EXTERNAL; return result;
    }
    // FFmpeg's primary-context mode insists on BLOCKING_SYNC flags. Reuse the
    // explicitly retained current primary context instead, preserving CUDA
    // runtime/TensorRT scheduling flags and one context across all cameras.
    result = av_hwdevice_ctx_create(&d->codec->hw_device_ctx,
                                   AV_HWDEVICE_TYPE_CUDA, device, NULL,
                                   AV_CUDA_USE_CURRENT_CONTEXT);
    cuCtxPopCurrent(&previous);
    if (result < 0)
      return result;
    d->codec->get_format = cuda_format;
    d->codec->extra_hw_frames = 6;
#else
    result = AVERROR(ENOSYS);
    return result;
#endif
  }
  result = avcodec_open2(d->codec, codec, NULL);
  if (result < 0)
    return result;
  return 0;
}
void *cm_decoder_open(const char *source, int gpu, int (*interrupt)(void *),
                      void *opaque) {
  Decoder *d = calloc(1, sizeof(*d));
  if (!d) {
    fail("decoder allocation", AVERROR(ENOMEM));
    return NULL;
  }
  d->input = avformat_alloc_context();
  AVDictionary *options = NULL;
  int result = 0;
  if (!d->input) {
    result = AVERROR(ENOMEM);
    goto failed;
  }
  d->input->interrupt_callback = (AVIOInterruptCB){interrupt, opaque};
  d->input->probesize = 1048576;
  d->input->max_analyze_duration = 1000000;
  if (rtsp_source(source)) {
    av_dict_set(&options, "reorder_queue_size", "0", 0);
    av_dict_set(&options, "allowed_media_types", "video", 0);
  }
  result = open_media_input(&d->input, source, &options);
  if (result < 0)
    goto failed;
  result = avformat_find_stream_info(d->input, NULL);
  if (result < 0)
    goto failed;
  const AVCodec *codec = NULL;
  d->stream =
      av_find_best_stream(d->input, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
  if (d->stream < 0) {
    result = d->stream;
    goto failed;
  }
  // The previous CLI autorotated. Reject transforms until the native path
  // implements them; silently returning unrotated detections is unsafe.
  size_t matrix_size = 0;
  const uint8_t *matrix_data = NULL;
#if LIBAVCODEC_VERSION_MAJOR >= 61
  const AVPacketSideData *matrix = av_packet_side_data_get(
      d->input->streams[d->stream]->codecpar->coded_side_data,
      d->input->streams[d->stream]->codecpar->nb_coded_side_data,
      AV_PKT_DATA_DISPLAYMATRIX);
  if (matrix) {
    matrix_data = matrix->data;
    matrix_size = matrix->size;
  }
#else
  matrix_data = av_stream_get_side_data(
      d->input->streams[d->stream], AV_PKT_DATA_DISPLAYMATRIX, &matrix_size);
#endif
  if (matrix_data && matrix_size >= 9 * sizeof(int32_t)) {
    const int32_t identity[9] = {65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824};
    if (memcmp(matrix_data, identity, sizeof(identity))) {
      result = AVERROR(ENOSYS);
      goto failed;
    }
  }
  d->codec = avcodec_alloc_context3(codec);
  if (!d->codec) {
    result = AVERROR(ENOMEM);
    goto failed;
  }
  result = avcodec_parameters_to_context(
      d->codec, d->input->streams[d->stream]->codecpar);
  if (result < 0)
    goto failed;
  result = open_codec(d, codec, gpu);
  if (result < 0) goto failed;
  d->packet = av_packet_alloc();
  d->frame = av_frame_alloc();
  if (!d->packet || !d->frame) {
    result = AVERROR(ENOMEM);
    goto failed;
  }
  return d;
failed:
  fail("decoder open", result);
  cm_decoder_free(d);
  return NULL;
}
// Packet-only decoder: no AVFormatContext, socket or RTSP parser. The Rust
// transport owns networking, depacketization, timestamps and cancellation.
void *cm_packets_open(const char *name, const uint8_t *extra, size_t length,
                     int clock_rate, double fps, int gpu) {
  Decoder *d=calloc(1,sizeof(*d));
  if (!d) { fail("packet decoder allocation",AVERROR(ENOMEM)); return NULL; }
  enum AVCodecID id=!strcmp(name,"h264") ? AV_CODEC_ID_H264
                   : !strcmp(name,"hevc") ? AV_CODEC_ID_HEVC : AV_CODEC_ID_NONE;
  const AVCodec *codec=avcodec_find_decoder(id);
  int result=AVERROR(EINVAL);
  if (!codec || clock_rate<=0 || length>INT_MAX-AV_INPUT_BUFFER_PADDING_SIZE) goto failed_packets;
  d->codec=avcodec_alloc_context3(codec);
  if (!d->codec) {result=AVERROR(ENOMEM);goto failed_packets;}
  d->packet_clock_rate=clock_rate;
  d->codec->pkt_timebase=(AVRational){1,clock_rate};
  if (fps>0 && isfinite(fps)) d->codec->framerate=av_d2q(fps,1000000);
  if (length) {
    d->codec->extradata=av_mallocz(length+AV_INPUT_BUFFER_PADDING_SIZE);
    if (!d->codec->extradata) {result=AVERROR(ENOMEM);goto failed_packets;}
    memcpy(d->codec->extradata,extra,length);d->codec->extradata_size=(int)length;
  }
  result=open_codec(d,codec,gpu);
  if (result<0) goto failed_packets;
  d->packet=av_packet_alloc();d->frame=av_frame_alloc();
  if (!d->packet || !d->frame) {result=AVERROR(ENOMEM);goto failed_packets;}
  return d;
failed_packets:
  fail("packet decoder open",result);cm_decoder_free(d);return NULL;
}
// 0=accepted, 1=drain output then retry, -1=error. Always copy into padded,
// reference-counted AVPacket storage; the Rust input may be released on return.
int cm_packets_send(void *pointer,const uint8_t *bytes,size_t length,int64_t pts,int key) {
  Decoder *d=pointer;
  if (!length || length>INT_MAX-AV_INPUT_BUFFER_PADDING_SIZE)
    return fail("packet capacity",AVERROR(EINVAL));
  av_packet_unref(d->packet);
  int result=av_new_packet(d->packet,(int)length);
  if (result<0) return fail("packet allocation",result);
  memcpy(d->packet->data,bytes,length);
  d->packet->pts=pts;d->packet->dts=AV_NOPTS_VALUE;
  if (key) d->packet->flags|=AV_PKT_FLAG_KEY;
  result=avcodec_send_packet(d->codec,d->packet);
  av_packet_unref(d->packet);
  if (result==AVERROR(EAGAIN)) return 1;
  return result<0 ? fail("packet send",result) : 0;
}
int cm_packets_flush(void *pointer) {
  Decoder *d=pointer;int result=avcodec_send_packet(d->codec,NULL);
  if (result==AVERROR(EAGAIN)) return 1;
  return result<0 && result!=AVERROR_EOF ? fail("packet flush",result) : 0;
}
static int decoded_info(Decoder *d,double *pts,double *fps) {
  AVRational time_base=d->input ? d->input->streams[d->stream]->time_base
                                : (AVRational){1,d->packet_clock_rate};
  *pts=d->frame->best_effort_timestamp==AV_NOPTS_VALUE ? NAN
       : d->frame->best_effort_timestamp*av_q2d(time_base);
  AVRational rate=d->input ? av_guess_frame_rate(d->input,d->input->streams[d->stream],d->frame)
                           : d->codec->framerate;
  *fps=rate.num>0 && rate.den>0 ? av_q2d(rate) : 0.;
  if (d->gpu>=0) {
    AVHWFramesContext *frames=d->frame->hw_frames_ctx
      ? (AVHWFramesContext *)d->frame->hw_frames_ctx->data : NULL;
    if (d->frame->format!=AV_PIX_FMT_CUDA || !frames || frames->sw_format!=AV_PIX_FMT_NV12)
      return fail("NVDEC requires 8-bit NV12 output",AVERROR(ENOSYS));
  }
  return 1;
}
// 1=frame, 0=need packet, 2=EOF, -1=error.
int cm_packets_receive(void *pointer,double *pts,double *fps) {
  Decoder *d=pointer;av_frame_unref(d->frame);
  int result=avcodec_receive_frame(d->codec,d->frame);
  if (result==AVERROR(EAGAIN)) return 0;
  if (result==AVERROR_EOF) return 2;
  return result<0 ? fail("packet receive",result) : decoded_info(d,pts,fps);
}
// 1=frame ready, 0=EOF, -1=error. Every packet/frame has exactly one owner.
int cm_decoder_next(void *pointer, double *pts, double *fps) {
  Decoder *d = pointer;
  av_frame_unref(d->frame);
  for (;;) {
    int result = avcodec_receive_frame(d->codec, d->frame);
    if (result >= 0) {
      return decoded_info(d,pts,fps);
    }
    if (result == AVERROR_EOF)
      return 0;
    if (result != AVERROR(EAGAIN))
      return fail("decode frame", result);
    if (d->flushed)
      return 0;
    do {
      av_packet_unref(d->packet);
      result = av_read_frame(d->input, d->packet);
    } while (result >= 0 && d->packet->stream_index != d->stream);
    if (result < 0) {
      if (result != AVERROR_EOF)
        return fail("read packet", result);
      d->flushed = 1;
      result = avcodec_send_packet(d->codec, NULL);
    } else {
      if (d->packet_tap) d->packet_tap(d->packet_opaque,d);
      result = avcodec_send_packet(d->codec, d->packet);
    }
    av_packet_unref(d->packet);
    if (result < 0 && result != AVERROR_EOF)
      return fail("submit packet", result);
  }
}
void cm_decoder_packet_tap(void *pointer,void (*tap)(void*,void*),void *opaque) {
  Decoder *d=pointer;d->packet_tap=tap;d->packet_opaque=opaque;
}
typedef struct {AVPacket *packet;AVCodecParameters *parameters;AVRational clock;double fps;} EncodedPacket;
void cm_packet_free(void *pointer) {
  EncodedPacket *p=pointer;if(!p)return;
  av_packet_free(&p->packet);avcodec_parameters_free(&p->parameters);free(p);
}
void *cm_packet_retain(void *pointer,int *width,int *height,double *fps,int *key) {
  Decoder *d=pointer;if(!d->input)return NULL;
  AVStream *s=d->input->streams[d->stream];
  EncodedPacket *p=calloc(1,sizeof(*p));if(!p)return NULL;
  p->packet=av_packet_clone(d->packet);p->parameters=avcodec_parameters_alloc();
  if(!p->packet||!p->parameters||avcodec_parameters_copy(p->parameters,s->codecpar)<0){cm_packet_free(p);return NULL;}
  p->clock=s->time_base;p->fps=av_q2d(av_guess_frame_rate(d->input,s,NULL));
  *width=p->parameters->width;*height=p->parameters->height;*fps=p->fps;
  *key=!!(p->packet->flags&AV_PKT_FLAG_KEY);
  return p;
}
typedef struct {AVFormatContext *output;int opened;} Remux;
void cm_remux_free(void *pointer) {
  Remux *m=pointer;if(!m)return;
  if(m->output){if(m->opened)av_write_trailer(m->output);
    if(m->output->pb) avio_closep(&m->output->pb);
    avformat_free_context(m->output);}
  free(m);
}
void *cm_remux_open(void *packet,const char *target,int (*interrupt)(void*),void *opaque) {
  EncodedPacket *p=packet;Remux *m=calloc(1,sizeof(*m));if(!m)return NULL;
  int result=avformat_alloc_output_context2(&m->output,NULL,"rtsp",target);
  AVDictionary *options=NULL;
  if(result<0||!m->output)goto failed_remux;
  m->output->interrupt_callback=(AVIOInterruptCB){interrupt,opaque};
  AVStream *s=avformat_new_stream(m->output,NULL);if(!s){result=AVERROR(ENOMEM);goto failed_remux;}
  result=avcodec_parameters_copy(s->codecpar,p->parameters);if(result<0)goto failed_remux;
  s->codecpar->codec_tag=0;s->time_base=p->clock;
  av_dict_set(&options,"rtsp_transport","tcp",0);av_dict_set(&options,"timeout","5000000",0);
  result=avformat_write_header(m->output,&options);av_dict_free(&options);
  if(result<0) goto failed_remux;
  m->opened=1;
  return m;
failed_remux:
  av_dict_free(&options);fail("remux open",result);cm_remux_free(m);return NULL;
}
int cm_remux_write(void *pointer,void *packet) {
  Remux *m=pointer;EncodedPacket *p=packet;AVPacket *copy=av_packet_clone(p->packet);
  if(!copy)return fail("remux packet",AVERROR(ENOMEM));
  av_packet_rescale_ts(copy,p->clock,m->output->streams[0]->time_base);copy->stream_index=0;copy->pos=-1;
  int result=av_interleaved_write_frame(m->output,copy);av_packet_free(&copy);
  return result<0?fail("remux write",result):0;
}
static const int *coefficients(enum AVColorSpace space) {
  switch (space) {
  case AVCOL_SPC_BT709:
    return sws_getCoefficients(SWS_CS_ITU709);
  case AVCOL_SPC_FCC:
    return sws_getCoefficients(SWS_CS_FCC);
  case AVCOL_SPC_SMPTE240M:
    return sws_getCoefficients(SWS_CS_SMPTE240M);
  case AVCOL_SPC_BT2020_NCL:
  case AVCOL_SPC_BT2020_CL:
    return sws_getCoefficients(SWS_CS_BT2020);
  default:
    return sws_getCoefficients(SWS_CS_DEFAULT);
  }
}
static int frame_bgr(AVFrame *f, struct SwsContext **scale, uint8_t *target,
                     size_t length, int width, int height) {
  if (width <= 0 || height <= 0 || length < (size_t)width * height * 3 + 64)
    return fail("BGR capacity", AVERROR(EINVAL));
  *scale = sws_getCachedContext(*scale, f->width, f->height, f->format,
                                  width, height, AV_PIX_FMT_BGR24, SWS_BILINEAR,
                                  NULL, NULL, NULL);
  if (!*scale)
    return fail("BGR scaler", AVERROR(ENOMEM));
  int result = sws_setColorspaceDetails(
      *scale, coefficients(f->colorspace), f->color_range == AVCOL_RANGE_JPEG,
      sws_getCoefficients(SWS_CS_DEFAULT), 1, 0, 1 << 16, 1 << 16);
  if (result < 0)
    return fail("BGR colour matrix", result);
  uint8_t *planes[4] = {target, NULL, NULL, NULL};
  int strides[4] = {width * 3, 0, 0, 0};
  result = sws_scale(*scale, (const uint8_t *const *)f->data, f->linesize, 0,
                     f->height, planes, strides);
  return result == height ? 0 : fail("BGR conversion", AVERROR(EINVAL));
}
int cm_decoder_bgr(void *pointer, uint8_t *target, size_t length, int width,
                   int height) {
  Decoder *d = pointer;
  return frame_bgr(d->frame, &d->scale, target, length, width, height);
}

typedef struct { AVFrame *frame; } HostFrame;
typedef struct {
  uintptr_t y, u, v;
  int width, height, yp, up, vp, colorspace, range, band_format;
} YuvInfo;
enum HostBandFormat { HOST_BANDS_UNSUPPORTED, HOST_BANDS_420, HOST_BANDS_422_OR_444 };
static enum HostBandFormat host_band_format(enum AVPixelFormat format) {
  switch (format) {
  case AV_PIX_FMT_YUV420P: case AV_PIX_FMT_YUVJ420P:
    return HOST_BANDS_420;
  case AV_PIX_FMT_YUV422P: case AV_PIX_FMT_YUVJ422P:
  case AV_PIX_FMT_YUV444P: case AV_PIX_FMT_YUVJ444P:
    return HOST_BANDS_422_OR_444;
  default:
    return HOST_BANDS_UNSUPPORTED;
  }
}
void cm_host_free(void *pointer) {
  HostFrame *f = pointer;
  if (f) { av_frame_free(&f->frame); free(f); }
}
void *cm_decoder_retain_host(void *pointer, YuvInfo *info) {
  Decoder *d = pointer; AVFrame *f = d->frame;
  if (d->gpu >= 0 || f->width <= 0 || f->height <= 0) {
    fail("Not a CPU frame",AVERROR(EINVAL)); return NULL;
  }
  HostFrame *owner = calloc(1, sizeof(*owner));
  if (!owner) { fail("Host frame allocation",AVERROR(ENOMEM)); return NULL; }
  owner->frame = av_frame_clone(f);
  if (!owner->frame) { free(owner); fail("Host frame reference",AVERROR(ENOMEM)); return NULL; }
  *info = (YuvInfo){(uintptr_t)f->data[0],(uintptr_t)f->data[1],(uintptr_t)f->data[2],
    f->width,f->height,f->linesize[0],f->linesize[1],f->linesize[2],f->colorspace,
    f->color_range,
    host_band_format(f->format)};
  return owner;
}
int cm_host_bgr(void *pointer,uint8_t *target,size_t length,int width,int height) {
  HostFrame *f=pointer; struct SwsContext *scale=NULL;
  int result=frame_bgr(f->frame,&scale,target,length,width,height);
  sws_freeContext(scale); return result;
}

typedef struct {
  struct SwsContext *scale;
  int width, rows, format;
  enum AVColorSpace space;
  enum AVColorRange range;
} HostBandScaler;

void cm_host_band_free(void *pointer) {
  HostBandScaler *band = pointer;
  if (band) {
    sws_freeContext(band->scale);
    free(band);
  }
}

int cm_host_bgr_band(const void *pointer, void **scaler, int row, int rows,
                     uint8_t *target, size_t length) {
  const HostFrame *owner = pointer;
  const AVFrame *frame = owner->frame;
  int width = frame->width;
  if (host_band_format(frame->format) == HOST_BANDS_UNSUPPORTED ||
      width <= 0 || width > INT_MAX / 3 || row < 0 || row % 2 != 0 ||
      rows <= 0 || rows % 2 != 0 || row > frame->height - rows ||
      length < (size_t)width * rows * 3 + 64)
    return fail("Host BGR scanline band", AVERROR(EINVAL));
  HostBandScaler *band = *scaler;
  if (!band) {
    band = calloc(1, sizeof(*band));
    if (!band) return fail("Host BGR scanline scaler", AVERROR(ENOMEM));
    *scaler = band;
  }
  if (!band->scale || band->width != width || band->rows != rows || band->format != frame->format ||
      band->space != frame->colorspace || band->range != frame->color_range) {
    band->scale = sws_getCachedContext(band->scale, width, rows, frame->format,
        width, rows, AV_PIX_FMT_BGR24, SWS_BILINEAR, NULL, NULL, NULL);
    if (!band->scale) return fail("Host BGR scanline context", AVERROR(ENOMEM));
    int result = sws_setColorspaceDetails(band->scale,
        coefficients(frame->colorspace), frame->color_range == AVCOL_RANGE_JPEG,
        sws_getCoefficients(SWS_CS_DEFAULT), 1, 0, 1 << 16, 1 << 16);
    if (result < 0) {
      sws_freeContext(band->scale);
      band->scale = NULL;
      return fail("Host BGR scanline colour matrix", result);
    }
    band->width = width;
    band->rows = rows;
    band->format = frame->format;
    band->space = frame->colorspace;
    band->range = frame->color_range;
  }
  // Start at the same chroma boundary as full-frame conversion.
  int chroma_row = row >> av_pix_fmt_desc_get(frame->format)->log2_chroma_h;
  const uint8_t *source[4] = {
    frame->data[0] + (size_t)row * frame->linesize[0],
    frame->data[1] + (size_t)chroma_row * frame->linesize[1],
    frame->data[2] + (size_t)chroma_row * frame->linesize[2], NULL
  };
  uint8_t *output[4] = {target, NULL, NULL, NULL};
  const int strides[4] = {width * 3, 0, 0, 0};
  int result = sws_scale(band->scale, source, frame->linesize, 0, rows, output, strides);
  return result == rows ? 0 : fail("Host BGR scanline conversion", AVERROR(EINVAL));
}

int cm_cpu_lut_fill(uint8_t *lookup, size_t length, int colorspace, int range) {
  const int width = 512, height = 512;
  if (length != (size_t)256 * 256 * 256 * 3) return fail("Colour table capacity", AVERROR(EINVAL));
  uint8_t *source = av_malloc((size_t)width * height * 3 / 2 + 64);
  uint8_t *output = av_malloc((size_t)width * height * 3 + 64);

  struct SwsContext *scale = sws_getContext(width, height, AV_PIX_FMT_YUV420P,
      width, height, AV_PIX_FMT_BGR24, SWS_BILINEAR, NULL, NULL, NULL);

  int result = AVERROR(ENOMEM);
  if (!source || !output || !lookup || !scale) goto done;
  result = sws_setColorspaceDetails(scale, coefficients(colorspace), range == AVCOL_RANGE_JPEG,
      sws_getCoefficients(SWS_CS_DEFAULT), 1, 0, 1 << 16, 1 << 16);
  if (result < 0) goto done;
  uint8_t *u = source + width * height, *v = u + width * height / 4;
  for (int row = 0; row < height; ++row)
    for (int x = 0; x < width; ++x) source[row * width + x] = x / 2;
  for (int row = 0; row < height / 2; ++row) memset(u + row * width / 2, row, width / 2);
  const uint8_t *planes[4] = {source, u, v, NULL};
  int strides[4] = {width, width / 2, width / 2, 0};
  uint8_t *targets[4] = {output, NULL, NULL, NULL};
  int target_strides[4] = {width * 3, 0, 0, 0};
  for (int vv = 0; vv < 256; ++vv) {
    memset(v, vv, width * height / 4);
    if (sws_scale(scale, planes, strides, 0, height, targets, target_strides) != height) {
      result = AVERROR_EXTERNAL; goto done;
    }
    for (int uu = 0; uu < 256; ++uu) for (int yy = 0; yy < 256; ++yy) {
      const uint8_t *pixel = output + ((size_t)uu * 2 * width + yy * 2) * 3;
      if (memcmp(pixel, pixel + 3, 3) || memcmp(pixel, pixel + width * 3, 3) ||
          memcmp(pixel, pixel + width * 3 + 3, 3)) {
        result = AVERROR(ENOSYS); goto done;
      }
      memcpy(lookup + (((size_t)uu * 256 + vv) * 256 + yy) * 3, pixel, 3);
    }
  }
  result = 0;
done:
  sws_freeContext(scale); av_free(source); av_free(output);
  return result < 0 ? fail("YUV420P colour-table calibration", result) : 0;
}

int cm_host_freeze_sample(const void *host, uint8_t *sample, size_t capacity) {
  const AVFrame *frame = ((const HostFrame *)host)->frame;
  const AVPixFmtDescriptor *format = av_pix_fmt_desc_get(frame->format);
  if (!format || !format->nb_components || frame->width < 1 || frame->height < 1 ||
      capacity > INT_MAX || capacity < 8)
    return fail("Freeze sample shape", AVERROR(EINVAL));

  const int components = FFMIN(format->nb_components, 3);
  const int rgb = format->flags & AV_PIX_FMT_FLAG_RGB;
  const uint8_t *planes[4] = {frame->data[0], frame->data[1], frame->data[2], frame->data[3]};
  size_t total = 0;
  for (int component = 0; component < components; ++component) {
    // Sample the luma plane densely and retain chroma samples so an
    // isoluminant colour change also resets the freeze timer.
    const int chroma = !rgb && component != 0;
    const int width = AV_CEIL_RSHIFT(frame->width, chroma ? format->log2_chroma_w : 0);
    const int height = AV_CEIL_RSHIFT(frame->height, chroma ? format->log2_chroma_h : 0);
    const uint64_t pixels = (uint64_t)width * height;
    const unsigned depth = format->comp[component].depth;
    const size_t budget = components == 1 ? capacity :
        rgb ? capacity / components :
        (component == 0 ? capacity - 2 * (capacity / 8) : capacity / 8);
    const size_t count = FFMIN(pixels, budget);
    for (size_t i = 0; i < count; ++i) {
      const uint64_t index = i * pixels / count;
      uint32_t value = 0;
      av_read_image_line2(&value, planes, frame->linesize,
                          format, index % width, index / width, component, 1, 0, 4);
      sample[total++] = depth >= 8 ? value >> (depth - 8) :
          value * 255 / ((1u << depth) - 1);
    }
  }
  return (int)total;
}

static int jpeg_scale(AVFrame *dst, const AVFrame *src, const int *matrix,
                      int full_range) {
  struct SwsContext *scale = sws_getContext(src->width, src->height, src->format,
      dst->width, dst->height, dst->format, SWS_BILINEAR, NULL, NULL, NULL);
  if (!scale) return AVERROR(ENOMEM);
  int result = sws_setColorspaceDetails(scale, matrix,
      src->color_range == AVCOL_RANGE_JPEG, matrix, full_range, 0, 1 << 16, 1 << 16);
  if (result >= 0 && sws_scale(scale, (const uint8_t *const *)src->data, src->linesize,
           0, src->height, dst->data, dst->linesize) != dst->height)
    result = AVERROR(EINVAL);
  sws_freeContext(scale);
  return result;
}

static int jpeg_colour(AVFrame *target, const AVFrame *source) {
  const int *jpeg_matrix = sws_getCoefficients(SWS_CS_ITU601);
  const int *source_matrix = coefficients(source->colorspace);
  const AVPixFmtDescriptor *format = av_pix_fmt_desc_get(source->format);
  if (format && (format->flags & AV_PIX_FMT_FLAG_RGB))
    source_matrix = jpeg_matrix;
  if (!memcmp(source_matrix, jpeg_matrix, 4 * sizeof(int)))
    return jpeg_scale(target, source, jpeg_matrix, 1);

  // Changing the matrix via swscale uses packed RGB internally. Bound that
  // workspace to a stripe and keep any intermediate resize in planar YUV.
  AVFrame *resized = NULL;
  struct SwsContext *scale = NULL;
  int result = AVERROR(ENOMEM);
  if (source->width != target->width || source->height != target->height ||
      (source->format != AV_PIX_FMT_YUV420P && source->format != AV_PIX_FMT_YUVJ420P)) {
    resized = av_frame_alloc();
    if (!resized) goto done;
    resized->width = target->width;
    resized->height = target->height;
    resized->format = AV_PIX_FMT_YUV420P;
    resized->color_range = source->color_range;
    result = av_frame_get_buffer(resized, 32);
    if (result < 0) goto done;
    result = jpeg_scale(resized, source, source_matrix,
                        resized->color_range == AVCOL_RANGE_JPEG);
    if (result < 0) goto done;
    source = resized;
  }
  for (int row = 0; row < target->height; row += 32) {
    const int rows = FFMIN(32, target->height - row);
    scale = sws_getCachedContext(scale, source->width, rows, source->format,
        target->width, rows, target->format, SWS_BILINEAR, NULL, NULL, NULL);
    if (!scale) { result = AVERROR(ENOMEM); goto done; }
    result = sws_setColorspaceDetails(scale, source_matrix,
        source->color_range == AVCOL_RANGE_JPEG, jpeg_matrix, 1, 0, 1 << 16, 1 << 16);
    if (result < 0) goto done;
    const uint8_t *src[4] = {
      source->data[0] + (ptrdiff_t)row * source->linesize[0],
      source->data[1] + (ptrdiff_t)(row / 2) * source->linesize[1],
      source->data[2] + (ptrdiff_t)(row / 2) * source->linesize[2], NULL
    };
    uint8_t *dst[4] = {
      target->data[0] + (ptrdiff_t)row * target->linesize[0],
      target->data[1] + (ptrdiff_t)(row / 2) * target->linesize[1],
      target->data[2] + (ptrdiff_t)(row / 2) * target->linesize[2], NULL
    };
    if (sws_scale(scale, src, source->linesize, 0, rows, dst, target->linesize) != rows) {
      result = AVERROR(EINVAL);
      goto done;
    }
  }
done:
  sws_freeContext(scale);
  av_frame_free(&resized);
  return result;
}

void *cm_host_jpeg(const void *host, int width, int height,
                   const uint8_t **data, size_t *length) {
  const AVFrame *source = ((const HostFrame *)host)->frame;
  const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
  if (!codec) { fail("JPEG codec", AVERROR_ENCODER_NOT_FOUND); return NULL; }
  AVCodecContext *encoder = avcodec_alloc_context3(codec);
  AVFrame *frame = av_frame_alloc();
  AVPacket *packet = av_packet_alloc();
  const char *stage = "JPEG allocation";
  int result = AVERROR(ENOMEM);
  if (!encoder || !frame || !packet) goto done;
  encoder->width = frame->width = width;
  encoder->height = frame->height = height;
  // The JPEG pixel format sets full range before swscale initializes its
  // fast same-size converter.
  encoder->pix_fmt = frame->format = AV_PIX_FMT_YUVJ420P;
  encoder->color_range = frame->color_range = AVCOL_RANGE_JPEG;
  encoder->colorspace = frame->colorspace = AVCOL_SPC_SMPTE170M;
  encoder->time_base = (AVRational){1, 25};
  encoder->thread_count = 1;
  encoder->flags |= AV_CODEC_FLAG_QSCALE;
  encoder->global_quality = frame->quality = 2 * FF_QP2LAMBDA;
  frame->pts = 0;
  stage = "JPEG open";
  result = avcodec_open2(encoder, codec, NULL);
  if (result < 0) goto done;
  stage = "JPEG frame allocation";
  result = av_frame_get_buffer(frame, 32);
  if (result < 0) goto done;
  stage = "JPEG colour conversion";
  result = jpeg_colour(frame, source);
  if (result < 0) goto done;
  stage = "JPEG encode";
  result = avcodec_send_frame(encoder, frame);
  if (result >= 0) result = avcodec_receive_packet(encoder, packet);
done:
  av_frame_free(&frame);
  avcodec_free_context(&encoder);
  if (result < 0) {
    fail(stage, result);
    av_packet_free(&packet);
    return NULL;
  }
  *data = packet->data;
  *length = packet->size;
  return packet;
}

void cm_jpeg_free(void *pointer) {
  AVPacket *packet = pointer;
  av_packet_free(&packet);
}

typedef struct {
  uintptr_t y, uv, ready, lut;
  int width, height, yp, uvp, colorspace, range, device;
} Nv12Info;
#ifdef CHD_CUDA
typedef struct { AVFrame *frame; CUevent ready; int device; int primary_retained; } DeviceFrame;
static AVCUDADeviceContext *cuda_device(AVFrame *frame) {
  AVHWFramesContext *frames = (AVHWFramesContext *)frame->hw_frames_ctx->data;
  AVHWDeviceContext *device = (AVHWDeviceContext *)frames->device_ref->data;
  return (AVCUDADeviceContext *)device->hwctx;
}
void cm_frame_free(void *pointer) {
  DeviceFrame *f = pointer;
  if (!f) return;
  if (f->ready) {
    CUcontext previous;
    if (cuCtxPushCurrent(cuda_device(f->frame)->cuda_ctx) == CUDA_SUCCESS) {
      cuEventDestroy(f->ready);
      cuCtxPopCurrent(&previous);
    }
  }
  av_frame_free(&f->frame);
  if (f->primary_retained) {
    CUdevice device;
    if (cuDeviceGet(&device, f->device) == CUDA_SUCCESS) cuDevicePrimaryCtxRelease(device);
  }
  free(f);
}
void *cm_decoder_retain_cuda(void *pointer, Nv12Info *info) {
  Decoder *d = pointer;
  if (d->gpu < 0 || d->frame->format != AV_PIX_FMT_CUDA) {
    fail("Not a CUDA frame", AVERROR(EINVAL)); return NULL;
  }
  DeviceFrame *f = calloc(1, sizeof(*f));
  if (!f) { fail("CUDA frame allocation", AVERROR(ENOMEM)); return NULL; }
  f->frame = av_frame_clone(d->frame);
  if (!f->frame) { free(f); fail("CUDA frame reference", AVERROR(ENOMEM)); return NULL; }
  f->device = d->gpu;
  CUdevice ordinal; CUcontext primary;
  if (cuDeviceGet(&ordinal, d->gpu) != CUDA_SUCCESS ||
      cuDevicePrimaryCtxRetain(&primary, ordinal) != CUDA_SUCCESS) {
    cm_frame_free(f); fail("Retain CUDA primary context", AVERROR_EXTERNAL); return NULL;
  }
  f->primary_retained = 1;
  AVCUDADeviceContext *device = cuda_device(f->frame);
  CUcontext previous;
  CUresult result = cuCtxPushCurrent(device->cuda_ctx);
  if (result == CUDA_SUCCESS) {
    result = cuEventCreate(&f->ready, CU_EVENT_DISABLE_TIMING);
    if (result == CUDA_SUCCESS) result = cuEventRecord(f->ready, device->stream);
    cuCtxPopCurrent(&previous);
  }
  if (result != CUDA_SUCCESS) {
    cm_frame_free(f); fail("CUDA ready event", AVERROR_EXTERNAL); return NULL;
  }
  *info = (Nv12Info){(uintptr_t)f->frame->data[0], (uintptr_t)f->frame->data[1],
    (uintptr_t)f->ready, 0, f->frame->width, f->frame->height,
    f->frame->linesize[0], f->frame->linesize[1], f->frame->colorspace,
    f->frame->color_range, f->device};
  return f;
}
int cm_frame_bgr(void *pointer, uint8_t *target, size_t length) {
  DeviceFrame *f = pointer;
  AVFrame *cpu = av_frame_alloc();
  if (!cpu) return fail("download allocation", AVERROR(ENOMEM));
  int result = av_hwframe_transfer_data(cpu, f->frame, 0);
  if (result >= 0) result = av_frame_copy_props(cpu, f->frame);
  // Match the software decoder's YUV420P converter. libswscale's NV12 -> BGR
  // path reconstructs chroma differently, even with identical colour flags.
  if (result >= 0 && cpu->format == AV_PIX_FMT_NV12) {
    AVFrame *planar = av_frame_alloc();
    if (!planar) result = AVERROR(ENOMEM);
    else {
      planar->format = AV_PIX_FMT_YUV420P;
      planar->width = cpu->width; planar->height = cpu->height;
      result = av_frame_get_buffer(planar, 32);
      if (result >= 0) result = av_frame_copy_props(planar, cpu);
      if (result >= 0) {
        for (int row=0; row<cpu->height; ++row)
          memcpy(planar->data[0]+row*planar->linesize[0],cpu->data[0]+row*cpu->linesize[0],cpu->width);
        for (int row=0; row<cpu->height/2; ++row) for(int x=0;x<cpu->width/2;++x){
          planar->data[1][row*planar->linesize[1]+x]=cpu->data[1][row*cpu->linesize[1]+x*2];
          planar->data[2][row*planar->linesize[2]+x]=cpu->data[1][row*cpu->linesize[1]+x*2+1];
        }
        av_frame_free(&cpu); cpu=planar;
      } else av_frame_free(&planar);
    }
  }
  struct SwsContext *scale = NULL;
  if (result >= 0) result = frame_bgr(cpu, &scale, target, length, cpu->width, cpu->height);
  else fail("CUDA frame download", result);
  sws_freeContext(scale);
  av_frame_free(&cpu);
  return result;
}
typedef struct { CUdevice device; CUdeviceptr data; } ColorTable;
void cm_lut_free(void *pointer) {
  ColorTable *table = pointer;
  if (!table) return;
  CUcontext context, previous;
  if (cuDevicePrimaryCtxRetain(&context, table->device) == CUDA_SUCCESS) {
    if (cuCtxPushCurrent(context) == CUDA_SUCCESS) {
      if (table->data) cuMemFree(table->data);
      cuCtxPopCurrent(&previous);
    }
    cuDevicePrimaryCtxRelease(table->device);
  }
  cuDevicePrimaryCtxRelease(table->device);
  free(table);
}
// CPU calibration is shared with the Rust host-YUV path.
void *cm_lut_create(int gpu, int colorspace, int range, uintptr_t *address) {
  const size_t table_bytes = (size_t)256 * 256 * 256 * 3;
  uint8_t *lookup = av_malloc(table_bytes);
  ColorTable *table = NULL;
  int result = AVERROR(ENOMEM);
  if (!lookup) goto done;
  result = cm_cpu_lut_fill(lookup, table_bytes, colorspace, range);
  if (result < 0) goto done;
  table = calloc(1, sizeof(*table));
  if (!table) { result = AVERROR(ENOMEM); goto done; }
  CUcontext context, previous;
  if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&table->device, gpu) != CUDA_SUCCESS ||
      cuDevicePrimaryCtxRetain(&context, table->device) != CUDA_SUCCESS) {
    free(table); table = NULL; result = AVERROR_EXTERNAL; goto done;
  }
  if (cuCtxPushCurrent(context) != CUDA_SUCCESS) { result = AVERROR_EXTERNAL; goto failed_table; }
  CUresult status = cuMemAlloc(&table->data, table_bytes);
  if (status == CUDA_SUCCESS) status = cuMemcpyHtoD(table->data, lookup, table_bytes);
  cuCtxPopCurrent(&previous);
  if (status != CUDA_SUCCESS) { result = AVERROR_EXTERNAL; goto failed_table; }
  *address = table->data;
  result = 0;
  goto done;
failed_table:
  cm_lut_free(table); table = NULL;
done:
  av_free(lookup);
  if (result < 0) fail("YUV420P colour-table calibration", result);
  return table;
}
#else
void cm_frame_free(void *p) { (void)p; }
void *cm_decoder_retain_cuda(void *p, Nv12Info *info) {
  (void)p; (void)info; fail("Built without CUDA", AVERROR(ENOSYS)); return NULL;
}
int cm_frame_bgr(void *p, uint8_t *target, size_t length) {
  (void)p; (void)target; (void)length; return fail("Built without CUDA", AVERROR(ENOSYS));
}
void cm_lut_free(void *p) { (void)p; }
void *cm_lut_create(int gpu,int space,int range,uintptr_t *address) {
  (void)gpu; (void)space; (void)range; (void)address;
  fail("Built without CUDA", AVERROR(ENOSYS)); return NULL;
}
#endif

typedef struct {
  AVFormatContext *output;
  AVCodecContext *codec;
  AVStream *stream;
  AVFrame *frame;
  AVPacket *packet;
  struct SwsContext *scale;
  int opened;
  int width;
  int height;
  int frame_cached;
  int64_t next_pts;
  uint64_t written_packets;
} Encoder;
void cm_encoder_free(void *pointer) {
  Encoder *e = pointer;
  if (!e)
    return;
  sws_freeContext(e->scale);
  av_frame_free(&e->frame);
  av_packet_free(&e->packet);
  avcodec_free_context(&e->codec);
  if (e->output) {
    if (e->output->pb)
      avio_closep(&e->output->pb);
    avformat_free_context(e->output);
  }
  free(e);
}
static enum AVColorSpace encoder_matrix(enum AVColorSpace space) {
  switch (space) {
  case AVCOL_SPC_BT709:
  case AVCOL_SPC_FCC:
  case AVCOL_SPC_SMPTE240M:
  case AVCOL_SPC_BT2020_NCL:
  case AVCOL_SPC_BT470BG:
  case AVCOL_SPC_SMPTE170M:
    return space;
  case AVCOL_SPC_BT2020_CL:
    return AVCOL_SPC_BT2020_NCL;
  default:
    return AVCOL_SPC_SMPTE170M;
  }
}
void *cm_encoder_open(const char *target, int width, int height, int fps,
                      int gpu, const void *initial_host,
                      int (*interrupt)(void *), void *opaque) {
  Encoder *e = calloc(1, sizeof(*e));
  if (!e) {
    fail("encoder allocation", AVERROR(ENOMEM));
    return NULL;
  }
  AVDictionary *options = NULL;
  int result;
  int remote =
      !strncmp(target, "rtsp://", 7) || !strncmp(target, "rtsps://", 8);
  result = avformat_alloc_output_context2(&e->output, NULL,
                                          remote ? "rtsp" : NULL, target);
  if (result < 0 || !e->output) {
    result = AVERROR(EINVAL);
    goto failed;
  }
  e->output->interrupt_callback = (AVIOInterruptCB){interrupt, opaque};
  const AVCodec *codec =
      avcodec_find_encoder_by_name(gpu >= 0 ? "h264_nvenc" : "libx264");
  if (!codec) {
    result = AVERROR_ENCODER_NOT_FOUND;
    goto failed;
  }
  e->stream = avformat_new_stream(e->output, NULL);
  e->codec = avcodec_alloc_context3(codec);
  e->frame = av_frame_alloc();
  e->packet = av_packet_alloc();
  if (!e->stream || !e->codec || !e->frame || !e->packet) {
    result = AVERROR(ENOMEM);
    goto failed;
  }
  e->width = width;
  e->height = height;
  e->codec->width = width;
  e->codec->height = height;
  e->codec->time_base = (AVRational){1, fps};
  e->codec->framerate = (AVRational){fps, 1};
  e->codec->pix_fmt = AV_PIX_FMT_YUV420P;
  e->codec->gop_size = fps * 2;
  e->codec->max_b_frames = 0;
  e->codec->thread_count = 1;
  e->codec->color_range = AVCOL_RANGE_MPEG;
  if (initial_host) {
    const AVFrame *source = ((const HostFrame *)initial_host)->frame;
    // Set the matrix before writing the stream header. BGR slates and later
    // decoded frames are converted to the same limited-range profile.
    e->codec->colorspace = encoder_matrix(source->colorspace);
    e->codec->color_primaries = source->color_primaries;
    e->codec->color_trc = source->color_trc;
  }
  if (e->output->oformat->flags & AVFMT_GLOBALHEADER)
    e->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  av_dict_set(&options, "preset", gpu >= 0 ? "p1" : "ultrafast", 0);
  av_dict_set(&options, "tune", gpu >= 0 ? "ull" : "zerolatency", 0);
  if (gpu < 0) {
    av_dict_set(&options, "crf", "23", 0);
    av_dict_set(&options, "sc_threshold", "0", 0);
  } else {
    char device[32];
    snprintf(device, sizeof(device), "%d", gpu);
    av_dict_set(&options, "gpu", device, 0);
  }
  result = avcodec_open2(e->codec, codec, &options);
  av_dict_free(&options);
  if (result < 0)
    goto failed;
  e->stream->time_base = e->codec->time_base;
  result = avcodec_parameters_from_context(e->stream->codecpar, e->codec);
  if (result < 0)
    goto failed;
  e->frame->width = width;
  e->frame->height = height;
  e->frame->format = e->codec->pix_fmt;
  e->frame->colorspace = e->codec->colorspace;
  e->frame->color_range = e->codec->color_range;
  e->frame->color_primaries = e->codec->color_primaries;
  e->frame->color_trc = e->codec->color_trc;
  result = av_frame_get_buffer(e->frame, 32);
  if (result < 0)
    goto failed;
  if (!(e->output->oformat->flags & AVFMT_NOFILE)) {
    result = avio_open2(&e->output->pb, target, AVIO_FLAG_WRITE,
                        &e->output->interrupt_callback, NULL);
    if (result < 0)
      goto failed;
  }
  if (remote) {
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "pkt_size", "1200", 0);
  } else if (strstr(e->output->oformat->name, "mp4"))
    av_dict_set(&options, "movflags", "+faststart", 0);
  result = avformat_write_header(e->output, &options);
  av_dict_free(&options);
  if (result < 0)
    goto failed;
  e->opened = 1;
  return e;
failed:
  fail("encoder open", result);
  av_dict_free(&options);
  cm_encoder_free(e);
  return NULL;
}
static int packets(Encoder *e) {
  for (;;) {
    int result = avcodec_receive_packet(e->codec, e->packet);
    if (result == AVERROR(EAGAIN) || result == AVERROR_EOF)
      return 0;
    if (result < 0)
      return fail("encode packet", result);
    if (e->packet->duration <= 0)
      e->packet->duration = 1;
    av_packet_rescale_ts(e->packet, e->codec->time_base, e->stream->time_base);
    e->packet->stream_index = e->stream->index;
    result = av_interleaved_write_frame(e->output, e->packet);
    av_packet_unref(e->packet);
    if (result < 0)
      return fail("write packet", result);
    e->written_packets++;
  }
}
static int submit_encoder_frame(Encoder *e) {
  e->frame->pts = e->next_pts++;
  int result = avcodec_send_frame(e->codec, e->frame);
  if (result < 0)
    return fail("submit image", result);
  return packets(e);
}
int cm_encoder_repeat_frame(void *pointer) {
  Encoder *e = pointer;
  if (!e->opened || !e->frame_cached)
    return fail("no cached encoder frame", AVERROR(EINVAL));
  // The encoder holds its own frame reference. Only the submission timestamp
  // changes here; the original converted plane buffers remain immutable.
  int result = submit_encoder_frame(e);
  e->frame_cached = result == 0;
  return result;
}
static int scale_encoder_planes(Encoder *e, const uint8_t *const source[4],
                                const int stride[4], int width, int height,
                                enum AVPixelFormat format,
                                enum AVColorSpace space, int full_range) {
  e->frame_cached = 0;
  e->scale = sws_getCachedContext(e->scale, width, height, format,
      e->width, e->height, e->codec->pix_fmt, SWS_BILINEAR, NULL, NULL, NULL);
  if (!e->scale) return fail("encoder scaler", AVERROR(ENOMEM));
  int result = sws_setColorspaceDetails(e->scale, coefficients(space), full_range,
      coefficients(e->codec->colorspace), 0, 0, 1 << 16, 1 << 16);
  if (result < 0) return fail("encoder colour matrix", result);
  result = av_frame_make_writable(e->frame);
  if (result < 0)
    return fail("encoder buffer", result);
  result = sws_scale(e->scale, source, stride, 0, height, e->frame->data,
                     e->frame->linesize);
  if (result != e->height)
    return fail("encoder colour conversion", AVERROR(EINVAL));
  return 0;
}
int cm_encoder_submit(Encoder *e) {
  int result = submit_encoder_frame(e);
  e->frame_cached = result == 0;
  return result;
}
int cm_encoder_write_scaled(void *pointer, const uint8_t *pixels, size_t length, int width, int height) {
  if (width <= 0 || height <= 0 || (uint64_t)width * height > 64 * 1024 * 1024 ||
      length != (size_t)width * height * 3 + 64)
    return fail("encoder image shape", AVERROR(EINVAL));
  const uint8_t *source[4] = {pixels, NULL, NULL, NULL};
  const int stride[4] = {width * 3, 0, 0, 0};
  int result = scale_encoder_planes(pointer, source, stride, width, height,
                                    AV_PIX_FMT_BGR24, AVCOL_SPC_UNSPECIFIED, 1);
  return result < 0 ? result : cm_encoder_submit(pointer);
}
typedef struct {
  uintptr_t y, u, v;
  int width, height, yp, up, vp, colorspace;
} EncoderPlaneInfo;
int cm_encoder_prepare_host(void *pointer, const void *host, EncoderPlaneInfo *target) {
  Encoder *e = pointer;
  const AVFrame *frame = ((const HostFrame *)host)->frame;
  // Resize retained decoded planes into the reusable encoder frame.
  int result = scale_encoder_planes(e, (const uint8_t *const *)frame->data,
      frame->linesize, frame->width, frame->height, frame->format,
      frame->colorspace, frame->color_range == AVCOL_RANGE_JPEG);
  if (result >= 0 && target) {
    const AVFrame *output = e->frame;
    *target = (EncoderPlaneInfo){(uintptr_t)output->data[0], (uintptr_t)output->data[1],
      (uintptr_t)output->data[2], output->width, output->height,
      output->linesize[0], output->linesize[1], output->linesize[2],
      output->colorspace};
  }
  return result;
}
int cm_encoder_write_host(void *pointer, const void *host) {
  int result = cm_encoder_prepare_host(pointer, host, NULL);
  return result < 0 ? result : cm_encoder_submit(pointer);
}
uint64_t cm_encoder_packets_written(void *pointer) {
  return ((Encoder*)pointer)->written_packets;
}
int cm_encoder_write(void *pointer, const uint8_t *pixels, size_t length) {
  Encoder *e=pointer;
  return cm_encoder_write_scaled(pointer,pixels,length,e->width,e->height);
}
int cm_encoder_finish(void *pointer) {
  Encoder *e = pointer;
  if (!e->opened)
    return 0;
  e->opened = 0;
  int result = avcodec_send_frame(e->codec, NULL);
  int status = 0;
  if (result < 0 && result != AVERROR_EOF)
    status = fail("flush encoder", result);
  else
    status = packets(e);
  // Always run the muxer's teardown, including when codec drain or packet
  // output failed. RTSP's trailer releases its private TCP/RTP contexts.
  char saved[sizeof(error_text)];
  memcpy(saved, error_text, sizeof(saved));
  result = av_write_trailer(e->output);
  if (status < 0) {
    memcpy(error_text, saved, sizeof(saved));
    return -1;
  }
  return result < 0 ? fail("finish output", result) : 0;
}

/* Preview scaling uses one owner-local context and never retains source pixels. */
void cm_preview_scale_free(void *context) {
  sws_freeContext(context);
}

int cm_preview_scale(void **context, const void *host, const uint8_t *pixels,
                     int source_width, int source_height, uint8_t *target,
                     size_t length, int width, int height) {
  struct SwsContext *scale = *context;
  if (host) {
    const HostFrame *frame = host;
    int result = frame_bgr(frame->frame, &scale, target, length, width, height);
    *context = scale;
    return result;
  }

  scale = sws_getCachedContext(scale, source_width, source_height,
      AV_PIX_FMT_BGR24, width, height, AV_PIX_FMT_BGR24, SWS_BILINEAR, NULL, NULL, NULL);
  *context = scale;
  if (!scale)
    return fail("Preview scaler", AVERROR(ENOMEM));

  const uint8_t *source[4] = {pixels, NULL, NULL, NULL};
  int strides[4] = {source_width * 3, 0, 0, 0};
  uint8_t *destination[4] = {target, NULL, NULL, NULL};
  int destination_strides[4] = {width * 3, 0, 0, 0};
  int result = sws_scale(scale, source, strides, 0, source_height,
                         destination, destination_strides);
  return result == height ? 0 : fail("Preview resize", AVERROR(EINVAL));
}
