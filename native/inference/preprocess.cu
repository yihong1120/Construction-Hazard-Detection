// One launch converts BGR HWC bytes to RGB CHW and normalizes. Multiplication
// matches PyTorch's CUDA float32 division by 255 for every uint8 value.
extern "C" __global__ void normalize(const unsigned char* source, void* target,
                                     int pixels, int batch) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixels * batch) return;
    const int image = index / pixels;
    const int pixel = index % pixels;
    for (int channel = 0; channel < 3; ++channel) {
        const int offset = (image * 3 + channel) * pixels + pixel;
        const float value = float(source[index * 3 + 2 - channel]) * (1.0f / 255.0f);
        static_cast<float*>(target)[offset] = value;
    }
}

extern "C" __global__ void nv12_to_bgr(const unsigned char* y,const unsigned char* uv,
 unsigned char* bgr,const unsigned char* lut,int width,int height,int pitch){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=width*height)return;
 int x=i%width,row=i/width,yp=y[row*pitch+x],u=uv[(row/2)*pitch+(x/2)*2],v=uv[(row/2)*pitch+(x/2)*2+1];
 int index=((u*256+v)*256+yp)*3;
 for(int ch=0;ch<3;++ch)bgr[i*3+ch]=lut[index+ch];
}
// Calibrated unscaled YUV420P colour conversion fused with exact odd-integer
// sampling and normalization. No full-resolution RGB/BGR intermediate.
extern "C" __global__ void nv12_letterbox(const unsigned char* yp,const unsigned char* uv,
 const unsigned char* lut,float* output,int width,int height,int ypitch,int uvpitch,int tw,int th,
 int rw,int rh,int left,int top){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=tw*th)return;
 int x=i%tw-left,y=i/tw-top;float r=114,g=114,b=114;
 if(x>=0 && x<rw && y>=0 && y<rh){
  int scale=width/rw,sx=x*scale+scale/2,sy=y*scale+scale/2;
  int yy=yp[sy*ypitch+sx],u=uv[(sy/2)*uvpitch+(sx/2)*2],v=uv[(sy/2)*uvpitch+(sx/2)*2+1];
  int index=((u*256+v)*256+yy)*3;
  b=lut[index];g=lut[index+1];r=lut[index+2];
 }
 output[i]=r*(1.0f/255.0f);output[tw*th+i]=g*(1.0f/255.0f);output[2*tw*th+i]=b*(1.0f/255.0f);
}
extern "C" __global__ void gpu_letterbox(const unsigned char* bgr,float* output,
 int width,int height,int tw,int th,int rw,int rh,int left,int top){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=tw*th)return;
 int x=i%tw-left,y=i/tw-top;int scale=width/rw;
 for(int ch=0;ch<3;++ch){unsigned int value=114;
  if(x>=0 && x<rw && y>=0 && y<rh){
   int sx=x*scale+scale/2,sy=y*scale+scale/2;
   if(scale%2)value=bgr[(sy*width+sx)*3+2-ch];
   else value=(unsigned(bgr[((sy-1)*width+sx-1)*3+2-ch])+bgr[((sy-1)*width+sx)*3+2-ch]
                +bgr[(sy*width+sx-1)*3+2-ch]+bgr[(sy*width+sx)*3+2-ch]+2)/4;
  }
  output[ch*tw*th+i]=float(value)*(1.0f/255.0f);
 }
}
extern "C" __global__ void frame_blend(unsigned char* frame,const unsigned char* text,
 int w,int h,int ow,int oh,int x,int y){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ow*oh)return;
 int dx=i%ow+x,dy=i/ow+y;if(dx<0||dx>=w||dy<0||dy>=h)return;
 unsigned a=text[i*4+3];int dst=(dy*w+dx)*3;
 for(int ch=0;ch<3;++ch)frame[dst+ch]=(unsigned(text[i*4+ch])*a+unsigned(frame[dst+ch])*(255-a)+127)/255;
}
extern "C" __global__ void frame_to_nv12(const unsigned char* frame,unsigned char* y,
 unsigned char* uv,int w,int h,int pitch){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=w*h)return;
 int x=i%w,row=i/w;int b=frame[i*3],g=frame[i*3+1],r=frame[i*3+2];
 y[row*pitch+x]=max(0,min(255,((66*r+129*g+25*b+128)>>8)+16));
 if(x%2==0 && row%2==0){
  int sb=0,sg=0,sr=0;for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
   int j=((row+dy)*w+x+dx)*3;sb+=frame[j];sg+=frame[j+1];sr+=frame[j+2];
  }
  sb=(sb+2)/4;sg=(sg+2)/4;sr=(sr+2)/4;
  uv[(row/2)*pitch+x]=max(0,min(255,((-38*sr-74*sg+112*sb+128)>>8)+128));
  uv[(row/2)*pitch+x+1]=max(0,min(255,((112*sr-94*sg-18*sb+128)>>8)+128));
 }
}
