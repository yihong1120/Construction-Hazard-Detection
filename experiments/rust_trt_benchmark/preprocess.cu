// Same BGR -> RGB, HWC -> CHW and float32 normalization as the Python worker.
// LUT values are prepared with the reference PyTorch CUDA divide operation.
extern "C" __global__ void normalize(const unsigned char* source, float* target, const float* lut,
                          int pixels, int batch) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixels * batch) return;
    const int image = index / pixels;
    const int pixel = index % pixels;
    for (int channel = 0; channel < 3; ++channel)
        target[(image * 3 + channel) * pixels + pixel] =
            lut[source[index * 3 + 2 - channel]];
}
