"""Measure CPU resize/upload against an experimental CUDA bilinear resize."""
from __future__ import annotations

import argparse
import ctypes as c
import json
import os
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch

CUDA_SOURCE = r'''
extern "C" __global__ void resize_bgr(const unsigned char* input,
 unsigned char* output,
 int h,int w,int rh,int rw,int top,int left,int size) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=size*size*3)return;
 int ch=i%3,x=(i/3)%size-left,y=i/(size*3)-top;
 if(x<0 || x>=rw || y<0 || y>=rh){output[i]=114;return;}
 float sx=(x+0.5f)*w/rw-0.5f,sy=(y+0.5f)*h/rh-0.5f;
 int ax=(int)floorf(sx),ay=(int)floorf(sy);
 float fx=sx-ax,fy=sy-ay;
 int x0=max(0,min(w-1,ax)),x1=max(0,min(w-1,ax+1));
 int y0=max(0,min(h-1,ay)),y1=max(0,min(h-1,ay+1));
 float a=input[(y0*w+x0)*3+ch]*(1-fx)+input[(y0*w+x1)*3+ch]*fx;
 float b=input[(y1*w+x0)*3+ch]*(1-fx)+input[(y1*w+x1)*3+ch]*fx;
 output[i]=(unsigned char)__float2uint_rn(a*(1-fy)+b*fy);
}
'''


class ResizeKernel:
    def __init__(self):
        self.nvrtc = c.CDLL('libnvrtc.so.13')
        self.driver = c.CDLL('libcuda.so.1')
        program = c.c_void_p()
        create = self.nvrtc.nvrtcCreateProgram
        create.argtypes = [
            c.POINTER(
                c.c_void_p,
            ),
            c.c_char_p,
            c.c_char_p,
            c.c_int,
            c.c_void_p,
            c.c_void_p,
        ]
        assert create(
            c.byref(program),
            CUDA_SOURCE.encode(),
            b'resize.cu',
            0,
            None,
            None,
        ) == 0
        major, minor = torch.cuda.get_device_capability()
        options = (
            c.c_char_p * 2
        )(f'--gpu-architecture=sm_{major}{minor}'.encode(), b'--std=c++17')
        self.nvrtc.nvrtcCompileProgram.argtypes = [
            c.c_void_p, c.c_int, c.POINTER(c.c_char_p),
        ]
        try:
            assert self.nvrtc.nvrtcCompileProgram(program, 2, options) == 0
            length = c.c_size_t()
            self.nvrtc.nvrtcGetCUBINSize.argtypes = [
                c.c_void_p, c.POINTER(c.c_size_t),
            ]
            self.nvrtc.nvrtcGetCUBIN.argtypes = [c.c_void_p, c.c_void_p]
            assert self.nvrtc.nvrtcGetCUBINSize(program, c.byref(length)) == 0
            binary = c.create_string_buffer(length.value)
            assert self.nvrtc.nvrtcGetCUBIN(program, binary) == 0
        finally:
            self.nvrtc.nvrtcDestroyProgram.argtypes = [c.POINTER(c.c_void_p)]
            self.nvrtc.nvrtcDestroyProgram(c.byref(program))
        self.module, self.function = c.c_void_p(), c.c_void_p()
        self.driver.cuModuleLoadData.argtypes = [
            c.POINTER(c.c_void_p), c.c_void_p,
        ]
        self.driver.cuModuleGetFunction.argtypes = [
            c.POINTER(c.c_void_p), c.c_void_p, c.c_char_p,
        ]
        assert self.driver.cuModuleLoadData(c.byref(self.module), binary) == 0
        assert self.driver.cuModuleGetFunction(
            c.byref(self.function), self.module, b'resize_bgr',
        ) == 0
        self.driver.cuLaunchKernel.argtypes = [
            c.c_void_p, c.c_uint, c.c_uint, c.c_uint, c.c_uint, c.c_uint,
            c.c_uint, c.c_uint, c.c_void_p, c.POINTER(c.c_void_p), c.c_void_p,
        ]
        self.runtime = c.CDLL('libcudart.so.13')
        self.runtime.cudaMemcpy.argtypes = [
            c.c_void_p, c.c_void_p, c.c_size_t, c.c_int,
        ]

    def upload(self, target, source):
        assert self.runtime.cudaMemcpy(
            target.data_ptr(),
            source.ctypes.data,
            source.nbytes,
            1,
        ) == 0

    def resize(self, source, target, shape, resized):
        h, w = shape[:2]
        rh, rw = resized
        values = [c.c_void_p(source.data_ptr()), c.c_void_p(target.data_ptr())]
        values.extend(
            c.c_int(value) for value in (
                h,
                w,
                rh,
                rw,
                (640 - rh) // 2,
                (640 - rw) // 2,
                640,
            )
        )
        pointers = (c.c_void_p * len(values))(
            *
            (c.cast(c.byref(value), c.c_void_p) for value in values),
        )
        assert self.driver.cuLaunchKernel(
            self.function,
            (640 * 640 * 3 + 255) // 256,
            1,
            1,
            256,
            1,
            1,
            0,
            None,
            pointers,
            None,
        ) == 0

    def close(self):
        self.driver.cuModuleUnload.argtypes = [c.c_void_p]
        self.driver.cuModuleUnload(self.module)


def run_case(kernel, height, rounds):
    original = cv2.imread(
        str(Path('tests/cv_dataset/images').glob('*').__iter__().__next__()),
    )
    frame = cv2.resize(original, (height * 16 // 9, height))
    rh, rw = 360, 640
    host = np.full((640, 640, 3), 114, dtype=np.uint8)
    raw = torch.empty(frame.shape, dtype=torch.uint8, device='cuda')
    target = torch.empty(host.shape, dtype=torch.uint8, device='cuda')
    cv2.resize(frame, (rw, rh), dst=host[140:500])
    kernel.upload(raw, frame)
    kernel.resize(raw, target, frame.shape, (rh, rw))
    torch.cuda.synchronize()
    actual = target.cpu().numpy()
    differences = abs(actual.astype(np.int16) - host.astype(np.int16))
    times = {'cpu': [], 'gpu': []}
    for round_index in range(rounds):
        for mode in (
            ('cpu', 'gpu') if round_index %
            2 == 0 else (
                'gpu', 'cpu',
            )
        ):
            started = time.perf_counter_ns()
            for _ in range(50):
                if mode == 'cpu':
                    cv2.resize(frame, (rw, rh), dst=host[140:500])
                    kernel.upload(target, host)
                else:
                    kernel.upload(raw, frame)
                    kernel.resize(raw, target, frame.shape, (rh, rw))
                torch.cuda.synchronize()
            times[mode].append((time.perf_counter_ns() - started) / 50e6)
    return {
        'height': height, 'cpu_ms': statistics.median(times['cpu']),
        'gpu_ms': statistics.median(times['gpu']), 'rounds_ms': times,
        'max_pixel_error': int(differences.max()),
        'different_channel_values': int(np.count_nonzero(differences)),
        'enabled_in_production': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('native/inference/resize_candidate_results.json'),
    )
    args = parser.parse_args()
    os.sched_setaffinity(0, {4})
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.cuda.init()
    context_anchor = torch.empty(1, device='cuda')
    kernel = ResizeKernel()
    try:
        result = {
            'scope': (
                'Resize/pad + pageable H2D; output remains GPU uint8 BGR; '
                'excludes inference'
            ),
            'candidate': (
                'Experimental standalone CUDA bilinear kernel controlled '
                'from Python; not a Rust production path'
            ),
            'gate': (
                'Preserve CPU resize unless full detection parity and '
                'end-to-end gains are established'
            ),
            'cases': [
                run_case(
                    kernel,
                    height,
                    7,
                ) for height in (
                    720,
                    1080,
                )
            ],
        }
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
    finally:
        kernel.close()
        del context_anchor


if __name__ == '__main__':
    main()
