🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# Python and Rust inference with the same INT8 TensorRT engine

This experiment compares two inference paths using local video, test images and a TensorRT engine. It writes the results to `results.json`.

## Implementations

- Python uses Ultralytics 8.4.183 as pinned in `uv.lock` for this run, with `rect=True` and `quantize=8`. This reproduces the rectangular-input comparison. The production INT8 worker uses `rect=False`. [Production Rust inference](../../native/inference/README.md)
- Rust handles batching, letterbox dimensions, detection filtering before hazard analysis and coordinate restoration. A small C++ ABI calls OpenCV and TensorRT. A fused CUDA kernel performs BGR-to-RGB conversion, HWC-to-CHW conversion and normalisation. NVRTC compiles it during initialisation.
- Both use **the same existing INT8 `.engine` and the same TensorRT 11.1 shared library** and call synchronous `executeV2`. Rust inference runs without Python or PyTorch.
- This experiment loads the engine directly. The `ultralytics-inference` crate, which primarily loads ONNX files, is outside its scope.

Timing differences include the fused kernel, simplified result construction and the Rust/C++ calling convention. A 256-entry normalisation lookup table generated with PyTorch CUDA gives exactly the same floating-point result for each uint8 value as the reference implementation.

## Inputs, correctness and timing

The input consists of 48 frames sampled every 18 frames from `tests/videos/test.mp4`, plus 10 local test images. Files are decoded to BGR bytes before timing. The test keeps Ultralytics `rect=True`, stride 32 and `imgsz=640`. This permits rectangular tensors such as 352×640. Batches with mixed shapes use 640×640. For batch size 8, the final batch contains two images.

YOLO26n and YOLO26s each run at batch sizes 1 and 8. Each configuration checks all 58 frames before timing.

1. Compare normalised input tensors bit for bit and stop on any difference.
2. Compare all raw network outputs and record the maximum error.
3. Require the same detection count, classes and order. Stop if confidence differs by more than 1e-5 or original-image coordinates by more than 0.002 pixels.

The benchmark times two sections separately.

| Name | Scope |
| --- | --- |
| `infer` | Calls TensorRT with the same batch already on the GPU and keeps the output there. Includes host call and synchronisation time. |
| `detect` | Starts with in-memory BGR frames and includes letterboxing, transfer, normalisation, inference and confidence filtering. Returns CPU detection rows in original-image coordinates. |

After warm-up, both implementations run seven alternating rounds pinned to the same Intel P-core. OpenCV and PyTorch use one CPU compute thread. The driver and TensorRT still have helper threads. Timings include per-frame result allocation in Rust. JSON commands, reading/writing results, loading the model and initialisation are outside the timed interval. `mean_ms_per_image` divides elapsed time by the actual number of frames, counting the final two images as two. `p95_ms_per_batch` reports batch latency. Speed-up is the median of paired round ratios and the range gives observed extremes. These measurements include more than GPU kernel time.

`detect` excludes camera capture, decoding, shared-memory/queue IPC, batch waits, tracking, hazard rules, drawing, encoding, storage and notifications. It cannot establish how many cameras a complete server can handle. Existing GPU workloads continue to run. Each round records `nvidia-smi` before and after, so small percentage differences may reflect external load or clock changes.

## Reproducing the experiment

Use Python 3.15 with the project's inference dependencies, Rust 1.93.1, a C++17 compiler, matching TensorRT 11.1 headers, runtime and Python bindings, OpenCV core and imgproc, and the CUDA runtime, driver and NVRTC. The tested OpenCV versions were C++ 4.10.0 and Python 5.0.0. Preprocessing must match bit for bit. The CUDA kernel was built for `sm_89` and tested on an RTX 4090. Adjust the architecture in `bridge.cpp` before using another GPU.

Prepare an isolated environment with a TensorRT version and shared libraries capable of loading the same engine. Point `PYTHONPATH` and `LD_LIBRARY_PATH` to matching Python bindings and runtime libraries.

Adjust the paths for your installation. `CUDA_INCLUDE` must contain the full toolkit headers. `CUDA_LIB` must provide linkable `libcudart.so` and `libnvrtc.so`.

```bash
export TRT_INCLUDE=/path/to/TensorRT-11.1/include
export TRT_LIB=/path/to/TensorRT-11.1/lib
export CUDA_INCLUDE=/path/to/cuda/include
export CUDA_LIB=/path/to/cuda/lib
export OPENCV_ROOT=/path/to/opencv-prefix  # Use / for a system installation
export PYTHONPATH=/path/to/isolated-python-packages
export LD_LIBRARY_PATH="$TRT_LIB:$CUDA_LIB:$OPENCV_ROOT/usr/lib/x86_64-linux-gnu"
export PYTHON_DOTENV_DISABLED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

cargo build --release --locked \
  --manifest-path experiments/rust_trt_benchmark/Cargo.toml
python experiments/rust_trt_benchmark/benchmark.py --prepare
python experiments/rust_trt_benchmark/benchmark.py \
  --models yolo26n yolo26s --batches 1 8 --rounds 7 --core 4
```

Git ignores `data/`. It stores the input manifest, BGR frames, normalisation LUT, raw per-batch timings, worker logs and tensors/network outputs from the latest verification. `results.json` records timing rounds for every configuration and SHA-256 hashes of the engine, runtime, source and input. `VmHWM` is the peak CPU RSS of the entire test process, including the framework, inputs and buffers. Both processes preload the same 58 frames. This figure does not measure per-camera memory on the complete server.

See the [NVIDIA TensorRT runtime API](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/c-api-docs.html) and [Ultralytics Rust inference](https://docs.ultralytics.com/inference/).
