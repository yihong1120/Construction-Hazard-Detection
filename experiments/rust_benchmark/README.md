🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# Python and Rust benchmark

This reproducible experiment uses the project's default centroid tracker in `YoloDetector` and hazard rules in `DangerDetector` as its baseline. The Rust version preserves matching order, thresholds, track IDs, driver exclusions, spatial indexing and alert output. Each run records per-frame results, round summaries and input hashes.

## Scope

- The real dataset contains 48 frames sampled every 18 frames from `tests/videos/test.mp4` and 10 images from `tests/cv_dataset/images`. It uses YOLO26n, 640×640, FP32, batch 1 and a confidence threshold of 0.25. A separate inference comparison uses 16 preprocessed tensors.
- The rules cover missing helmets, missing safety vests and people near machinery or vehicles, including the driver exclusion.
- The timings exclude cone and utility-pole polygons, the Hungarian tracker, camera connections, decoding and preprocessing, overlay drawing, encoding, MediaMTX, Redis/PostgreSQL, the outbox and notifications.
- Seeded synthetic cases with 16, 64 and 256 objects test how processing scales under load. They do not estimate the load of a live camera.
- Both inference implementations use the same ONNX model, ONNX Runtime version and shared library file. CPU and CUDA may use separate Python environments with their respective ORT runtimes.

## Timing

Build Rust with `cargo build --release --locked`. Each configuration runs seven rounds, alternating which implementation goes first, pinned to the same Intel P-core with one ORT compute thread. Timings include per-frame result allocation, alert construction and deallocation. Reading input, loading the model, warming up, reading/writing JSON and rebuilding the tracker happen outside the timed interval.

Each round's millisecond value in `results.json` is the elapsed processing time divided by the number of frames. `median_paired_speedup` is the median of the within-round Python/Rust ratios. `paired_speedup_range` gives the minimum and maximum of the seven ratios. These endpoints are **not a statistical confidence interval**. P95 is the median across rounds of the 95th percentile of per-frame times.

Existing GPU workloads continue running. Each CUDA measurement records `nvidia-smi` before and after, so differences of a few per cent may reflect other workloads or clock changes. `peak_rss_kib` includes initialisation, input loading and dependencies. It measures the benchmark process rather than a complete running server.

## Reproducing the experiment

Use Linux, Rust ≥1.92, ONNX Runtime 1.31.0 CPU and CUDA runtimes, and Python environments with NumPy, OpenCV, scikit-learn, Shapely and ORT bindings. `--core` selects a logical CPU available to the test process. The CPU benchmark uses the current Python interpreter. Set `CHD_BENCH_GPU_PYTHON` if CUDA bindings are installed in a different environment, or `CHD_BENCH_GPU_SITE` for bindings supplied in an isolated package directory.

```bash
cargo build --release --locked --manifest-path experiments/rust_benchmark/Cargo.toml

export CHD_BENCH_CPU_ORT=/path/to/cpu/libonnxruntime.so.1.31.0
export CHD_BENCH_GPU_ORT=/path/to/cuda/libonnxruntime.so.1.31.0
export CHD_BENCH_GPU_PYTHON=/path/to/gpu-python/bin/python
export LD_LIBRARY_PATH="/path/to/cuda/lib:/path/to/cudnn/lib:${LD_LIBRARY_PATH:-}"

# Use Python 3.15 with NumPy, OpenCV, scikit-learn, Shapely and ORT.
PYTHON_DOTENV_DISABLED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python experiments/rust_benchmark/benchmark.py prepare

PYTHON_DOTENV_DISABLED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python experiments/rust_benchmark/run_experiment.py --core 4 --rounds 7
```

`prepare` uses local test video and images without reading cameras from `.env` or connecting to the production database. Supply `models/pt/best_yolo26n.onnx` separately. The manifest records the model and source SHA-256 hashes. By default, the benchmark compares every frame's tracking and alerts and the complete raw CPU and CUDA inference outputs for each implementation. A mismatch stops timing.

Raw per-frame timings, output, environment information and GPU state are stored in `data/`. The summary is written to `results.json`. Use `--verify-only` for correctness checks alone. The benchmark runs the original Python class definition to avoid parsing the module's top-level PEP 810 lazy import with Python 3.14. It uses the original tracker implementation.
