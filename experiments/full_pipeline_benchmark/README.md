🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# End-to-end camera pipeline benchmark

Run three camera pipelines sequentially on the same host. All three use the same YOLO26n INT8 engine, an input size of 640, a maximum batch size of 8, a 2 ms batch wait, RTSP sources and hazard rules.

| Variant | Pipeline |
| --- | --- |
| `python` | `main.py` with a Python TensorRT worker and Python postprocessing |
| `hybrid` | `main.py` with a Rust TensorRT worker and Rust postprocessing |
| `rust` | Standalone `construction-pipeline`, managing capture, inference, analysis, output and the outbox in one process |

For both Python variants, `legacy.py` adds hazard-decision tracing and limits OpenCV and the decoder to one thread each. Rust uses `record_detections` to trace the equivalent point. Test outputs use libx264, the ultrafast preset, CRF 23, one thread and a two-second GOP, at 15 FPS by default. Python shares captured frames between clean and annotated outputs. A wrapper supplies the output options to the FFmpeg subprocesses. The standalone Rust pipeline calls libav in-process.

## Build and dependencies

Run the commands from the repository root. The host needs an NVIDIA GPU, TensorRT and CUDA versions that load the test engine, Git, Docker, FFmpeg and the project's Python 3.15. The FFmpeg wrapper uses `/usr/bin/ffmpeg`. Install the project and measurement tools, then prepare the TensorRT bindings in the same Python environment.

```bash
uv sync --locked --extra streaming --extra yolo --extra yolo-gpu
source .venv/bin/activate
uv pip install psutil nvidia-ml-py matplotlib
```

- Follow the [Python setup guide](../../README.md#1-prepare-python) for the TensorRT Python bindings and SDK requirements. The versions used by the Python bindings and Rust binary must both load the same engine.
- For `hybrid`, build and install the [Rust inference wheel](../../native/inference/README.md#tensorrt-versions-and-build) and [Rust postprocessing wheel](../../bindings/python/postprocess/README.md#build-and-backend-selection) into that environment. The inference guide also covers `LD_LIBRARY_PATH` and `YOLO_WORKER_NATIVE_LIBRARY_DIRS`.
- For the standalone binary, enable the `tensorrt` feature using the [TensorRT build instructions](../../native/pipeline/README.md#enable-tensorrt-inference).
- Supply a matching YOLO26n checkpoint at `models/pt/best_yolo26n.pt` and an exported INT8 `.engine`. The test checks both SHA-256 hashes. The default engine path is `models/int8_engine/best_yolo26n.engine`. Use `--engine` to supply another path. Both model files must be provided separately.

## Start isolated services

Bind PostgreSQL, Redis, MediaMTX and the HTTP mock to loopback. All credentials, tokens and uploads are test fixtures. Use this separate set of containers and database for the benchmark.

```bash
docker run -d --rm --name chd-e2e-pg \
  -e POSTGRES_USER=fixture -e POSTGRES_PASSWORD=fixture \
  -e POSTGRES_DB=chd_e2e_benchmark -p 127.0.0.1:25433:5432 postgres:17.11-bookworm
docker run -d --rm --name chd-e2e-redis \
  -p 127.0.0.1:26380:6379 redis:7.4.10-alpine3.21
docker run -d --rm --name chd-e2e-rtsp \
  -v "$PWD/experiments/full_pipeline_benchmark/mediamtx.yml:/mediamtx.yml:ro" \
  -p 127.0.0.1:28554:8554 -p 127.0.0.1:29997:9997 \
  bluenviron/mediamtx@sha256:f8628851106cc053f9175b248050bb5f362a6e65abd72297c167a1cb5a9a3be2
```

Check that PostgreSQL accepts TCP connections, Redis returns `PONG` and the MediaMTX API lists its paths. Leave the HTTP mock running.

```bash
docker exec chd-e2e-pg pg_isready -h 127.0.0.1 -U fixture -d chd_e2e_benchmark
docker exec chd-e2e-redis redis-cli ping
curl --fail http://127.0.0.1:29997/v3/paths/list

mkdir -p outputs/full_pipeline_fixture
python -m experiments.full_pipeline_benchmark.mock \
  --journal outputs/full_pipeline_fixture/delivery.jsonl
```

## Prepare and publish the RTSP source

In a second terminal, turn `tests/videos/test.mp4` into a 30-second loop. Upscaling the 640×352 source measures decoding, frame transfers and encoding at 1920×1080. When changing the resolution or frame rate, update both the video command and the experiment settings below.

```bash
ffmpeg -nostdin -n -filter_threads 1 -i tests/videos/test.mp4 \
  -vf scale=1920:1080:flags=bilinear,fps=15 -t 30 -an \
  -c:v libx264 -preset ultrafast -crf 18 -threads 1 \
  -g 15 -keyint_min 15 -sc_threshold 0 -bf 0 -pix_fmt yuv420p \
  outputs/full_pipeline_fixture/source.mp4
ffmpeg -nostdin -re -stream_loop -1 \
  -i outputs/full_pipeline_fixture/source.mp4 \
  -an -c:v copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:28554/source
```

## Measure and generate the report

Activate the same Python environment in a third terminal. Use a new `--output` directory for each run and let the variants execute sequentially. The example measures 32 camera inputs and demand from four viewers, each requesting a clean and an annotated output.

```bash
export PYTHON_DOTENV_DISABLED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m experiments.full_pipeline_benchmark.benchmark \
  --backends python hybrid rust --cameras 32 --viewers 4 \
  --width 1920 --height 1080 --fps 15 --language en \
  --rounds 3 --warmup 30 --seconds 30 \
  --output outputs/full_pipeline_32

python -m experiments.full_pipeline_benchmark.report \
  --primary outputs/full_pipeline_32
```

Each round waits for the first detection trace on every camera, then checks readiness and dimensions of the requested outputs. The first variant rotates each round. All variants warm up and run for the specified intervals. Use `--cameras 1 --viewers 1` for a single-camera comparison.

Each input directory for `report` must contain results for one camera count, with a complete `results.json` covering all three variants in every round. Supply other completed experiments with `--single` or `--low-resolution`. By default `summary.json`, `comparison.png` and `comparison.svg` are written under `report/` in the primary experiment directory. Use `--output` to change the destination.

- `settings.json` records engine and executable hashes. Raw results include each round's `result.json`, `samples.json`, logs and frame traces. The report gives the median, minimum and maximum observed for each variant.
- For CPU percentages, 100% is one logical core. PSS apportions shared memory between processes. Process-tree totals cover the parent, workers and FFmpeg subprocesses. Available service CPU metrics are recorded separately. Host-wide GPU utilisation may include other workloads.
- `frame_age_p95_ms` measures time from retrieving the decoded frame to completing the hazard decision. Interpret it alongside completed decisions, source backlog, dropped frames and output readiness. Measure viewer end-to-end latency separately.

Stop the source and mock in their terminals, then stop the test containers.

```bash
docker stop chd-e2e-pg chd-e2e-redis chd-e2e-rtsp
```
