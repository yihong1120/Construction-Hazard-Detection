🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# Standalone Rust camera pipeline

`construction-pipeline` captures local video and RTSP or HTTP(S) streams. It runs TensorRT inference, tracks objects, analyses hazards, renders overlays and publishes the output. A PostgreSQL outbox stores evidence for incidents, and delivery uses the existing HTTP endpoints for uploads and notifications. The Rust process uses native interfaces to FFmpeg, TensorRT, CUDA and OpenCV. It runs without Python, PyTorch or Ultralytics. The management API and model training can run independently.

```mermaid
flowchart LR
    C["JSON / PostgreSQL configuration"] --> S["Rust supervisor and generation tracking"]
    S --> V["In-process libav decoding<br/>One owner per source"]
    V --> B["Bounded frame pool<br/>Retained YUV or pooled BGR"]
    B --> L["Latest-frame mailbox"]
    L --> I["Shared worker per model<br/>Select the latest frame after GPU admission"]
    I --> P["Shared preprocessing pool<br/>Fused YUV or OpenCV INTER_LINEAR"]
    P --> G["TensorRT / CUDA<br/>Synchronous or two-slot asynchronous inference"]
    G --> T["Bounded CPU executor<br/>Tracking and hazard analysis"]
    T --> M["Metadata snapshot<br/>Generation and freshness"]
    L --> R["Latest frame<br/>Multilingual Rust overlays"]
    M --> R
    R --> E["In-process libav encoding and muxing<br/>NVENC or libx264"]
    L -->|"Clean output with direct YUV scaling"| E
    E --> O["RTSP / MP4"]
    T --> D["Redis alerts and model status"]
    T --> J["Rust JPEG evidence from the source frame"]
    J --> Q["Persistent PostgreSQL outbox"]
    Q --> U["Bounded evidence upload workers"]
    U -->|"Save upload receipt"| Q
    Q -->|"Events with receipts"| N["Independent notification workers and retries"]
```

## Build from source

Run the commands from the repository root. The recommended environment is Linux with Rust 1.93.1, a C compiler and the FFmpeg development libraries. The example uses Ubuntu packages and rustup.

```bash
sudo apt-get install build-essential pkg-config ffmpeg \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
  fonts-noto-core fonts-noto-cjk
rustup toolchain install 1.93.1 --profile minimal --component rustfmt
cargo +1.93.1 build --release --locked --manifest-path native/pipeline/Cargo.toml
```

Use the same major version of FFmpeg for the development and runtime libraries. Set `FFMPEG_ROOT` to the installation root when using an SDK outside the system directories. Noto fonts support multilingual overlays.

### Validate hazard rules on the CPU

This example analyses hazards from a JSONL file and writes tracking data, alerts and zone polygons to another JSONL file. It reads and writes local files only.

```bash
work_dir="$(mktemp -d)"
printf '%s\n' '{"tracked":[[0,0,10,20,0.9,2,0,0]],"reset":true,"rules":{"detect_no_safety_vest_or_helmet":true}}' \
  > "$work_dir/input.jsonl"
native/pipeline/target/release/construction-pipeline analyze \
  --input "$work_dir/input.jsonl" --output "$work_dir/result.jsonl"
cat "$work_dir/result.jsonl"
```

The CPU build can also replay recorded detections as input for testing the standalone pipeline. Set `experimental_replay` to enable replay. Delivery endpoints must be on loopback if delivery is enabled, and Redis requires an isolated key prefix.

### Enable TensorRT inference

Build with the `tensorrt` feature to run inference. Install compatible TensorRT, CUDA and OpenCV core and imgproc SDKs and runtime libraries. Adjust the paths to suit your installation.

```bash
export TRT_INCLUDE=/opt/TensorRT/include
export TRT_LIB=/opt/TensorRT/lib
export CUDA_INCLUDE=/usr/local/cuda/include
export CUDA_LIB=/usr/local/cuda/lib64
export OPENCV_ROOT=/
export LD_LIBRARY_PATH="$TRT_LIB:$CUDA_LIB:${LD_LIBRARY_PATH:-}"
cargo +1.93.1 build --release --locked --features tensorrt \
  --manifest-path native/pipeline/Cargo.toml
```

Set `YOLO_TRT_ENGINE` to an `.engine` file that your TensorRT runtime and GPU can load. `init` detects the source dimensions, calculates the engine's SHA-256 and writes a configuration for a limited number of frames with local file output. `run` and `outbox` require a reachable PostgreSQL database. See the database guide before setting `DATABASE_URL`. [Database connections and migration (Traditional Chinese)](POSTGRES_OUTBOX.md)

```bash
run_dir="$(mktemp -d)"
pipeline=native/pipeline/target/release/construction-pipeline
"$pipeline" init --config "$run_dir/config.json" --engine "$YOLO_TRT_ENGINE" \
  --source tests/videos/test.mp4 --state-dir "$run_dir/state" \
  --version local-experiment --max-frames 120
"$pipeline" validate --config "$run_dir/config.json"
"$pipeline" run --config "$run_dir/config.json"
"$pipeline" outbox --config "$run_dir/config.json"
```

The example produces an unannotated MP4, annotated MP4s in English and Traditional Chinese, and `summary.json`. Delivery is disabled by default. Events during working hours can still be written to the outbox. `validate` checks the JSON structure, and the engine's SHA-256 is verified when the model loads. Use a fresh configuration filename because `init` protects existing files.

## CPU worker configuration

The numeric `cpu_workers` setting uses fixed mode by default. When omitted, it is based on the available CPU parallelism, up to eight workers. With `cpu_workers_auto` configured, `cpu_workers` becomes the upper limit. The example allocates one shared worker slot for every two cameras, up to 16.

```json
{
  "cpu_workers": 16,
  "cpu_workers_auto": {
    "cameras_per_worker": 2,
    "min_workers": 1,
    "reserve_cpus": 4
  }
}
```

Rust counts loaded, enabled cameras whose configuration has not expired, divides the count by `cameras_per_worker` and rounds it up before applying the bounds. The effective upper limit is the lower of `cpu_workers` and the available CPU parallelism minus `reserve_cpus`, with at least one slot. Enabled RTSP cameras still count while reconnecting. Omit `cpu_workers_auto` to use fixed mode.

The mode and upper limit are read at start-up, and idle threads in the pool sleep. When cameras start, stop or expire, or the PostgreSQL catalogue is refreshed, the same pool adjusts its active and queued capacity. Running and queued jobs finish when capacity is reduced. The `cpu_workers` entry in `summary.json` records the initial capacity, current capacity, upper limit and number of adjustments. Logs record the start-up configuration and subsequent changes to capacity.

The shared pool handles tracking, zone analysis, overlays and evidence JPEGs. The total number of queued and running jobs is capped at twice its concurrency limit. Stream decoding, encoding and model preprocessing have separate resource settings. When benchmarking, compare `render_skipped`, `encoded_unique` and `encoded_repeated` alongside input resolution, frame rate, viewing demand and CPU usage.

## Scheduling and reliability

- All cameras using a model share one Rust inference thread and TensorRT context. The maximum batch wait is 2 ms, and the default request queue holds 32 items per model. The worker keeps batches within the engine profile.
- Live annotated previews with boxes or alerts are drawn at the output resolution. Language variants at the same resolution share a scaled base image. Frames with nothing to draw pass directly to the encoder. The live encoder reuses converted pixels when the same frame is repeated, maintaining the output frame rate with new timestamps.
- Overlays, alert summaries and stream status support `zh-TW`, `zh-CN`, `en`, `ja`, `vi`, `id`, `fr` and `th`. Built-in class names use the playback language. Set `Model.labels` to supply names for custom classes in each language. On-demand playback maps aliases such as `zh-Hans`, `fr-CA` and `th_TH` to the corresponding stream language and starts the requested variant only. CJK and Thai text can use system Noto fonts or a font specified in `fonts`.
- Live HTTP(S) sources reconnect independently on a read timeout or EOF. HTTP(S) files follow media timestamps, and frame-by-frame offline processing ends at EOF.
- `lossless: false` selects live mode, which takes the latest frame and skips frames to limit latency. `lossless: true` uses a bounded queue for frame-by-frame offline validation. `fps` sets the output clock. There is currently no separate inference FPS budget per camera. `summary.json` records captured, processed, skipped, stale and encoded frames, as well as failures.
- An `Arc` holds each frame until its last reader releases it. The buffer then returns to the pool. Reconnecting the capture source advances its generation, discards old inference results and resets tracking.
- RTSP uses TCP and reconnects with backoff after a read timeout or stall. Advancing presentation timestamps (PTS) allow valid static images. With no PTS, a low-cost image difference check detects freezes. Live output shows an offline screen after more than five seconds without a new frame.
- Processes can coordinate GPU inference, model memory and NVENC budgets with `flock` by sharing the same lease directory, namespace and capacity. The memory budget represents a configured reservation. Actual VRAM use requires separate measurement.
- Decoding and encoding each have a dedicated Rust owner thread. Long-running I/O stays outside Tokio's blocking pool. The process calls libav directly for video, with no FFmpeg subprocess or rawvideo pipe. The NVENC lease is held until the native encoder releases it. Failed RTSP output can reconnect and fall back to CPU encoding. Failed file output reports an error immediately to protect existing recordings from being overwritten when reopening. Raw libav error logs omit sources, output destinations and credentials.
- The JSON configuration can connect to a PostgreSQL catalogue and check the version and SHA-256 of published weights against the engine manifest. LISTEN and polling refresh the configuration. Existing jobs stop before a model switch. If a refresh fails, the pipeline keeps the last verified configuration.
- Each stream uses `work_start_hour` and `work_end_hour` in the frame's local time. The start hour is inclusive and the end hour is exclusive. A range of `0` to `24` covers the whole day. The JSON setting `streams[].timezone` defaults to `Asia/Taipei`, which also applies to cameras from the PostgreSQL catalogue. Outside working hours, evidence and notifications are generated only when a person enters a controlled area around traffic cones or utility poles. Live detection and overlays continue throughout the day.
- Redis alerts use the existing warning stream format. Rust also publishes the latest metadata. The site generation and camera owner token block stale writes, and model status can respond to a publication challenge.
- The source JPEG and event are committed in the same PostgreSQL transaction before the HTTP dispatcher sends them. The upload receipt is saved before notification, so a notification retry or process restart can reuse the completed upload.
- The default outbox limits are 512 MiB of retained JPEGs and metadata and 20 attempts per delivery stage. Permanent errors go to the failed queue. The limit covers payloads. PostgreSQL tables, indexes and WAL require additional space. The endpoint needs to support `Idempotency-Key` to prevent a duplicate if a request succeeds remotely before the local result is recorded.
- `record_detections` is off by default. When enabled, each camera's diagnostic JSONL file rotates at 10 MiB and keeps the current and previous files. SIGINT and SIGTERM stop capture, join native media owner threads, finish encoder trailers, close models and write the summary.

`delivery` uses the existing `/upload` multipart, notification JSON and OAuth client credentials protocols. Credentials are read only from the environment variables referenced by the configuration.

`delivery.evidence_workers` and `delivery.notification_workers` each default to two concurrent requests. A slow push delivery leaves both evidence upload slots available. Notification retries use the stored receipt and metadata without loading the JPEG from PostgreSQL. When the endpoint supplies a retry delay, the dispatcher waits for at least that delay, bounded to one day.

Set `experimental_replay` explicitly to use recorded detections. Replay delivery is limited to loopback, and Redis requires an isolated prefix to keep test data out of live services.

Use `outbox --config ... --requeue-failed` to requeue failed events.

## Capture and processing clocks

- `capture.buffer_frames` is the hard limit on BGR buffers per source, with a live default of six. When the pool is full, live capture keeps decoding and draining the source while skipping BGR conversion. Offline lossless mode waits for space and preserves every frame. Each buffer has 64 bytes of padding for codec SIMD operations and is reclaimed when its last `Arc` is dropped.
- The inference request queues the camera mailbox. The worker takes its latest frame after GPU admission and passes the actual inference frame with the result to tracking. Each camera can have one outstanding request.
- Live overlays pair the latest frame with the latest analysis. Retained YUV frames trigger annotation preparation on arrival, and each encoder paces its output at its configured FPS. The metadata holds no full-size images. If the analysis exceeds `overlay_max_age_ms` (default 1000) or belongs to a different generation, the renderer displays `Analysis pending`. Lossless mode matches frames and analysis frame by frame.
- `capture.max_lag_ms` defaults to 2000. Capture reconnects if the lag relative to the first decoded PTS grows beyond the threshold, which prevents a growing backlog of old frames. A significant PTS jump backwards also advances the generation. `read_timeout_ms` bounds interruptible native I/O.
- `summary.json` includes `converted`, `capture_dropped`, `capture_reconnects`, `capture_pool_peak`, `rendered`, `render_skipped` and `render_pending`. The trace includes the actual `source_pts`, `source_lag_ms` and `batch_size` values. Use these fields to distinguish inference frame skipping, conversion admission and overlay load shedding.

`native/media` provides a small C ABI adaptor. Rust owns lifecycle, cancellation, scheduling and backpressure. The default CPU decode and BGR preprocessing path still moves image data. An optional NVDEC and GPU NV12 path is described below. Files with rotation or mirroring display matrices are rejected to preserve consistent detection coordinates. Standard unrotated RTSP streams and video files are supported. [FFmpeg decoding send/receive API](https://ffmpeg.org/doxygen/8.0/demux_decode_8c-example.html), [FFmpeg encoding and muxing example](https://ffmpeg.org/doxygen/8.0/mux_8c-example.html)

## Frame admission and preprocessing optimisations

All models in the standalone Rust pipeline share the preprocessing pool. `preprocess_workers` defaults to one and sets the number of model preprocessing threads. This budget is separate from `cpu_workers`, which covers tracking, geometry, overlays and JPEGs. Overall CPU usage includes both pools and the other pipeline stages. Batches of multiple images can preprocess images in parallel. A single-image batch can still use the exact CPU sampling path below.

```json
{
  "preprocess_workers": 8,
  "exact_cpu_resize": true,
  "gpu": {
    "asynchronous": false,
    "pipeline_depth": 1
  }
}
```

Merge the fragment into an existing configuration. Batch size and input dimensions must fit the engine profile. A larger preprocessing pool can help batches of multiple images. Monitor overall CPU usage when adjusting the pool.

`exact_cpu_resize` defaults to `true`. Same-size images and images downscaled proportionally by an odd integer factor can be sampled directly in Rust, with byte-for-byte matching letterbox padding and BGR values against OpenCV INTER_LINEAR. Images at other scales, upscaled images and individual incompatible images within mixed batches use the OpenCV path. Set the option to `false` to use OpenCV for every image. The setting applies to the standalone Rust pipeline. The Python bindings use their existing preprocessing path.

`gpu.exact_resize` is experimental, defaults to `false` and requires `gpu.asynchronous: true`. It uploads the full CPU BGR frame and fuses resizing with RGB CHW normalisation on CUDA. The whole batch must meet the exact scaling conditions above to use this path. Other batches use CPU preprocessing. The path uses pinned and GPU buffers and still transfers the full frame.

`gpu.asynchronous: true` enables pinned buffers and CUDA streams. With `pipeline_depth: 2`, CPU preprocessing for the next batch overlaps the current GPU work. In-flight batches are drained before changing the shape of a TensorRT context and when cancelling or shutting down. An empty queue proceeds without waiting for a second batch. Use depth one for global `inference_slots` admission. Depth two currently requires global inference slots to be disabled.

With `record_detections` enabled, each trace record includes `inference_timing` fields `batch_id`, `queue_ms`, `admission_ms`, `prepare_ms`, `submit_ms`, `gpu_wait_decode_ms`, `batch_span_ms` and `in_flight`, plus `analysis_queue_ms` and `analysis_cpu_ms`.

The trace also records `gpu_preprocess` and `cpu_exact_images`. The latter counts the images in the batch that used exact Rust sampling. The same batch timings appear on multiple frames, so deduplicate by model and batch ID when aggregating. Batch IDs reset when the model worker is rebuilt.

These figures measure host elapsed time. Use GPU events to measure kernel timing separately. In synchronous mode, `submit_ms` includes execution and waiting. In asynchronous mode it measures submission. `gpu_wait_decode_ms` includes collection waits and result decoding.

`summary.json` includes `capture_read_ms` (including source waits), `capture_convert_ms`, cumulative analysis queue and work times, and `encode_ms`. Output counters also include `encoded_unique`, `encoded_repeated` and `encoded_offline`. Unique frames are counted by source generation and sequence, independently of changes in image content. These totals include warm-up and start-up, so use the change in each total to calculate rates for a specific measurement window.

`capture.max_convert_fps` defaults to `0` (unlimited). Set it to `15` if the application needs only 15 FPS per camera. Decoding and PTS or connection liveness checks continue. Frames over budget skip BGR conversion before they would be overwritten in the latest-frame mailbox.

The cap applies to conversion per stream and also limits the rate at which clean and annotated output receive fresh frames. The output `fps` setting still controls its own clock, so output above the conversion cap may repeat frames. Keep the cap at `0` or increase it for higher refresh rates or inference on every frame.

`lossless: true` preserves every frame regardless of this setting. In `summary.json`, `capture_rate_dropped` counts frames skipped by conversion admission and `capture_dropped` counts frames dropped when buffers run out. Rate limiting uses a fixed phase on a monotonic clock with tolerance for early frames. Reconnection resumes with current slots.

## CPU YUV and on-demand output

With `capture.host_yuv: true`, CPU decoding keeps frames in reference-counted YUV storage. The encoder scales clean and annotated outputs into its own limited-range YUV420P planes. For annotated outputs the renderer sends geometry, masks and glyphs to the encoder, which paints them after scaling using the output colour matrix and 4:2:0 chroma coverage. This applies to live previews and annotated files. Evidence JPEGs encode the retained frame and source freeze checks sample the decoded planes. In `summary.json`, `encoder_host_frames` counts submissions from retained frames and `encoder_annotated_host_frames` counts the subset with YUV annotation, excluding repeats.

For matching native dimensions and even frame sizes, exact preprocessing supports 8-bit planar 4:2:0, 4:2:2 and 4:4:4. When available, calibrated lookup tables sample same-size images and odd-integer reductions directly. Other supported sizes use reusable scanline buffers to produce the same letterbox pixels as OpenCV, including a 2× reduction from 720p to 640 × 360. Unsupported layouts use the full-size BGR reference path.

Server-side playback supports two optional `output` settings, both disabled by default.

- `clean_passthrough` forwards compressed packets from the existing input connection. It is limited to live RTSP with libav ingest, and switches to encoding if the source dimensions or FPS differ from the configured output.
- `on_demand` starts and stops live output based on existing Redis detail and language viewing leases. One poller per pipeline process reads the registered cameras' leases together every 250 milliseconds. Server-side annotation sleeps when all its outputs are inactive. Inference and event processing continue independently of viewing demand.

Standalone JSON fixtures can use `output.client_overlay` to exercise the metadata renderer in the browser benchmark.

The CPU path falls back per image. In a mixed batch, BGR is created only for frames that cannot use fused preprocessing. After verification over all 256³ colour combinations, the lookup table can be reduced from 48 MiB to 16.125 MiB per profile. The full table remains in use when calibration cannot separate it into smaller components. Construction still temporarily needs the full table, so start-up peak memory is not reduced by the full difference. Fused sampling fills only the letterbox borders and writes the image area directly.

Two further options are disabled by default and can be benchmarked independently.

```json
{
  "metadata_async": true,
  "capture": { "host_yuv": true },
  "gpu": { "asynchronous": true, "direct_pinned": true, "pipeline_depth": 1 }
}
```

`gpu.direct_pinned` writes fused and per-image fallback preprocessing results directly to a reserved pinned slot. It saves one model-sized CPU buffer copy and still uploads the input to the GPU. It requires asynchronous mode. Errors and panics release unsubmitted reservations. Submitted slots become available after collection.

`metadata_async` uses a latest-only mailbox that holds one snapshot of tracking and geometry data per camera, without images. When Redis slows down, visual state updates are coalesced. Evidence JPEGs and events still use the persistent outbox. Shutdown sends the last snapshot and cleans up its owner. `summary.json` includes `metadata_offered`, `metadata_sent`, `metadata_superseded` and `metadata_errors`.

The hazard engine reuses polygons while cone centres and the pole circles used for geometry are identical. The resulting coordinates are shared with annotation, metadata and evidence. `geometry_cache_hits` and `geometry_cache_misses` count each enabled geometry check in `summary.json`.

Live output creates its native owner when viewing demand and the first frame are both present. An expired lease closes the thread and muxer. Playback after renewal waits for a keyframe before reinitialising. The shared font system initialises on the first server-side text render, and `fonts_initialized` reports its process-wide status.

Choose either `host_yuv` or `nvdec_device` as the capture mode.

PostgreSQL-managed cameras publish server-annotated preview and detail streams by language. Set `capture.host_yuv: true` with CPU decoding to draw directly into the encoder's YUV planes. Configure the preview resolution and use `database.media.on_demand` with Redis to open encoders only for requested streams and languages.

Flutter requests the `overlay` profile and the selected language through the playback API. A Rust playback wall supports up to 32 cameras. Flutter plays each returned HLS URL and renews the wall lease.

### 20 FPS annotated previews

Merge this fragment into a PostgreSQL camera configuration that already uses Redis viewing leases. It selects CPU YUV capture and 640 × 360 previews encoded at 21 FPS, with one frame per second of headroom for a target of 20 distinct frames.

```json
{
  "cpu_workers": 16,
  "metadata_async": true,
  "capture": { "host_yuv": true, "max_convert_fps": 0 },
  "database": {
    "media": {
      "on_demand": true,
      "preview": { "width": 640, "height": 360, "fps": 21 }
    }
  }
}
```

For 20 distinct preview frames per second, start with cameras that deliver a stable 24, 25 or 30 FPS. This gives the preview encoder margin for ordinary source jitter. Standalone stream configurations set the same target in `streams[].output.preview.fps`. On a run with only the preview active, divide `encoded_unique` by the camera's elapsed seconds to measure distinct FPS. Compare `encoded_repeated`, `capture_rate_dropped` and `render_skipped` to check whether the source, capture limit or annotation queue reduced it.

## Native Rust RTSP configuration

```json
{
  "capture": { "rust_rtsp": true, "max_convert_fps": 15 }
}
```

`capture.rust_rtsp` defaults to `false`. When enabled for `rtsp://`, the Rust `retina` library handles TCP RTSP and RTP reception, H.264 and H.265 reassembly and timestamps. CPU decoding still uses in-process libavcodec. This ingest mode does not require NVDEC, OpenCV capture or an FFmpeg subprocess.

The option changes the receiver. The rest of the pipeline still handles decoding, BGR conversion and model preprocessing. `rtsps://` and local files continue to use libavformat. If the Rust receiver encounters an unsupported codec, it reports the error and follows the existing reconnect logic.

Each source can queue up to two complete compressed frames. A full queue applies backpressure to preserve reference frames. The receiver waits for a random-access frame on initial connection or after RTP packet loss. After loss, it rebuilds the decoder and advances the generation to discard earlier tracking results. A parameter change triggers reconnection. Handshakes and video stalls have deadlines. Stopping cancels the receiver and closes the TCP connection without leaving background TEARDOWN retries.

`summary.json` includes `rust_rtsp_frames`, `rust_rtsp_bytes`, `rust_rtsp_lost_packets`, `rust_rtsp_keyframe_dropped`, `rust_rtsp_handoff_ms` and `rust_rtsp_decoder_resets`. Frame and byte counts cover complete access units and exclude network headers. Handoff time is the cumulative wait for queue handoffs across the run. Measure frame or end-to-end latency separately. Validate long-running capture and reconnection on the target camera model under expected network conditions before adopting the option.

## Optional NVDEC and NV12 capture

Merge the fragment into the full configuration. The `tensorrt` build also enables `construction-media/cuda` and requires CUDA headers and a driver library, along with FFmpeg built with NVDEC support. The CPU-only build uses the existing CPU dependencies.

```json
{
  "capture": { "nvdec_device": 0, "max_convert_fps": 15 },
  "gpu": { "device": 0, "asynchronous": true, "pipeline_depth": 1 }
}
```

NVDEC is disabled by default, must use the inference device and is unavailable with replay models. Supported 8-bit NV12 frames remain on the GPU at their native resolution. CUDA fuses sampling, letterboxing and RGB CHW normalisation, using a shared 48 MiB colour table per device and profile calibrated at start-up against the FFmpeg YUV420P converter.

The direct path requires a width divisible by 32, same-size or proportional odd-integer downscaling, and BT.601 or BT.709. Other batches are downloaded for CPU preprocessing. In this mode, `max_convert_fps` limits how many frames are retained for inference and viewing while NVDEC continues decoding. Set it to `0` to measure the full source rate.

Frames are downloaded and cached as BGR only when overlays, evidence JPEGs or the current encoder need pixels. The existing CPU path handles output. Without PTS, the freeze check samples at most once per second. When PTS is available, the pipeline checks its progress directly.

The summary records `gpu_admitted`, `gpu_materialized` and `gpu_download_payload_bytes`. The trace records `nv12_preprocess`.

## Verification

Run these commands from the repository root. Unit tests can run directly. The full file replay and outbox tests require PostgreSQL.

```bash
cargo +1.93.1 fmt --manifest-path native/pipeline/Cargo.toml -- --check
cargo +1.93.1 test --locked --manifest-path native/pipeline/Cargo.toml --lib
node --test native/pipeline/web/overlay.test.mjs
```

The following Docker example starts an isolated PostgreSQL container under the same database name as CI. `CHD_RUST_TEST_DATABASE_URL` accepts only a loopback connection to `chd_rust_pipeline_test`.

```bash
(
  set -eu
  test_db="$(docker run --detach --rm \
    -p 127.0.0.1:25433:5432 \
    -e POSTGRES_USER=fixture -e POSTGRES_PASSWORD=fixture \
    -e POSTGRES_DB=chd_rust_pipeline_test postgres:17.11-bookworm)"
  trap 'docker stop "$test_db" >/dev/null' EXIT
  for attempt in $(seq 1 30); do
    if docker exec "$test_db" pg_isready -h 127.0.0.1 -U fixture -d chd_rust_pipeline_test >/dev/null; then
      break
    fi
    sleep 1
  done
  docker exec "$test_db" pg_isready -h 127.0.0.1 -U fixture -d chd_rust_pipeline_test
  export CHD_RUST_TEST_DATABASE_URL=postgresql://fixture:fixture@127.0.0.1:25433/chd_rust_pipeline_test
  cargo +1.93.1 test --locked --manifest-path native/pipeline/Cargo.toml
)
```

CI also starts isolated Redis and MediaMTX instances to test 32 RTSP sessions, language viewing leases, catalogue updates and expiry. The 32 sessions share a local video source and replay detections, exercising correctness in isolation. To estimate real camera capacity, benchmark with the target model, sources and viewing demand. [CI workflow](../../.github/workflows/rust-pipeline.yml)

To compare with Python hazard rules or TensorRT inference on a CUDA device, install the repository's Python dependencies and run the relevant script. Use `--help` for binary, engine and output directory options. [Hazard rule comparison script](../../scripts/validate_rust_pipeline.py), [CUDA validation script](../../scripts/validate_standalone_cuda.py)
