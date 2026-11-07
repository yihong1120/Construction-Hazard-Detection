🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Construction Hazard Detection

<img width="100%" src="./assets/images/project_graphics/banner.gif" alt="AI-driven construction safety monitoring banner">

<div align="center">
   <a href="examples/YOLO_server_api">Server API</a> |
   <a href="examples/local_notification_server">FCM Notification Server</a> |
   <a href="examples/violation_records">Violation Records Server</a> |
   <a href="examples/bff">Web BFF</a> |
   <a href="https://github.com/yihong1120/visionnaire-flutter">Flutter App</a> |
   <a href="examples/db_management">Data Management Server</a> |
   <a href="examples/streaming_web">Streaming Web</a> |
   <a href="examples/mcp_server">MCP Server</a> |
   <a href="examples/YOLO_data_augmentation">Data Augmentation</a> |
   <a href="examples/YOLO_evaluation">Evaluation</a> |
   <a href="examples/YOLO_train">Train</a>
</div>

<br>

<div align="center">
   <a href="https://www.python.org/downloads/">
      <img src="https://img.shields.io/badge/python-3.15-blue?logo=python" alt="Python 3.15">
   </a>
   <a href="https://github.com/ultralytics/ultralytics">
      <img src="https://img.shields.io/badge/ultralytics-8.4.183-blue?logo=yolo" alt="Ultralytics 8.4.183">
   </a>
   <a href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html">
      <img src="https://img.shields.io/badge/HDBSCAN-sklearn-orange?logo=scikit-learn" alt="HDBSCAN sklearn">
   </a>
   <a href="https://fastapi.tiangolo.com/">
      <img src="https://img.shields.io/badge/FastAPI-0.128.0-blue?logo=fastapi" alt="FastAPI 0.128.0">
   </a>
   <a href="https://redis.io/">
      <img src="https://img.shields.io/badge/redis--py-7.1.0-red?logo=redis" alt="redis-py 7.1.0">
   </a>
   <a href="https://www.docker.com/">
      <img src="https://img.shields.io/badge/Docker-Container-blue?logo=docker" alt="Docker">
   </a>
   <a href="https://codecov.io/github/yihong1120/Construction-Hazard-Detection">
      <img src="https://codecov.io/github/yihong1120/Construction-Hazard-Detection/graph/badge.svg?token=E0M66BUS8D" alt="Codecov">
   </a>
   <a href="https://universe.roboflow.com/object-detection-qn97p/construction-hazard-detection">
      <img src="https://app.roboflow.com/images/download-dataset-badge.svg" alt="Download Dataset from Roboflow">
   </a>
   <a href="https://huggingface.co/yihong1120/Construction-Hazard-Detection">
      <img src="https://img.shields.io/badge/HuggingFace-Model%20Repo-yellow?logo=huggingface" alt="Hugging Face Model Repo">
   </a>
</div>

<br>

AI-assisted construction-site safety monitoring for live cameras. The current
runtime is centred on `main.py`, `src/stream_processor.py`, local YOLO worker
processes, MediaMTX live publishing, PostgreSQL records, Redis coordination,
and FastAPI services for management, notifications, streaming metadata, and
violation records.

## What It Detects

- Workers without hard hats.
- Workers without safety vests.
- Workers too close to machinery or vehicles.
- Workers inside cone-derived controlled areas.
- Machinery or vehicles too close to utility poles.

Supported label and notification languages include Traditional Chinese,
Simplified Chinese, English, French, Thai, Vietnamese, Indonesian, and
Japanese.

<img width="100%" src="./assets/images/hazard-detection.png" alt="Construction hazard detection examples">

## Hazard Detection Examples

Below are examples of real-time hazard detection by the system.

<div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
  <div style="text-align: center; flex-basis: 33%;">
    <img src="./assets/images/demo/person_did_not_wear_safety_vest.png" alt="Workers without helmets or safety vests" style="width: 300px; height: 200px; object-fit: cover;">
    <p>Workers without helmets or safety vests</p>
  </div>
  <div style="text-align: center; flex-basis: 33%;">
    <img src="./assets/images/demo/person_near_machinery.jpg" alt="Workers near machinery or vehicles" style="width: 300px; height: 200px; object-fit: cover;">
    <p>Workers near machinery or vehicles</p>
  </div>
  <div style="text-align: center; flex-basis: 33%;">
    <img src="./assets/images/demo/persons_in_restricted_zones.jpg" alt="Workers in restricted areas" style="width: 300px; height: 200px; object-fit: cover;">
    <p>Workers in restricted areas</p>
  </div>
</div>

## Runtime Architecture

<img width="100%" src="./assets/flowcharts/site_safety_monitor_en.png" alt="Construction hazard detection runtime architecture">

The diagram summarises the live data path, the 11 configured detection classes,
the safety-warning rules, and the video/metadata outputs. Redis is not used to
store live video frames; MediaMTX owns live playback, while Redis keeps only
small coordination state such as authentication cache, FCM token cache, compact
warning metadata, overlay demand keys, and overlay ready keys.

## Repository Map

- `main.py`: supervises configured streams and worker processes.
- `src/`: production runtime modules.
- `examples/bff/`: Web session, CSRF, media capability, and API gateway.
- `examples/db_management/`: users, groups, sites, stream configuration API.
- `examples/local_notification_server/`: FCM token and site notification API.
- `examples/streaming_web/`: labels, playback URLs, metadata channels,
  media-session auth, and WebRTC ICE settings.
- `examples/violation_records/`: violation record and image API.
- `examples/YOLO_server_api/`: optional standalone YOLO API.
- `examples/mcp_server/`: optional AI tools for detection, records, models and multi-platform notifications; [setup](examples/mcp_server/README.md).
- `examples/YOLO_train/`, `examples/YOLO_evaluation/`,
  `examples/YOLO_data_augmentation/`: training, export, evaluation and augmentation tools.
- `scripts/`: database initialisation and TensorRT rebuild helpers.

Notifications support FCM, LINE Messaging API, Messenger, Telegram, WeChat
and HTTP broadcast. Model-development tools remain in this repository with optional dependencies.

## Flutter Frontend

The Flutter source is maintained separately in
[visionnaire-flutter](https://github.com/yihong1120/visionnaire-flutter).
`frontend/web/` in this repository is an ignored, generated deployment
artifact, not editable frontend source. See [frontend/README.md](frontend/README.md)
for the release and backend-contract boundary.

## Recommended Runtime

Use database mode unless you are testing a short-lived local JSON config.

1. PostgreSQL stores sites, users, stream configurations, and violations.
2. Redis stores auth/session cache, notification token cache, and compact live
   coordination keys.
3. MediaMTX serves RTSP ingest plus HLS/WebRTC playback.
4. `main.py` polls stream configuration and starts one stream process per active
   camera.
5. Local YOLO workers share GPU inference across cameras through shared memory.

The standalone YOLO server remains available for API testing or separate
deployment, but the recommended high-throughput local path is the YOLO worker
mode in `src/yolo_worker.py`.

## Quick Start

### 1. Prepare Python

The project requires Python `>=3.15,<3.16` and uses PEP 810 explicit lazy imports.
For the shared API runtime:

```bash
uv sync --locked
```

For complete training and INT8 TensorRT export on Python 3.15, run in the target environment:

```bash
conda activate construction_env
python deploy/python315/bootstrap.py
```

This uses official PyTorch CUDA 13.0 CPython 3.15 wheels and builds ONNX Runtime
1.31.0 and TensorRT 11.4.0.106 Python bindings from pinned upstream revisions.
It requires Linux x86_64, standard CPython 3.15 with the GIL enabled, `uv`, `git`,
a C++20 compiler and CUDA development headers (`/usr/local/cuda/include` by default;
set `CUDA_HOME` to override). Local wheels are written to `wheels/python315/`.
Allow tens of GB of disk space for the first build; use `--jobs 4` to reduce build
memory usage or `--build-only` to skip installation. These are local native builds,
not official Microsoft/NVIDIA Python 3.15 support. Run the bootstrap on a new host
rather than only installing `requirements.txt`.

For CPU calibration, development and tests without TensorRT or CUDA headers,
run `python deploy/python315/bootstrap.py --cpu-only` instead. The bootstrap
installs pinned ONNX Runtime/TensorRT builds separately from the registry-only
`uv.lock`; `uv sync` alone does not install these native packages. CI uses this
CPU path, and the GPU Docker base builds its native wheels inside the image.

On Python 3.15, `examples/YOLO_train/export_int8_trt.py` and `export_int8_engine.py` use ONNX Runtime CPU
calibration to produce an INT8 Q/DQ graph and TensorRT to build and execute the GPU
engine. Image sampling, calibration over all images and node exclusions are retained.
This replaces ModelOpt, which does not yet support Python 3.15. Evaluate quantized
model accuracy on your validation dataset before deployment.

The default dependency set contains only shared API runtime packages.  Docker
images install their own `streaming`, `violation`, `notification`, or `yolo`
extra; use `--extra yolo-gpu` only on CUDA/TensorRT hosts.


Use `uv sync --extra social-notifications` for multi-platform notifications,
or `uv sync --extra mcp --extra social-notifications` with MCP. Install
`uv sync --extra training` for the registry training dependencies; use the
bootstrap above to include native calibration/export dependencies.

Optional inference and notification dependencies are declared with `lazy import`
at module scope. Keep eager imports for route/model registration and catch
optional dependency errors at their first use. Deployment scripts that configure
the environment or module search path before importing retain that ordering.

### 2. Download Models

```bash
hf download yihong1120/Construction-Hazard-Detection \
  --repo-type model \
  --include "models/pt/*.pt" \
  --local-dir .
```

Expected worker model filenames are `models/pt/best_<model_key>.pt`, for
example `models/pt/best_yolo26n.pt`.

### 3. Create `.env`

Start from `.env.example`, then adjust hostnames and secrets.

Important values:

```dotenv
DATABASE_URL='postgresql+asyncpg://username:password@127.0.0.1/construction_hazard_detection'

REDIS_HOST='127.0.0.1'
REDIS_PORT=6379
REDIS_PASSWORD='set-a-strong-password'

OIDC_ENABLED=true
OIDC_ISSUER_URL=https://sso.example.com/realms/visionnaire
OIDC_JWKS_URL=https://sso.example.com/realms/visionnaire/protocol/openid-connect/certs
OIDC_AUDIENCE=visionnaire-api
OIDC_ACCOUNT_URL=https://sso.example.com/realms/visionnaire/account

DB_MANAGEMENT_API_URL='http://127.0.0.1:8005'
FCM_API_URL='http://127.0.0.1:8003'
VIOLATION_RECORD_API_URL='http://127.0.0.1:8002'
STREAMING_API_URL='http://127.0.0.1:8800'

YOLO_WORKER_CAMERAS_PER_ENGINE=3
# Override capacity by model size; unlisted model keys use the default above.
YOLO_WORKER_CAMERAS_PER_ENGINE_BY_MODEL=yolo26n=8,yolo26s=6,yolo26m=4,yolo26l=2,yolo26x=1
YOLO_WORKER_DEVICES=cuda:0,cuda:0
YOLO_WORKER_QUEUE_SIZE=64
YOLO_WORKER_RESULT_QUEUE_SIZE=8
YOLO_WORKER_RING_SLOTS=2
YOLO_WORKER_RING_SLOT_CLEANUP_SECONDS=120
YOLO_WORKER_BATCH_SIZE=8
YOLO_WORKER_BATCH_WAIT_MS=10
YOLO_WORKER_PRECISION=f16
# f32/f16 use models/pt/*.pt; int8 uses models/int8_engine/*.engine.

# Each camera keeps a bounded shared-memory frame ring and receives results on
# its own queue. For Docker, set YOLO_WORKER_SHM_SIZE high enough for
# camera_count * YOLO_WORKER_RING_SLOTS * maximum_frame_bytes.

MEDIA_PUBLISH_RTSP_BASE_URL='rtsp://127.0.0.1:8554'
MEDIA_PUBLIC_HLS_BASE_URL='/hazard/media'
MEDIA_PUBLIC_WEBRTC_BASE_URL='/hazard/media/webrtc'
MEDIA_PUBLISH_CLEAN_STREAM=true
MEDIA_PUBLISH_ANNOTATED_STREAM=true
# Detail clean video uses the shared latest-frame capture and Intel iGPU. This
# drops stale frames, avoids a second RTSP connection to each camera, and keeps
# HLS keyframes regular. Enable direct source restreaming only when the
# untouched camera bitstream is explicitly required.
MEDIA_PUBLISH_CLEAN_SOURCE_RESTREAM=false
MEDIA_PUBLISH_CLEAN_ENCODER=h264_vaapi
MEDIA_PUBLISH_ENCODER=h264_vaapi
MEDIA_PUBLISH_VAAPI_DEVICE=/dev/dri/renderD128
# Share viewer-demand reads between a camera's clean/detail/preview publishers.
MEDIA_DEMAND_CACHE_SECONDS=0.5
```

API services share the configured OIDC issuer, JWKS URL and API audience.

### 4. Start Infrastructure

Redis, PostgreSQL, and MediaMTX are required for the normal database-backed
runtime:

```bash
docker compose up -d redis postgres media-server
```

Check that the containers are running:

```bash
docker compose ps redis postgres media-server
docker compose logs -f redis postgres media-server
```

If PostgreSQL and Redis are managed on the host and only MediaMTX is needed,
use the standalone Compose file. It does not interpolate database or Redis
credentials:

```bash
docker compose -f docker-compose.media.yml up -d
docker compose -f docker-compose.media.yml ps
docker compose -f docker-compose.media.yml logs -f media-server
```

The full `docker-compose.yml` starts its own PostgreSQL container, so it
requires `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD` separately.
Those fields cannot be safely or reliably reconstructed from `DATABASE_URL`.

When the Python services run on the host, use local addresses in `.env`:

```dotenv
DATABASE_URL='postgresql+asyncpg://username:password@127.0.0.1/construction_hazard_detection'
REDIS_HOST='127.0.0.1'
REDIS_PORT=6379
REDIS_PASSWORD='password'
MEDIA_PUBLISH_RTSP_BASE_URL='rtsp://127.0.0.1:8554'
```

When `main.py` or the FastAPI services run inside Docker Compose, use service
names instead:

```dotenv
DATABASE_URL='postgresql+asyncpg://username:password@postgres/construction_hazard_detection'
REDIS_HOST='redis'
POSTGRES_DB='construction_hazard_detection'
POSTGRES_USER='construction_app'
POSTGRES_PASSWORD='set-a-strong-password'
REDIS_PASSWORD='set-a-strong-password'
MEDIA_PUBLISH_RTSP_BASE_URL='rtsp://media-server:8554'
```

Infrastructure roles:

- PostgreSQL stores users, sites, stream settings, and violation records.
- Redis stores authentication cache, token cache, compact live metadata, and
  overlay demand keys.
- MediaMTX receives RTSP streams from `main.py` and exposes HLS/WebRTC
  playback.

Redis and MediaMTX do not store violation images or database records.

For a new, empty PostgreSQL database, import the bootstrap schema if it was
not mounted at container creation time. This script drops application tables,
so never run it against a database that contains data:

```bash
cat ./scripts/init.postgres.sql | docker exec -i postgres-container \
  psql -U username -d construction_hazard_detection
```

For an existing database, follow the organisation's database change-management
runbook. Migration assets and privileged operator tooling are intentionally
not distributed with the application source.

### 5. Configure MediaMTX

MediaMTX is the live media server. `main.py` publishes processed H.264 streams
to MediaMTX over RTSP, and viewers play those streams through HLS or WebRTC.
Redis only stores compact metadata and demand keys; it does not carry video
frames.

The Docker Compose service exposes local-only ports:

```text
127.0.0.1:8554  RTSP ingest from main.py / ffmpeg
127.0.0.1:8890  HLS playback, mapped to MediaMTX port 8888
127.0.0.1:8889  WebRTC/WHEP playback
```

Use these settings when `main.py` runs on the host:

```dotenv
MEDIA_PUBLISH_RTSP_BASE_URL='rtsp://127.0.0.1:8554'
MEDIA_PUBLIC_HLS_BASE_URL='/hazard/media'
MEDIA_PUBLIC_WEBRTC_BASE_URL='/hazard/media/webrtc'
```

The host account that runs `main.py` needs access to the Intel render node:

```bash
sudo usermod -aG render "$USER"
# Sign out and back in before starting main.py.
```

With the configured on-demand publishers, clean detail video uses FFmpeg
stream copy and consumes no encoder session. Only active preview and annotated
paths use `h264_vaapi` on the Intel iGPU; MediaMTX then remuxes and distributes
that one encoded stream to all viewers. This avoids the RTX 4090 GeForce NVENC
eight-session policy without moving overlay drawing into the browser.

Use the Compose service name when `main.py` runs inside Docker:

```dotenv
MEDIA_PUBLISH_RTSP_BASE_URL='rtsp://media-server:8554'
```

Docker Compose maps `/dev/dri/renderD128` into the detection container. Set
`INTEL_RENDER_GID` to the host `render` group ID if it is not `992`.

The streaming backend returns playback URLs shaped like:

```text
/hazard/media/<media-path>/index.m3u8
/hazard/media/webrtc/<media-path>/whep
```

For public deployments, proxy MediaMTX through Nginx and protect it with
`examples.streaming_web` media authorisation. Use
`examples/streaming_web/nginx.hazard-media.conf` as the starting point:

```text
Nginx /hazard/media/*        -> MediaMTX HLS port 8888
Nginx /hazard/media/webrtc/* -> MediaMTX WebRTC port 8889
Nginx /hazard/api/media-auth -> examples.streaming_web /media-auth
```

Recommended live-buffer defaults are already in `docker-compose.yml`:

```dotenv
MTX_HLSSEGMENTDURATION=2s
MTX_HLSSEGMENTCOUNT=14
MTX_HLSALWAYSREMUX=yes
MTX_HLSMUXERCLOSEAFTER=60s
```

Lower `MTX_HLSSEGMENTCOUNT` reduces disk and memory use. The default 14
segments (about 28 seconds at 2-second segments) gives wall tiles enough room
to recover from a short publisher or network interruption.

### 6. Start APIs

Run each service from the repository root:

```bash
uvicorn examples.db_management.app:app --host 127.0.0.1 --port 8005 --workers 2 --timeout-graceful-shutdown 10
uvicorn examples.local_notification_server.app:app --host 127.0.0.1 --port 8003 --workers 2
uvicorn examples.violation_records.app:app --host 127.0.0.1 --port 8002 --workers 2
uvicorn examples.streaming_web.app:app --host 127.0.0.1 --port 8800 --workers 2 --timeout-graceful-shutdown 10
```

Optional standalone detector API:

```bash
uvicorn examples.YOLO_server_api.app:app --host 127.0.0.1 --port 8000 --workers 2
```

### 7. Start Stream Processing

```bash
python main.py
```

Optional polling interval:

```bash
python main.py --poll 5
```

Optional file-based mode for development:

```bash
python main.py --config config/configuration.json
```

Do not run database mode and JSON mode at the same time for the same cameras.

## Live Viewing

Clients should call the streaming web backend:

- `GET /labels`
- `GET /streams/{label}`
- `GET /metadata/stream-id/{label}/{stream_id}` for SSE metadata
- `WebSocket /ws/metadata-id/{label}/{stream_id}` for metadata
- `GET /webrtc/ice-servers` when WebRTC needs STUN/TURN settings

Video playback comes from MediaMTX HLS/WebRTC URLs. Warning metadata is compact
and usually contains only current warning state. Detection boxes, polygons, and
labels are rendered into backend-published annotated video streams.

For public deployments, put MediaMTX behind Nginx `auth_request` and let
`examples.streaming_web` validate JWT and site access.

## Storage Retention

Violation images are stored under each service's configured `static/` location.
Without archive/NAS/external disk, a practical default is:

- keep images and DB records for 18 months;
- delete image files and matching database rows together;
- run cleanup during low-traffic hours;
- keep HLS segment retention short because MediaMTX HLS is a live buffer, not
  an archive.

Do not delete database records while keeping broken image paths, and do not
delete image files while keeping records that the UI still needs to display.

## Development Checks

```bash
pre-commit run -a
python -m pytest -q --tb=short
```

The mypy pre-commit hook checks production code (`main.py`, `src`, `examples`,
and `scripts`). Tests are still validated by `pytest` and `flake8`; many test
files intentionally use dynamic mocks and invalid payloads.

## Dataset And Models

The project uses the Construction Hazard Detection model repository:

- Hugging Face: <https://huggingface.co/yihong1120/Construction-Hazard-Detection>
- Roboflow: <https://universe.roboflow.com/side-projects/construction-hazard-detection>

Labels:

```text
0 Hardhat
1 Mask
2 NO-Hardhat
3 NO-Mask
4 NO-Safety Vest
5 Person
6 Safety Cone
7 Safety Vest
8 Machinery
9 Utility Pole
10 Vehicle
```

## Licence

This project is licensed under the [AGPL-3.0 Licence](LICENSE.md).

### Durable evidence delivery

The dispatcher shares one worker token provider and pooled credential transport.
Credential requests make one attempt; event delivery owns the retry/backoff budget.
Its FIFO camera queue uses bounded workers, requeues busy cameras at the tail,
and drains successful backlogs without waiting for the next directory scan.
`VIOLATION_OUTBOX_CONCURRENCY` defaults to 4 and
`VIOLATION_OUTBOX_EVENT_TIMEOUT_SECONDS` to 60 (maximum 100, below the 120-second
claim lease). Cancellation releases in-flight claims before transports close.

The camera process only writes versioned evidence metadata and image bytes to
its local SQLite outbox. The supervisor owns one dispatcher for every persisted
camera queue, so disabling a camera does not strand its pending events. The
outbox also reads the previous unversioned payload format. Keep its persistent
volume when deploying the updated worker.

Evidence upload and notification delivery have separate completion stages. An
uploaded ID is persisted and its image bytes are released before notification
retries. No subscribers or sendable devices complete as `skipped`; temporary
failures retry with backoff; permanent failures and exhausted retries remain
quarantined with their failure reason. The queue's byte/event budgets include
quarantined evidence, so capacity never silently grows or discards failed images.

```bash
python -m src.delivery_outbox_cli
python -m src.delivery_outbox_cli --retry EVENT_ID
```

Set `VIOLATION_OUTBOX_MAX_ATTEMPTS` (default 20) and
`VIOLATION_OUTBOX_POLL_SECONDS` (default 1) to tune recovery. Failed events require
operator inspection and explicit retry after the underlying cause is fixed.
Delivery leases prevent concurrent dispatchers from taking the same event.
Provider push acknowledgements remain at least once across an uncertain network
response; the event UUID and notification-centre recipient key prevent duplicate
evidence and in-app records.

Native social links stage and validate the local mirror in one transaction
before calling Keycloak. An uncertain remote result or failed commit reports
`account_reconciliation_required`; it never removes a shared provider identity
as compensation. Repeating the normal link flow with fresh verified proof
recognises the existing remote link and repairs the local mirror.

Native social orchestration shares a `VerifiedSocialIdentity` across account
lookup and linking; proof validation, one-use transaction storage and broker
redemption have separate modules. FCM delivery captures registration revisions
before sending. Its result updates require the same owner and registration;
successful sends update diagnostics without activating a disabled device.

Playback negotiation validates an entire batch before atomically committing its
sessions, producer leases and reverse indexes. A missing or foreign existing
session leaves the batch unchanged. Metadata subscriptions own their overlay
lease, readiness and heartbeat state and close their shared fanout subscription
when the viewer disconnects.


Media publishing captures `MediaPublishSettings` for each stream. By default,
capture keeps every decoded frame and detail, overlay and preview outputs follow
the camera frame rate, including fractional rates such as 29.97 FPS. Reconnecting
reads the camera rate again and replaces publishers if it changes. Inference
consumes the latest frame independently, so slower inference does not set the
video frame rate. Preview reduces resolution and bitrate without reducing FPS.

`MEDIA_PUBLISH_SOURCE_FPS`, `MEDIA_PUBLISH_FPS`, `MEDIA_PREVIEW_FPS` and
`MEDIA_PUBLISH_CLEAN_FPS` default to `auto` (an empty value also means auto).
A numeric value explicitly enables capture sampling or output resampling;
preview inherits an explicit detail rate unless separately configured.
Direct source restreaming preserves input timestamps in auto mode. Raw-frame
publishers temporarily use 30 FPS for startup slates or cameras that do not
report a usable FPS. A renderer holds its settings and copies each shared source
frame once per rendered language.
All renditions and the source restreamer share the same captured encoder settings
for a camera, including codec options, GOP, bitrate and device selection.

Model publication uses `python -m scripts.sync_huggingface_models`; see
[`MODEL_CATALOG.md`](examples/YOLO_server_api/MODEL_CATALOG.md) for automatic
affected-worker restarts, loaded-version verification and restoration behavior.

Evidence timestamps without an offset are interpreted as UTC consistently across
HTTP, MCP, outbox serialization and request fingerprints. Explicit offsets are
converted to UTC. Use offset-bearing timestamps when retrying historical events
that previously relied on a server's local timezone.

Live Metadata is a latest-snapshot channel; violation/notification delivery is
handled separately. Redis reads merge a bounded batch (default 64 entries,
`STREAMING_METADATA_READ_BATCH_SIZE`); a full batch resolves the newest entry in
one extra bounded read to skip backlog. Normal batches require no extra read.
This policy does not change source, publish or preview FPS.

Tracked violation rows use one validation contract at HTTP and MCP upload
boundaries: finite numeric values, positive box area, confidence in [0, 1], a
nonnegative integer class ID, and an integer track ID >= -1 (-1 means unassigned).
Coordinates may extend outside the frame for later clipping. Invalid uploads
return 422 rather than being serialized as null values. Historical records remain
accessible: invalid stored detection rows are omitted from rendering without
rewriting the original evidence.

BFF refreshes renew their ownership lease and save token pairs only when both
the owner and session revision still match. Expired refreshers cannot overwrite
newer tokens or revoke a newer session after a late authentication error. Browser
cancellation drains an in-progress rotation and save before releasing ownership.
A process crash or an exchange whose response is lost can still require sign-in.

HLS-driven playback renewal reads, watches and updates at most
`STREAM_PLAYBACK_REFRESH_BATCH_SIZE` live sessions per batch (default 256), while
pruning a bounded batch of expired reverse-index entries. Renewed scores move to
the end so later batches can reach other sessions. Throttling and index/session
changes commit in the same Redis transaction.

Forced BFF refresh waiters require both an advanced session revision and a
different valid access token. A busy peer cannot send the rejected token back
upstream merely because its expiry is still in the future. When a peer finishes
before this request acquires the lock, its valid token is reused without another
exchange.

The evidence dispatcher supervises its scheduler and recreates delivery resources
after failure, preserving disk queues and event leases. Recovery starts at
`VIOLATION_OUTBOX_RECOVERY_SECONDS` (1), doubles up to
`VIOLATION_OUTBOX_MAX_RECOVERY_SECONDS` (60), and resets after a successful poll.
Its `health` snapshot exposes running state, cumulative failures, last error type
and last successful poll. Shutdown interrupts recovery and drains active work.

Notification claims track a conservative monotonic deadline from each confirmed
acquisition or renewal. Redis operations take at most five seconds and owned
operations cannot exceed the confirmed lease deadline. An unconfirmed lease stops
new batches and drains cancellation; already accepted FCM requests still have
at-least-once delivery semantics.

A dead YOLO shard restarts with a new IPC generation: fresh request/result queues
replace potentially poisoned multiprocessing locks, and affected camera
processes are rebound through configuration reconciliation. Healthy shards keep
their IPC. Model startup uses an OS-owned file lock released automatically when
a loading process dies. Source, publish and preview FPS are unchanged.

Environment configuration is grouped into typed domain settings; clients and
workers retain a validated snapshot for their lifetime. Explicit constructor
values override environment values, including an explicit empty Redis password.
Malformed numeric, boolean and MCP transport settings fail at configuration
loading. See [configuration ownership](src/settings/README.md) for lifecycle,
credential rotation and validation rules. Stream FPS selection is unchanged.
