🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLO Server API Backend

Optional FastAPI detector service for HTTP/WebSocket testing, remote detector
deployment, and model file management. The recommended high-throughput local
runtime is still `src/yolo_worker.py`, where `main.py` sends frames through
shared memory and avoids JPEG plus WebSocket overhead.

Use this service when you deliberately want inference behind an API boundary.

## What It Provides

- `POST /detect`: accepts one uploaded image and returns YOLO detections.
- `WebSocket /ws/detect`: accepts image bytes and returns detections.
- `POST /model_file_update`: uploads a replacement `.pt` model file.
- `POST /get_new_model`: returns an updated model file when a client has an old
  timestamp.

Detection results are lists in this shape:

```text
[x1, y1, x2, y2, confidence, class_id]
```

## Files

- `app.py`: FastAPI application and lifespan.
- `routers.py`: detection and model-management routes.
- `websocket_handlers.py`: optional WebSocket detection path.
- `detection.py`: image decode, inference, and bounding-box post-processing.
- `models.py`: model manager and file watcher.
- `model_files.py`: model upload and retrieval helpers.
- `config.py`: runtime settings, including device selection.
- `schemas.py`: request and response models.

## Run

From the repository root:

```bash
uvicorn examples.YOLO_server_api.backend.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1
```

Use one worker per GPU model instance unless you intentionally want duplicate
model loads. Multiple Uvicorn workers do not share a loaded PyTorch model.

## Configuration

Common environment variables:

```dotenv
DETECT_API_AUTH_REQUIRED=true
DETECT_SERVER_MODEL_KEYS=yolo26n,yolo26s
DETECT_API_URL=http://127.0.0.1:8000
YOLO_MODEL_DIR=models/pt
```

Model files are expected to follow the project convention:

```text
models/pt/best_<model_key>.pt
```

For example, `model=yolo26n` loads `models/pt/best_yolo26n.pt`.

## Authentication

Routes use the shared JWT dependencies from `examples.auth`. Use the database
management API to log in and pass the access token as:

```text
Authorization: Bearer <access-token>
```

Model upload endpoints require model-management privileges.

## Performance Notes

- For the main multi-camera pipeline, prefer local YOLO workers in `src/`.
- This API path encodes or decodes images at the process boundary, which costs
  CPU time and memory bandwidth.
- WebSocket mode is useful for compatibility tests, but it is not the fastest
  local path.
