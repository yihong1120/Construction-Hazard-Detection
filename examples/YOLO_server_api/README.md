🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

Model discovery, defaults, and the Flutter integration contract: [Model API](DETECTION_MODEL_API.md) (Traditional Chinese).

# YOLO Server API

Optional FastAPI detector service for HTTP/WebSocket testing, remote detector
deployment, and model file management. The recommended high-throughput local
runtime is still `src/yolo_worker.py`, where `main.py` sends frames through
shared memory and avoids JPEG plus WebSocket overhead.

Use this service when you deliberately want inference behind an API boundary.

## What It Provides

- `POST /detect`: accepts one uploaded image and returns YOLO detections.
- `WebSocket /ws/detect`: accepts image bytes and returns detections.
- `POST /model_file_update`: the former `.pt` upload endpoint checks model-management
  privileges and returns HTTP 409 `IMMUTABLE_MODEL_REGISTRY` to authorised users.
  Publish a new version on Hugging Face and synchronise the catalogue key.
- `POST /get_new_model`: validates the supplied timestamp and returns the verified model
  file, or HTTP 304 when the client’s `If-None-Match` matches the artefact’s ETag.

Detection results are lists in this shape:

```text
[x1, y1, x2, y2, confidence, class_id]
```

## Files

- `app.py`: FastAPI application and lifespan.
- `routers.py`: detection and model-management routes.
- `websocket_handlers.py`: optional WebSocket detection path.
- `detection.py`: image decode, inference, and bounding-box post-processing.
- `models.py`: versioned model cache and leases for active inference.
- `model_files.py`: streaming SHA-256 checksums for model delivery. Model file paths come from the catalogue.
- `config.py`: runtime settings, including device selection.
- `schemas.py`: request and response models.

## Run

From the repository root:

```bash
uvicorn examples.YOLO_server_api.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1
```

Use one worker per GPU model instance unless you intentionally want duplicate
model loads. Multiple Uvicorn workers do not share a loaded PyTorch model.

## Configuration

Deployment example (HTTP detection endpoints enforce authentication directly):

```dotenv
# DETECT_API_AUTH_REQUIRED=true (obsolete)
DETECT_API_URL=http://127.0.0.1:8000
```

The model ID selects a published catalogue entry. The service verifies the
artefact against the SHA-256 recorded for that version before loading it.

## Authentication

HTTP routes and authenticated WebSockets share the `jwt_access` verification
core from `examples.auth.jwt_config`, including OIDC, deployment and revocation
checks used by notification and violation APIs. With Keycloak enabled, pass
the access token obtained at login:

```text
Authorization: Bearer <access-token>
```

The `/detect` rate limiter only enforces quotas; it no longer requires the
legacy Redis login `jti_list`. WebSockets never auto-register token IDs.
The public WSS origin maps to
HTTPS for deployment lookup, so proxies must preserve the external host and
scheme. A separate WSS domain needs a corresponding deployment configuration.
The existing same-host pipeline option `YOLO_WS_ALLOW_LOCALHOST_BYPASS` remains
available; set it to `false` to require authentication on all connections.

The former upload endpoint checks for the `admin` or `model_manage` role before returning the immutable-catalogue response.

## Performance Notes

- For the main multi-camera pipeline, prefer local YOLO workers in `src/`.
- This API path encodes or decodes images at the process boundary, which costs
  CPU time and memory bandwidth.
- WebSocket mode is useful for compatibility tests, but it is not the fastest
  local path.
