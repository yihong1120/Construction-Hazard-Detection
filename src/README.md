# Source Modules

[English](./README.md) | [繁體中文](./README-zh-tw.md)

`src/` contains the production runtime used by `main.py`. The modules here
capture frames, run YOLO inference, derive warnings, publish live media,
upload violation records, and send notifications.

## Main Flow

```text
stream_capture.py
    -> yolo_detector.py / yolo_worker.py
    -> danger_detector.py
    -> stream_processor.py
    -> media_stream_publisher.py / media_restreamer.py
    -> violation_sender.py
    -> notifiers/*
```

## Key Modules

- `stream_processor.py`: one-camera task orchestration, demand-driven media
  publishing, metadata cleanup, and shutdown.
- `stream_detection.py`: capture/inference loops, warning events and alerts.
- `stream_overlay_frames.py`: immutable frames and per-language overlay caches.
- `stream_capture.py`: reads frames from RTSP, HTTP, or local files. It avoids
  unbounded buffering and keeps reconnect logic close to the capture source.
- `yolo_worker.py`: multiprocessing worker and client. Frames are copied into
  POSIX shared memory; queue messages contain metadata only. Workers load YOLO
  models once and batch requests across cameras.
- `yolo_detector.py`: detection facade used by stream processors. It can call
  the local worker path or the optional remote detector API.
- `danger_detector.py`: converts detections into construction-safety warnings
  and controlled-area polygons.
- `media_stream_publisher.py`: publishes clean or annotated frames to MediaMTX
  through ffmpeg.
- `media_restreamer.py`: publishes the original source stream to MediaMTX
  without waiting for detection.
- `violation_sender.py`: uploads violation images and metadata to the violation
  records API.

## Utilities

- `utils.py`: token handling, Redis helpers, geometry, encoding, and shared
  helpers.
- `warning_types.py`: warning payload type aliases.
- `model_fetcher.py`: model download/update helpers.
- `monitor_logger.py`: logging setup.
- `stream_viewer.py`: manual OpenCV stream viewer for diagnostics.

## Notification Adapters

`notifiers/` supports FCM, LINE Messaging API, Messenger, Telegram, WeChat and
HTTP broadcast. FCM uses the OIDC-authenticated backend; other platforms use
their own credentials. Detection and violation decisions stay in `stream_detection.py`.

Install `uv sync --extra social-notifications` for optional adapters. Call them
directly or through MCP notification tools; the existing camera worker uses FCM.
See the [LINE Messaging API guide](../docs/en/line_notify_guide_en.md).

## Runtime Notes

- Redis is not a video frame store. It is used for compact metadata, auth cache,
  notification token cache, and overlay coordination.
- MediaMTX owns live HLS/WebRTC playback.
- YOLO worker queue size limits outstanding inference requests. When the queue
  is full, the caller times out or skips stale work rather than growing memory
  without bound.
- Keep frame copies near capture, shared memory, and ffmpeg boundaries only.

## Background worker authentication

`main.py` checks Redis before starting camera and GPU processes. Set
`REDIS_PASSWORD` to the server password. Authentication occurs when a connection
is established; reads and writes reuse the connection.

Workers use `WORKER_OIDC_CLIENT_ID` and `WORKER_OIDC_CLIENT_SECRET`.
Set `WORKER_OIDC_TOKEN_ENDPOINT` to the realm token URL, or provide
`OIDC_ISSUER_URL`. HTTPS is required except for loopback development URLs.
Tokens are acquired and renewed with the client-credentials grant.

Provision a separate confidential Keycloak client for each worker tenant with
client authentication and service accounts enabled, browser/direct password
flows disabled, and an audience mapper including the configured API audience
(normally `visionnaire-api`). Do not reuse the BFF or user-linker client.

Link the service account's Keycloak user ID (the token `sub`) through the existing
`UserIdentity` table using the configured provider (normally `keycloak`) and a
dedicated active local worker user. Assign that local user its intended tenant,
site access, group and necessary features. Existing API checks remain authoritative;
an unlinked service account is rejected. No realm administration roles are needed.
The client secret alone does not grant local API permissions. Existing realms
must provision this client explicitly; restarting a realm import does not update
existing clients. Keep secrets in the local environment, not source control.

Token acquisition at startup validates the client configuration, but does not
prove API permissions: verify a request to the intended API after linking the
identity. Redis failures and missing worker credentials stop startup before
camera processes are spawned.
