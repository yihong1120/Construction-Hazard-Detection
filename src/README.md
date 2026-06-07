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

- `stream_processor.py`: one-camera runtime loop. It owns capture pacing,
  detection calls, warning handling, overlay publishing, metadata cleanup, and
  shutdown.
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
- `net/net_client.py`: authenticated HTTP and WebSocket client helpers.
- `stream_viewer.py`: manual OpenCV stream viewer for diagnostics.

## Notification Adapters

`notifiers/` contains FCM, LINE Message API, Messenger, Telegram, WeChat, and
broadcast adapters. Keep adapters thin: they should format and send messages,
while detection and violation decisions stay in `stream_processor.py`.

## Runtime Notes

- Redis is not a video frame store. It is used for compact metadata, auth cache,
  notification token cache, and overlay coordination.
- MediaMTX owns live HLS/WebRTC playback.
- YOLO worker queue size limits outstanding inference requests. When the queue
  is full, the caller times out or skips stale work rather than growing memory
  without bound.
- Keep frame copies near capture, shared memory, and ffmpeg boundaries only.
