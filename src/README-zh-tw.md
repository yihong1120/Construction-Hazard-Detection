🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Source Modules

`src/` 是 `main.py` 使用的正式執行模組。這裡負責擷取影像、執行 YOLO 推論、
產生安全警示、發布直播媒體、上傳違規紀錄，以及發送通知。

## 主流程

```text
stream_capture.py
    -> yolo_detector.py / yolo_worker.py
    -> danger_detector.py
    -> stream_processor.py
    -> media_stream_publisher.py / media_restreamer.py
    -> violation_sender.py
    -> notifiers/*
```

## 主要模組

- `stream_processor.py`：攝影機 task 編排、按觀看需求發布媒體、metadata 清理與關閉流程。
- `stream_detection.py`：影像擷取／推論 loop、警示事件與通知。
- `stream_overlay_frames.py`：不可變 frame 與各語言 overlay 快取。
- `stream_capture.py`：讀取 RTSP、HTTP 或本機檔案影像，避免無限制 buffering，並將
  重連邏輯集中在影像來源附近。
- `yolo_worker.py`：multiprocessing worker 與 client。frame 會複製到 POSIX
  shared memory，queue 訊息只包含 metadata。worker 會載入一次 YOLO 模型，並跨攝影機
  batching。
- `yolo_detector.py`：stream processor 使用的偵測 facade，可呼叫本機 worker 或可選的
  遠端 detector API。
- `danger_detector.py`：將 detection 轉成工地安全警示與管制區 polygon。
- `media_stream_publisher.py`：透過 ffmpeg 將 clean 或 annotated frame 發布到
  MediaMTX。
- `media_restreamer.py`：不等待偵測，直接將原始來源 stream 轉發到 MediaMTX。
- `violation_sender.py`：上傳違規圖片與 metadata 到 violation records API。

## 工具模組

- `utils.py`：token、Redis、幾何、編碼與共用 helper。
- `warning_types.py`：警示 payload type aliases。
- `model_fetcher.py`：模型下載與更新 helper。
- `monitor_logger.py`：logging 設定。
- `stream_viewer.py`：手動 OpenCV stream viewer，用於診斷。

## 通知 Adapter

`notifiers/` 保留 FCM、LINE Messaging API、Messenger、Telegram、WeChat 與
HTTP broadcast adapter。FCM 使用具 OIDC 驗證機制的 backend，其他平台使用各自憑證；
偵測與違規判斷留在 `stream_detection.py`。

多平台通知安裝 `uv sync --extra social-notifications`。可直接呼叫 adapter，
或透過 MCP 的通知工具使用；目前攝影機 worker 的既有通知流程使用 FCM。
LINE 的設定請參閱[Messaging API 指南](../docs/zh/line_notify_guide_zh.md)。

## 執行注意事項

- Redis 不是 video frame store，只用於 compact metadata、auth cache、notification
  token cache 與 overlay coordination。
- MediaMTX 負責 live HLS/WebRTC playback。
- YOLO worker queue size 會限制尚未完成的 inference request 數量。queue 滿時，caller
  會 timeout 或略過過舊工作，避免記憶體無限制成長。
- frame copy 應盡量只出現在 capture、shared memory 與 ffmpeg 邊界。
