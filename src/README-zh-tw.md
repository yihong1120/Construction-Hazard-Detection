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

- `stream_processor.py`：單一攝影機的執行 loop，負責擷取節奏、偵測呼叫、警示處理、
  overlay 發布、metadata 清理與關閉流程。
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
- `net/net_client.py`：具驗證能力的 HTTP 與 WebSocket client helper。
- `stream_viewer.py`：手動 OpenCV stream viewer，用於診斷。

## 通知 Adapter

`notifiers/` 包含 FCM、LINE Message API、Messenger、Telegram、WeChat 與 broadcast
adapter。adapter 應保持薄層，只負責格式化與發送訊息；偵測與違規判斷應留在
`stream_processor.py`。

## 執行注意事項

- Redis 不是 video frame store，只用於 compact metadata、auth cache、notification
  token cache 與 overlay coordination。
- MediaMTX 負責 live HLS/WebRTC playback。
- YOLO worker queue size 會限制尚未完成的 inference request 數量。queue 滿時，caller
  會 timeout 或略過過舊工作，避免記憶體無限制成長。
- frame copy 應盡量只出現在 capture、shared memory 與 ffmpeg 邊界。
