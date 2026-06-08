🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Construction Hazard Detection

<img width="100%" src="./assets/images/project_graphics/banner.gif" alt="AI 工地安全監測橫幅">

<div align="center">
   <a href="examples/YOLO_server_api">Server API</a> |
   <a href="examples/mcp_server">MCP Server</a> |
   <a href="examples/local_notification_server">FCM Notification Server</a> |
   <a href="examples/violation_records">Violation Records Server</a> |
   <a href="examples/db_management">Data Management Server</a> |
   <a href="examples/streaming_web">Streaming Web</a> |
   <a href="examples/YOLO_data_augmentation">Data Augmentation</a> |
   <a href="examples/YOLO_evaluation">Evaluation</a> |
   <a href="examples/YOLO_train">Train</a>
</div>

<br>

<div align="center">
   <a href="https://www.python.org/downloads/">
      <img src="https://img.shields.io/badge/python-3.12-blue?logo=python" alt="Python 3.12">
   </a>
   <a href="https://github.com/ultralytics/ultralytics">
      <img src="https://img.shields.io/badge/ultralytics-8.4.8-blue?logo=yolo" alt="Ultralytics 8.4.8">
   </a>
   <a href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html">
      <img src="https://img.shields.io/badge/HDBSCAN-sklearn-orange?logo=scikit-learn" alt="HDBSCAN sklearn">
   </a>
   <a href="https://fastapi.tiangolo.com/">
      <img src="https://img.shields.io/badge/FastAPI-0.128.0-blue?logo=fastapi" alt="FastAPI 0.128.0">
   </a>
   <a href="https://pypi.org/project/fastmcp/2.14.4/">
      <img src="https://img.shields.io/badge/FastMCP-2.14.4-purple" alt="FastMCP 2.14.4">
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

這是一套用於工地即時影像的 AI 安全監測系統。目前主要執行路徑以
`main.py`、`src/stream_processor.py`、本機 YOLO worker process、
MediaMTX 直播發布、PostgreSQL 紀錄、Redis 小狀態協調，以及多個
FastAPI 服務組成。

## 可偵測項目

- 工人未戴安全帽。
- 工人未穿反光背心。
- 工人太靠近機具或車輛。
- 工人進入由交通錐推估出的管制區。
- 機具或車輛太靠近電線桿。

標籤與通知支援繁體中文、簡體中文、英文、法文、泰文、越南文、印尼文與日文。

<img width="100%" src="./assets/images/hazard-detection.png" alt="工地危害偵測範例">

## 危害偵測範例

以下是系統即時偵測工地危害的範例。

<div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
  <div style="text-align: center; flex-basis: 33%;">
    <img src="./assets/images/demo/person_did_not_wear_safety_vest.png" alt="未戴安全帽或未穿反光背心的工人" style="width: 300px; height: 200px; object-fit: cover;">
    <p>未戴安全帽或未穿反光背心的工人</p>
  </div>
  <div style="text-align: center; flex-basis: 33%;">
    <img src="./assets/images/demo/person_near_machinery.jpg" alt="工人太靠近機具或車輛" style="width: 300px; height: 200px; object-fit: cover;">
    <p>工人太靠近機具或車輛</p>
  </div>
  <div style="text-align: center; flex-basis: 33%;">
    <img src="./assets/images/demo/persons_in_restricted_zones.jpg" alt="工人進入管制區" style="width: 300px; height: 200px; object-fit: cover;">
    <p>工人進入管制區</p>
  </div>
</div>

## 目前架構

<img width="100%" src="./assets/flowcharts/site_safety_monitor_zh.png" alt="工地危害偵測執行架構">

```text
PostgreSQL stream_configs
        |
        v
main.py
        |
        v
src/stream_processor.py
        |
        +--> src/stream_capture.py 讀取 RTSP/HTTP/檔案影像
        |
        +--> src/yolo_worker.py 本機 worker process
        |       - frame 透過 POSIX shared memory 傳遞
        |       - queue 只放 metadata
        |       - 每個 worker process 只載入一次模型
        |
        +--> src/danger_detector.py 產生安全警示
        |
        +--> src/media_stream_publisher.py 發布 H.264 到 MediaMTX
        |
        +--> src/violation_sender.py 上傳違規紀錄
        |
        +--> notifiers 發送 FCM / LINE / Telegram / broadcast 通知

Streaming Web Backend
        |
        +--> 回傳 MediaMTX HLS/WebRTC 播放 URL
        +--> 透過 SSE/WebSocket 傳送小型警示 metadata
        +--> Redis 只用於 auth cache、metadata、overlay demand keys
```

Redis 不儲存直播影像 frame。直播影像由 MediaMTX 負責，Redis 只保存登入快取、
FCM token 快取、即時警示 metadata、overlay demand key 與 overlay ready key
等小型狀態。

## 目錄說明

- `main.py`：監督 stream 設定與 worker process。
- `src/`：正式執行用的核心模組。
- `examples/db_management/`：使用者、群組、工地、stream 設定 API。
- `examples/local_notification_server/`：FCM token 與工地通知 API。
- `examples/streaming_web/`：標籤、播放 URL、metadata channel、
  media session 授權與 WebRTC ICE 設定。
- `examples/violation_records/`：違規紀錄與圖片 API。
- `examples/YOLO_server_api/backend/`：可選的獨立 YOLO API。
- `examples/mcp_server/`：提供給 agent 使用的 FastMCP tools。
- `examples/YOLO_train/`、`examples/YOLO_evaluation/`、
  `examples/YOLO_data_augmentation/`：模型開發工具。
- `scripts/`：資料庫初始化與 TensorRT 重建工具。

## 建議執行方式

除非只是短暫測試本機 JSON 設定，否則建議使用 database mode。

1. PostgreSQL 儲存工地、使用者、stream 設定與違規紀錄。
2. Redis 儲存 auth/session cache、通知 token cache 與小型直播協調 key。
3. MediaMTX 負責 RTSP ingest 與 HLS/WebRTC 播放。
4. `main.py` 輪詢 stream 設定，並為每支啟用的攝影機啟動一個 stream process。
5. 本機 YOLO workers 透過 shared memory 在多攝影機間共用 GPU 推論能力。

獨立 YOLO server 仍可用於 API 測試或獨立部署，但目前多鏡頭高吞吐的建議路徑是
`src/yolo_worker.py`。

## 快速開始

### 1. 準備 Python

目前專案鎖定 Python `>=3.12,<3.13`。

```bash
uv sync
```

若使用 pip：

```bash
uv export --format=requirements-txt --no-dev -o requirements.lock
pip install -r requirements.lock
```

### 2. 啟動基礎服務

```bash
docker compose up -d redis postgres media-server
```

若 PostgreSQL container 第一次啟動時沒有掛載初始化 schema，可手動匯入：

```bash
cat ./scripts/init.postgres.sql | docker exec -i postgres-container \
  psql -U username -d construction_hazard_detection
```

### 3. 下載模型

```bash
hf download yihong1120/Construction-Hazard-Detection \
  --repo-type model \
  --include "models/pt/*.pt" \
  --local-dir .
```

worker 模型命名規則為 `models/pt/best_<model_key>.pt`，例如
`models/pt/best_yolo26n.pt`。

### 4. 建立 `.env`

可從 `.env.example` 開始調整。重要設定如下：

```dotenv
DATABASE_URL='postgresql+asyncpg://username:password@127.0.0.1/construction_hazard_detection'

REDIS_HOST='127.0.0.1'
REDIS_PORT=6379
REDIS_PASSWORD='password'

JWT_SECRET_KEY='replace-with-a-long-random-secret'

DB_MANAGEMENT_API_URL='http://127.0.0.1:8005'
FCM_API_URL='http://127.0.0.1:8003'
VIOLATION_RECORD_API_URL='http://127.0.0.1:8002'
STREAMING_API_URL='http://127.0.0.1:8800'

YOLO_WORKER_ENABLED=true
YOLO_WORKER_COUNT=2
YOLO_WORKER_DEVICES=cuda:0,cuda:0
YOLO_WORKER_QUEUE_SIZE=64
YOLO_WORKER_BATCH_SIZE=8
YOLO_WORKER_BATCH_WAIT_MS=10
YOLO_WORKER_MODEL_DIR=models/pt

MEDIA_PUBLISH_RTSP_BASE_URL='rtsp://127.0.0.1:8554'
MEDIA_PUBLIC_HLS_BASE_URL='/hazard/media'
MEDIA_PUBLIC_WEBRTC_BASE_URL='/hazard/media/webrtc'
MEDIA_PUBLISH_CLEAN_STREAM=true
MEDIA_PUBLISH_ANNOTATED_STREAM=true
```

正式部署請使用穩定且高熵的 `JWT_SECRET_KEY`，不要提交真正的 secret。

### 5. 啟動 API

請從 repo 根目錄分別啟動：

```bash
uvicorn examples.db_management.app:app --host 127.0.0.1 --port 8005 --workers 2
uvicorn examples.local_notification_server.app:app --host 127.0.0.1 --port 8003 --workers 2
uvicorn examples.violation_records.app:app --host 127.0.0.1 --port 8002 --workers 2
uvicorn examples.streaming_web.app:app --host 127.0.0.1 --port 8800 --workers 2
```

可選的獨立 YOLO API：

```bash
uvicorn examples.YOLO_server_api.backend.app:app --host 127.0.0.1 --port 8000 --workers 2
```

### 6. 啟動影像處理

```bash
python main.py
```

可調整輪詢間隔：

```bash
python main.py --poll 5
```

開發時也可使用 JSON 設定：

```bash
python main.py --config config/configuration.json
```

同一批攝影機不要同時用 database mode 與 JSON mode 執行。

## 直播觀看

前端或 App 應呼叫 streaming web backend：

- `GET /labels`
- `GET /streams/{label}`
- `GET /metadata/stream-id/{label}/{stream_id}`：SSE metadata
- `WebSocket /ws/metadata-id/{label}/{stream_id}`：metadata WebSocket
- `GET /webrtc/ice-servers`：WebRTC STUN/TURN 設定

影像播放來自 MediaMTX HLS/WebRTC URL。警示 metadata 是小型狀態，通常只包含目前
警示狀態；偵測框、多邊形與文字標籤會被畫進後端發布的 annotated video stream。

公開部署時，MediaMTX 建議放在 Nginx `auth_request` 後面，並由
`examples.streaming_web` 驗證 JWT 與工地權限。

## 儲存與保留

違規圖片會存放在服務設定的 `static/` 位置。若暫不使用 archive、NAS 或外接碟，
實務上可先採用：

- 圖片與 DB 紀錄保留 18 個月；
- 刪除時圖片與對應 DB row 一起刪；
- 清理工作排在低流量時段；
- HLS segment 保留時間要短，因為 MediaMTX HLS 是直播緩衝，不是歸檔。

不要留下已刪圖片但 DB 還存在的紀錄，也不要留下沒有 DB 紀錄的孤兒圖片。

## 開發檢查

```bash
pre-commit run -a
python -m pytest -q --tb=short
```

mypy pre-commit hook 檢查 production code：`main.py`、`src`、`examples` 與
`scripts`。測試仍由 `pytest` 與 `flake8` 驗證。

## Dataset 與模型

- Hugging Face：<https://huggingface.co/yihong1120/Construction-Hazard-Detection>
- Roboflow：<https://universe.roboflow.com/side-projects/construction-hazard-detection>

標籤：

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

## 授權

本專案使用 [AGPL-3.0 Licence](LICENSE.md)。
