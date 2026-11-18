🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Construction Hazard Detection

<img width="100%" src="./assets/images/project_graphics/banner.gif" alt="AI 工地安全監測橫幅">

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

[獨立 Rust 攝影機流程](native/pipeline/README.md)提供影像擷取、TensorRT 推論、危險區域分析、多語標註，以及 PostgreSQL 事件儲存與通知派送。指南包含建置、執行與驗證方式。

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

## main.py 影像處理工作流程

`main.py` 讀取 PostgreSQL 或 JSON 設定，管理攝影機程序與共用 YOLO workers，
並啟動 outbox dispatcher。下圖由左至右呈現這些程序如何將影像串流轉換成
物件辨識結果、安全警示、違規證據與推播通知。

```mermaid
flowchart LR
    MAIN["main.py<br/>載入設定與管理程序"]
    CAPTURE["影像擷取<br/>RTSP／HTTP／本機影片"]
    MAIN -. 啟動與管理 .-> CAPTURE

    subgraph DETECTION["物件辨識與危害判斷"]
        YOLO["YOLO workers<br/>GPU 物件辨識"]
        RULES["危害判斷<br/>防護裝備、距離與管制區規則"]
        YOLO -->|偵測結果| RULES
    end
    CAPTURE -->|最新影格| YOLO

    subgraph LIVE["直播與即時警示"]
        MEDIA["影像發布／MediaMTX<br/>原始與標註影像"]
        REDIS[("Redis<br/>警示 metadata")]
        VIEW["Web／Flutter App<br/>觀看直播與警示"]
        MEDIA -->|HLS／WebRTC| VIEW
        REDIS -->|Streaming Web API<br/>SSE／WebSocket| VIEW
    end
    CAPTURE -->|原始影像| MEDIA
    RULES -->|標註影像| MEDIA
    RULES -->|警示狀態| REDIS

    subgraph EVIDENCE["違規儲存與通知"]
        OUTBOX[("SQLite outbox<br/>證據排隊與投遞重試")]
        STORE["違規紀錄 API<br/>PostgreSQL 紀錄＋圖片檔案"]
        NOTIFY["FCM 通知<br/>推播至工地成員裝置"]
        OUTBOX -->|Outbox dispatcher| STORE
        STORE -->|上傳成功後| NOTIFY
    end
    RULES -->|符合條件的違規事件| OUTBOX

    classDef control fill:#eef2ff,stroke:#6366f1,color:#182b45
    classDef video fill:#eaf1ff,stroke:#2563eb,color:#182b45
    classDef detection fill:#e4f4f3,stroke:#087f8c,color:#182b45
    classDef storage fill:#fff2df,stroke:#b45309,color:#182b45
    class MAIN control
    class CAPTURE,MEDIA,VIEW video
    class YOLO,RULES,REDIS detection
    class OUTBOX,STORE,NOTIFY storage
```

- **影像串流：**擷取、推論與發布各自運作。原始影像可直接發布，標註影像則加入
  偵測框、標籤與危害區域多邊形。發布器依觀看需求啟動，MediaMTX 負責分發影像。
- **即時警示：**Redis 保存小型警示 metadata，由 Streaming Web 透過 SSE 或
  WebSocket 傳送給觀看端。Redis 也保存登入／token 快取、觀看需求及 overlay
  ready 狀態，不傳送直播影格。
- **儲存與通知：**違規事件受工作時段與通知間隔限制。outbox dispatcher 先將
  證據上傳至違規紀錄 API，取得違規 ID 後再呼叫 FCM 通知服務。PostgreSQL 保存
  紀錄，圖片另外存成檔案。即時警示 metadata 也受工作時段限制；已啟用且未過期的
  串流仍可持續擷取與播放影像。

## 目錄說明

- `main.py`：監督 stream 設定與 worker process。
- `src/`：正式執行用的核心模組。
- `examples/bff/`：Web session、CSRF、media capability 與 API gateway。
- `examples/db_management/`：使用者、群組、工地、stream 設定 API。
- `examples/local_notification_server/`：FCM token 與工地通知 API。
- `examples/streaming_web/`：標籤、播放 URL、metadata channel、
  media session 授權與 WebRTC ICE 設定。
- `examples/violation_records/`：違規紀錄與圖片 API。
- `examples/YOLO_server_api/`：可選的獨立 YOLO API。
- `examples/mcp_server/`：AI 偵測、違規、模型與多平台通知工具入口；[設定說明](examples/mcp_server/README-zh-tw.md)。
- `scripts/`：資料庫初始化與 TensorRT 重建工具。

通知支援 FCM、LINE Messaging API、Messenger、Telegram、WeChat 與 HTTP broadcast。
模型開發工具保留於 repo，按需要安裝其選用依賴。

## Flutter 前端

Flutter 前端原始碼維護於獨立的
[visionnaire-flutter](https://github.com/yihong1120/visionnaire-flutter)。
本 repository 的 `frontend/web/` 是忽略的建置部署產物，並非可直接編輯的前端
原始碼。版本發布與前後端契約邊界請參閱
[frontend/README.md](frontend/README.md)。

## 建議執行方式

除非只是短暫測試本機 JSON 設定，否則建議使用 database mode。

1. PostgreSQL 儲存工地、使用者、stream 設定與違規紀錄。
2. Redis 儲存 auth/session cache、通知 token cache 與小型直播協調 key。
3. MediaMTX 負責 RTSP ingest 與 HLS/WebRTC 播放。
4. `main.py` 輪詢 stream 設定，並為每支啟用的攝影機啟動一個 stream process。
5. 本機 YOLO workers 透過 shared memory 在多攝影機間共用 GPU 推論能力。

獨立 YOLO server 仍可用於 API 測試或獨立部署，但目前多鏡頭高吞吐的建議路徑是
`src/yolo_worker.py`。

## Rust 後處理

串流程序已支援 Rust／PyO3 後端，處理 centroid 追蹤、PPE 及人員與機具／車輛
距離判斷。串流 Docker 映像內含 wheel；主機執行時可自行建置安裝。
`POSTPROCESS_BACKEND` 可選 `auto`（預設）、`python` 或 `rust`，變更後重啟程序。
建置方式、適用範圍與量測結果見 [Rust 後處理說明](native/postprocess/README.md)。

共用 YOLO worker 也已接入 Rust TensorRT 後端，可直接沿用通過 catalog／SHA256
驗證且版本相容的 INT8 engine。`YOLO_WORKER_BACKEND=auto|rust|python` 控制
推論後端，保留既有多鏡頭 shared memory、批次與獨立回覆 queue。
32 路模擬來源的整合測試、建置及版本條件見 [Rust GPU worker](native/inference/README.md)。

Rust worker 可直接讀取持有 lease 的共享影格，省去完整影格的 bytes 快照；
幾何分析套件也改為按需載入。詳見 [影格複製減量與量測](native/inference/README.md#共享影格快照減量)。

後續加入推論 workspace 重用、安全區域快取、Rust MST 選邊與標註混色，
並提供明確選用的 Rust／FFmpeg 擷取後端。詳見 [串流流程優化及驗證](native/postprocess/README.md)。

GPU 管理新增跨程序 NVENC／推論／記憶體配額，以及可選的 pinned-memory
非同步推論。1080p 實測 batch 1 約快 4%，batch 8 約慢 5%，因此非同步預設關閉。
NVDEC → GPU 前處理／標註 → TensorRT／NVENC 已有獨立實驗路徑，但尚未通過
既有色彩／偵測結果一致性門檻，未接入正式串流。
詳見 [GPU 資源、非同步推論與影格實驗](native/inference/README.md#跨程序-gpu-資源與非同步推論)。

## 快速開始

### 1. 準備 Python

目前專案鎖定 Python `>=3.15,<3.16`，使用 PEP 810 原生延遲導入。
共用 API runtime 可直接安裝：

```bash
uv sync --locked
```

Python 3.15 的完整訓練與 INT8 TensorRT 匯出，請在目標環境執行：

```bash
conda activate construction_env
python deploy/python315/bootstrap.py
```

此流程使用 PyTorch 官方 CUDA 13.0 的 CPython 3.15 wheels，並從固定的上游
revision 建置 ONNX Runtime 1.31.0 與 TensorRT 11.4.0.106 Python bindings。
需要 Linux x86_64、標準 CPython 3.15（啟用 GIL）、`uv`、`git`、C++20 編譯器
與 CUDA 開發標頭（預設 `/usr/local/cuda/include`，可用 `CUDA_HOME` 指定）。
本地 wheel 存於 `wheels/python315/`，首次建置應預留數十 GB 空間；
`--jobs 4` 可降低編譯記憶體需求，`--build-only` 只建置而不安裝。
這些原生 wheel 是本地建置產物，尚非 Microsoft／NVIDIA 官方 Python 3.15 支援。
在新主機應重新執行建置流程，不要只執行 `pip install -r requirements.txt`。

不需要 TensorRT／CUDA 標頭的 CPU 校正、開發與測試環境，改執行
`python deploy/python315/bootstrap.py --cpu-only`。bootstrap 會另外安裝固定版本的
ONNX Runtime／TensorRT 原生建置產物；只使用套件庫的 `uv.lock` 與 `uv sync`
不包含這些原生套件。CI 使用 CPU 流程，GPU Docker base 則在映像內建置 wheel。

Python 3.15 的 `examples/YOLO_train/export_int8_trt.py` 與 `export_int8_engine.py` 使用 ONNX Runtime
CPU 校正產生 INT8 Q/DQ 圖，再交由 TensorRT 建置及執行 GPU engine；
支援校正圖片選取、全部圖片校正與節點排除。此路徑取代尚未支援 3.15 的 ModelOpt，
部署前須以實際驗證集重新評估量化模型準確率。

預設只安裝共用 API runtime 相依。Docker image 會各自安裝 `streaming`、
`violation`、`notification` 或 `yolo` extra；只有 CUDA/TensorRT 主機才需要
額外加入 `--extra yolo-gpu`。


多平台通知工具使用 `uv sync --extra social-notifications`；搭配 MCP 時使用
`uv sync --extra mcp --extra social-notifications`。訓練、評估與資料增強使用
`uv sync --extra training` 安裝套件庫內的訓練相依；原生校正／匯出相依使用上述 bootstrap。

選用的推論與通知相依，統一以模組頂層 `lazy import` 宣告。
路由與資料模型註冊維持即時匯入，選用套件的錯誤則在首次使用處處理。
部署腳本若必須先設定環境變數或模組搜尋路徑，仍保留該初始化順序。

### 2. 下載模型

```bash
hf download yihong1120/Construction-Hazard-Detection \
  --repo-type model \
  --include "models/pt/*.pt" \
  --local-dir .
```

worker 模型命名規則為 `models/pt/best_<model_key>.pt`，例如
`models/pt/best_yolo26n.pt`。

### 3. 建立 `.env`

可從 `.env.example` 開始調整。重要設定如下：

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
# 依 model 尺寸覆寫；未列出的 model key 使用上方全域預設值。
YOLO_WORKER_CAMERAS_PER_ENGINE_BY_MODEL=yolo26n=8,yolo26s=6,yolo26m=4,yolo26l=2,yolo26x=1
YOLO_WORKER_DEVICES=cuda:0,cuda:0
YOLO_WORKER_QUEUE_SIZE=64
YOLO_WORKER_RESULT_QUEUE_SIZE=8
YOLO_WORKER_RING_SLOTS=2
YOLO_WORKER_RING_SLOT_CLEANUP_SECONDS=120
YOLO_WORKER_BATCH_SIZE=8
YOLO_WORKER_BATCH_WAIT_MS=10
YOLO_WORKER_PRECISION=f16
# f32/f16 使用 models/pt/*.pt；int8 使用 models/int8_engine/*.engine。

# 每台攝影機使用固定且有上限的 shared-memory frame ring 與原生 result
# queue；Docker 的 YOLO_WORKER_SHM_SIZE 至少應涵蓋
# camera_count * YOLO_WORKER_RING_SLOTS * maximum_frame_bytes。

MEDIA_PUBLISH_RTSP_BASE_URL='rtsp://127.0.0.1:8554'
MEDIA_PUBLIC_HLS_BASE_URL='/hazard/media'
MEDIA_PUBLIC_WEBRTC_BASE_URL='/hazard/media/webrtc'
MEDIA_PUBLISH_CLEAN_STREAM=true
MEDIA_PUBLISH_ANNOTATED_STREAM=true
```

各 API service 共用同一組 OIDC issuer、JWKS URL 與 API audience 設定。

### 4. 啟動基礎服務

正常 database-backed runtime 需要 Redis、PostgreSQL 與 MediaMTX：

```bash
docker compose up -d redis postgres media-server
```

確認 containers 已啟動：

```bash
docker compose ps redis postgres media-server
docker compose logs -f redis postgres media-server
```

若 PostgreSQL 與 Redis 是由 host 管理，而你只需要 MediaMTX，請使用不會解析
database 或 Redis 環境變數的獨立 Compose 設定：

```bash
docker compose -f docker-compose.media.yml up -d
docker compose -f docker-compose.media.yml ps
docker compose -f docker-compose.media.yml logs -f media-server
```

這個設定刻意不讀取 `DATABASE_URL`、`POSTGRES_*` 或 `REDIS_PASSWORD`。原本的
`docker-compose.yml` 會啟動內建 PostgreSQL container，因此必須分別提供
`POSTGRES_DB`、`POSTGRES_USER` 和 `POSTGRES_PASSWORD`；它無法從 URL 安全且可靠地
反解析這些值。

如果 Python services 在 host 上執行，`.env` 使用本機位址：

```dotenv
DATABASE_URL='postgresql+asyncpg://username:password@127.0.0.1/construction_hazard_detection'
REDIS_HOST='127.0.0.1'
REDIS_PORT=6379
REDIS_PASSWORD='password'
MEDIA_PUBLISH_RTSP_BASE_URL='rtsp://127.0.0.1:8554'
```

如果 `main.py` 或 FastAPI services 在 Docker Compose 內執行，使用 service names：

```dotenv
DATABASE_URL='postgresql+asyncpg://username:password@postgres/construction_hazard_detection'
REDIS_HOST='redis'
POSTGRES_DB='construction_hazard_detection'
POSTGRES_USER='construction_app'
POSTGRES_PASSWORD='set-a-strong-password'
REDIS_PASSWORD='set-a-strong-password'
MEDIA_PUBLISH_RTSP_BASE_URL='rtsp://media-server:8554'
```

基礎服務分工：

- PostgreSQL 儲存 users、sites、stream settings 與 violation records。
- Redis 儲存 authentication cache、token cache、小型 live metadata 與 overlay
  demand keys。
- MediaMTX 接收 `main.py` 發布的 RTSP streams，並提供 HLS/WebRTC 播放。

Redis 與 MediaMTX 都不負責保存違規圖片或資料庫紀錄。

若是全新且沒有資料的 PostgreSQL database，第一次啟動時沒有掛載初始化 schema，
可手動匯入。此腳本會刪除 application tables，絕不可對已有資料的 database 執行：

```bash
cat ./scripts/init.postgres.sql | docker exec -i postgres-container \
  psql -U username -d construction_hazard_detection
```

既有 database 請依組織的資料庫變更管理流程處理。migration 與特權 operator 工具
刻意不隨 application source 發布。

### 5. 設定 MediaMTX

MediaMTX 是直播 media server。`main.py` 會透過 RTSP 將處理後的 H.264 stream
發布到 MediaMTX，觀看端再透過 HLS 或 WebRTC 播放。Redis 只保存小型 metadata 與
demand keys，不傳送 video frames。

Docker Compose service 預設只綁定本機 ports：

```text
127.0.0.1:8554  RTSP ingest，給 main.py / ffmpeg 發布影像
127.0.0.1:8890  HLS playback，對應 MediaMTX port 8888
127.0.0.1:8889  WebRTC/WHEP playback
```

如果 `main.py` 在 host 上執行，使用：

```dotenv
MEDIA_PUBLISH_RTSP_BASE_URL='rtsp://127.0.0.1:8554'
MEDIA_PUBLIC_HLS_BASE_URL='/hazard/media'
MEDIA_PUBLIC_WEBRTC_BASE_URL='/hazard/media/webrtc'
```

如果 `main.py` 在 Docker 內執行，使用 Compose service name：

```dotenv
MEDIA_PUBLISH_RTSP_BASE_URL='rtsp://media-server:8554'
```

streaming backend 回傳的播放 URL 形狀如下：

```text
/hazard/media/<media-path>/index.m3u8
/hazard/media/webrtc/<media-path>/whep
```

公開部署時，建議透過 Nginx proxy MediaMTX，並交給 `examples.streaming_web`
做 media authorisation。可從
`examples/streaming_web/nginx.hazard-media.conf` 開始調整：

```text
Nginx /hazard/media/*        -> MediaMTX HLS port 8888
Nginx /hazard/media/webrtc/* -> MediaMTX WebRTC port 8889
Nginx /hazard/api/media-auth -> examples.streaming_web /media-auth
```

建議的 live-buffer 預設值已放在 `docker-compose.yml`：

```dotenv
MTX_HLSSEGMENTDURATION=2s
MTX_HLSSEGMENTCOUNT=14
MTX_HLSALWAYSREMUX=yes
MTX_HLSMUXERCLOSEAFTER=60s
```

降低 `MTX_HLSSEGMENTCOUNT` 可減少硬碟與記憶體使用量。預設 14 個片段（每段 2 秒時約
28 秒）可讓多鏡頭牆在短暫網路或發布中斷後恢復。

### 6. 啟動 API

請從 repo 根目錄分別啟動：

```bash
uvicorn examples.db_management.app:app --host 127.0.0.1 --port 8005 --workers 2
uvicorn examples.local_notification_server.app:app --host 127.0.0.1 --port 8003 --workers 2
uvicorn examples.violation_records.app:app --host 127.0.0.1 --port 8002 --workers 2
uvicorn examples.streaming_web.app:app --host 127.0.0.1 --port 8800 --workers 2
```

可選的獨立 YOLO API：

```bash
uvicorn examples.YOLO_server_api.app:app --host 127.0.0.1 --port 8000 --workers 2
```

### 7. 啟動影像處理

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

預設擷取每個解碼影格，原始、標註與預覽串流都沿用攝影機 FPS，包含 29.97 等小數
影格率。預覽只降低解析度與碼率；推論獨立取用最新影格，不以推論速度限制播放 FPS。
重新連線會重新讀取來源 FPS，必要時重建發布器。

`MEDIA_PUBLISH_SOURCE_FPS`、`MEDIA_PUBLISH_FPS`、`MEDIA_PREVIEW_FPS` 與
`MEDIA_PUBLISH_CLEAN_FPS` 預設為 `auto`，空值亦代表自動；填入數字才啟用指定的
擷取抽幀或輸出影格率。直接轉送來源時保留輸入時間戳。使用原始影格編碼的發布器在
啟動畫面或來源無法提供有效 FPS 時，暫以 30 FPS 發布。

同一路攝影機的發布器與來源轉送器共用啟動時載入的編碼設定，重新連線時仍保留這份
設定。Worker 派工與結果佇列使用資料庫的 `stream_id`，共用 RTSP 網址的攝影機
各自保有獨立識別。

批次播放會先驗證整批 session，再以同一個 Redis 交易提交 session、需求租約及
反向索引；既有 session 不存在或不屬於目前使用者時，整批維持原狀。

模型發布統一使用 `python -m scripts.sync_huggingface_models`。流程會驗證新舊
artifact、重啟受影響的 worker，並以新的 challenge 確認實際載入版本；失敗時還原
並驗證前版。INT8 manifest 必須保留可用的前版 engine。詳見
[`MODEL_CATALOG.md`](examples/YOLO_server_api/MODEL_CATALOG.md)。

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

本專案使用 [AGPL-3.0 Licence](LICENSE)。

YOLO shard 異常重啟會建立新的 IPC generation，更換 request／result queues，
並由設定協調流程重新綁定受影響的攝影機程序；正常 shard 保留既有資源。
模型載入使用程序死亡時由作業系統釋放的檔案鎖，避免沿用已損壞的程序鎖。
來源、發布及預覽 FPS 維持既有設定。

環境設定已依領域集中為具型別模型，client 與 worker 在自己的生命週期持有已驗證
設定；明確建構參數優先於環境變數，包含可覆寫 Redis 環境密碼的明確空字串。
錯誤的數值、布林與 MCP transport 在載入時報錯。設定生命週期、憑證更新與
驗證規則見[設定管理說明](src/settings/README.md)，串流 FPS 選擇規則維持不變。
