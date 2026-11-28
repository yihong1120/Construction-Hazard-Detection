# 工地危害辨識（Construction Hazard Detection）

<img width="100%" src="./assets/images/project_graphics/banner.gif" alt="AI 工地安全監測橫幅">

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
      <img src="https://app.roboflow.com/images/download-dataset-badge.svg" alt="從 Roboflow 下載資料集">
   </a>
   <a href="https://huggingface.co/yihong1120/Construction-Hazard-Detection">
      <img src="https://img.shields.io/badge/HuggingFace-Model%20Repo-yellow?logo=huggingface" alt="Hugging Face 模型儲存庫">
   </a>
</div>

<p align="center">
  <a href="./README.md">🇬🇧 English (UK)</a> |
  <a href="./README-zh-tw.md">🇹🇼 繁體中文（台灣）</a>
</p>

<div align="center">
   <a href="examples/YOLO_server_api/README-zh-tw.md">推論 API</a> |
   <a href="examples/local_notification_server/README-zh-tw.md">FCM 通知服務</a> |
   <a href="examples/violation_records/README-zh-tw.md">違規紀錄服務</a> |
   <a href="examples/bff/README-zh-tw.md">網頁 BFF</a> |
   <a href="https://github.com/yihong1120/visionnaire-flutter">Flutter App</a> |
   <a href="examples/db_management/README-zh-tw.md">資料管理服務</a> |
   <a href="examples/streaming_web/README-zh-tw.md">網頁串流服務</a> |
   <a href="examples/mcp_server/README-zh-tw.md">MCP 伺服器</a> |
   <a href="examples/YOLO_data_augmentation/README-zh-tw.md">資料增強</a> |
   <a href="examples/YOLO_evaluation/README-zh-tw.md">模型評估</a> |
   <a href="examples/YOLO_train/README-zh-tw.md">模型訓練</a>
</div>

這是一套用於工地即時影像的 AI 安全監測系統。下文介紹以 `main.py` 啟動的
Python 流程，包含串流處理模組 `src/stream_processor.py`、本機 YOLO 工作者、
MediaMTX 直播串流發布、PostgreSQL 紀錄、Redis 即時狀態協調，以及多個 FastAPI 服務。

[獨立 Rust 攝影機流程](native/pipeline/README-zh-tw.md)支援影像擷取、TensorRT 推論、
危害分析與多語標註，並透過 PostgreSQL 持久化佇列保存與派送事件。指南包含
`construction-pipeline` 的建置、執行與驗證步驟。

[獨立 Rust 發行預覽版](deploy/rust-runtime/README-zh-tw.md)提供攝影機流程與 Rust HTTP API 的建置、服務單元及隔離驗證方式。

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

`main.py` 讀取 PostgreSQL 或 JSON 設定，管理攝影機程序與共用 YOLO 工作者，
並啟動 outbox 派送器。下圖由左至右呈現這些程序如何將影像串流轉換成
物件辨識結果、安全警示、違規證據與推播通知。

```mermaid
flowchart LR
    MAIN["main.py<br/>載入設定與管理程序"]
    CAPTURE["影像擷取<br/>RTSP／HTTP／本機影片"]
    MAIN -. 啟動與管理 .-> CAPTURE

    subgraph DETECTION["物件辨識與危害判斷"]
        YOLO["YOLO 工作者<br/>GPU 物件辨識"]
        RULES["危害判斷<br/>防護裝備、距離與管制區規則"]
        YOLO -->|偵測結果| RULES
    end
    CAPTURE -->|最新影格| YOLO

    subgraph LIVE["直播與即時警示"]
        MEDIA["影像發布／MediaMTX<br/>原始與標註影像"]
        REDIS[("Redis<br/>警示中繼資料")]
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
        OUTBOX -->|Outbox 派送器| STORE
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

- **影像串流**的擷取、推論與發布各自運作。原始影像可直接發布，標註影像則加入
  偵測框、標籤與危害區域多邊形。發布器依觀看需求啟動，MediaMTX 負責分發影像。
- **即時警示**透過網頁串流服務的 SSE 或 WebSocket 傳送給觀看端。Redis 保存精簡的
  警示中繼資料、登入與權杖快取、觀看需求，以及標註串流的就緒狀態。
- **儲存與通知**依工作時段與通知間隔處理違規事件。Outbox 派送器先將
  證據上傳至違規紀錄 API，取得違規 ID 後再呼叫 FCM 通知服務。PostgreSQL 保存
  紀錄，圖片另外存成檔案。即時警示中繼資料也受工作時段限制。已啟用且未過期的
  串流仍可持續擷取與播放影像。

## 目錄說明

- `main.py` 管理串流設定與工作者程序。
- `src/` 包含正式執行用的核心模組。
- `native/pipeline/` 和 `native/api/` 提供獨立的 Rust 攝影機及 API 執行程式。
- `native/inference/`、`native/postprocess-core/` 和 `native/stream-contract/` 包含共用的推論、危害分析及中繼資料程式碼。
- `bindings/python/` 提供既有 Python 應用程式使用的相容 wheel。
- `examples/bff/` 管理網頁工作階段、CSRF 防護與播放權限，並提供 API 閘道。
- `examples/db_management/` 提供使用者、群組、工地及串流設定 API。
- `examples/local_notification_server/` 提供 FCM 權杖與工地通知 API。
- `examples/streaming_web/` 提供標籤、播放網址、中繼資料頻道、播放工作階段授權及 WebRTC ICE 設定。
- `examples/violation_records/` 提供違規紀錄與圖片 API。
- `examples/YOLO_server_api/` 提供可選用的獨立 YOLO API。
- `examples/mcp_server/` 提供可選用的 AI 偵測、違規紀錄、模型及多平台通知工具。[設定說明](examples/mcp_server/README-zh-tw.md)
- `examples/YOLO_train/`、`examples/YOLO_evaluation/` 和 `examples/YOLO_data_augmentation/` 提供模型訓練、匯出、評估與資料增強工具。
- `scripts/` 提供資料庫初始化與 TensorRT 重建工具。

通知支援 FCM、LINE Messaging API、Messenger、Telegram、WeChat 與 HTTP 廣播。
模型開發工具的相依套件可按需要安裝。

## Flutter 前端

Flutter 前端原始碼維護於獨立的
[visionnaire-flutter](https://github.com/yihong1120/visionnaire-flutter)。
本儲存庫的 `frontend/web/` 存放 Git 忽略的前端建置產物。版本發布與前後端介面請參閱
[前端交付說明](frontend/README-zh-tw.md)。

## 建議執行方式

一般部署請使用資料庫模式。本機短期測試可使用 JSON 設定。

1. PostgreSQL 儲存工地、使用者、串流設定與違規紀錄。
2. Redis 儲存驗證與工作階段快取、通知權杖快取，以及直播協調狀態。
3. MediaMTX 接收 RTSP 串流，並提供 HLS／WebRTC 播放。
4. `main.py` 輪詢串流設定，並為每支啟用的攝影機啟動一個串流程序。
5. 本機 YOLO 工作者透過共享記憶體，讓多支攝影機共用 GPU 推論資源。

獨立 YOLO 服務適用於 API 測試或單獨部署。多鏡頭高吞吐量處理建議使用
`src/yolo_worker.py` 的本機工作者模式。

## Rust 後處理

串流程序支援 Rust／PyO3 後端，用於中心點追蹤、防護裝備判斷，以及人員與機具或車輛
的距離判斷。串流 Docker 映像內含 wheel。在主機執行時可自行建置安裝。
`POSTPROCESS_BACKEND` 可選 `auto`（預設）、`python` 或 `rust`，變更後重啟程序。
建置方式、適用範圍與量測結果見 [Rust 後處理說明](bindings/python/postprocess/README-zh-tw.md)。

共用 YOLO 工作者也支援 Rust TensorRT 後端，可沿用通過模型目錄與 SHA256
驗證且版本相容的 INT8 引擎。`YOLO_WORKER_BACKEND=auto|rust|python` 控制
推論後端，支援多鏡頭共享記憶體、批次與獨立回覆佇列。
32 路模擬來源的整合測試、建置及版本條件見 [Rust GPU 工作者](native/inference/README-zh-tw.md)。

Rust 工作者可直接讀取持有租約的共享影格，省去整張影格的快照複製。幾何分析套件
也按需載入。詳見 [影格複製減量與量測](native/inference/README-zh-tw.md#共享影格快照減量)。

後續加入推論工作區重用、安全區域快取、Rust 最小生成樹（MST）選邊與標註混色，
並提供明確選用的 Rust／FFmpeg 擷取後端。詳見 [串流流程優化及驗證](bindings/python/postprocess/README-zh-tw.md)。

GPU 管理支援跨程序 NVENC、推論與記憶體配額，以及可選用的固定記憶體
非同步推論。1080p 實測批次大小為 1 時約快 4%，批次大小為 8 時約慢 5%，因此非同步預設關閉。
NVDEC → GPU 前處理／標註 → TensorRT／NVENC 已有獨立實驗路徑，但尚未通過
既有色彩／偵測結果一致性門檻，未接入正式串流。
詳見 [GPU 資源、非同步推論與影格實驗](native/inference/README-zh-tw.md#跨程序-gpu-資源與非同步推論)。

## 快速開始

### 1. 準備 Python

`main.py` 與 API 服務使用 Python `>=3.15,<3.16`，並採用 PEP 810 明確延遲匯入。
Python 相依套件與選用群組宣告在 [pyproject.toml](pyproject.toml)，Linux x86_64
的解析版本記錄在 [uv.lock](uv.lock)。從專案根目錄執行下列指令，將共用 API 安裝到專案的 `.venv`。

```bash
uv sync --locked
```

需要訓練與 INT8 TensorRT 匯出時，請用同一個環境的 Python 執行建置程式。

```bash
.venv/bin/python deploy/python315/bootstrap.py
```

此流程使用 PyTorch 官方 CUDA 13.0 的 CPython 3.15 wheel，並從鎖定的上游
版本建置 ONNX Runtime 1.31.0 與 TensorRT 11.4.0.106 的 Python 繫結。
需要 Linux x86_64、標準 CPython 3.15（啟用 GIL）、`uv`、`git`、C++20 編譯器
與 CUDA 開發標頭（預設 `/usr/local/cuda/include`，可用 `CUDA_HOME` 指定）。
本機建置的 wheel 存於 `wheels/python315/`，首次建置應預留數十 GB 空間。
`--jobs 4` 可降低編譯所需的記憶體，`--build-only` 只執行建置。
這些 Microsoft／NVIDIA 套件的 Python 3.15 版本由專案在本機建置。
請在新主機執行上述流程以取得這些原生套件。

不需要 TensorRT／CUDA 標頭的 CPU 校正、開發與測試環境，改執行
`.venv/bin/python deploy/python315/bootstrap.py --cpu-only`。建置指令會在鎖定的套件庫相依項目
之外安裝指定版本的原生套件。CI 使用 CPU 流程，GPU Docker 基底映像則在映像內建置 wheel。
建置後使用 `uv run --no-sync` 執行程式，以保留這些本機 wheel。

若其他工具需要 pip 格式的套件清單，可按需匯出。例如
`uv export --locked --no-emit-project --extra training -o /tmp/requirements.txt` 會產生包含訓練套件的清單。

Python 3.15 的 `examples/YOLO_train/export_int8_trt.py` 與 `export_int8_engine.py` 使用 ONNX Runtime
CPU 校正產生 INT8 Q/DQ 圖，再交由 TensorRT 建置及執行 GPU 引擎。此流程支援
選取校正圖片、使用全部圖片校正，或排除指定節點。ModelOpt 支援 Python 3.15 前，
請使用此校正流程。部署前請以實際驗證資料集重新評估量化模型準確率。

預設安裝共用 API 執行環境的相依套件。Docker 映像各自安裝 `streaming`、
`violation`、`notification` 或 `yolo` 選用套件群組。CUDA／TensorRT 主機可加上
`--extra yolo-gpu`。


多平台通知工具使用 `uv sync --locked --extra social-notifications`，搭配 MCP 時使用
`uv sync --locked --extra mcp --extra social-notifications`。訓練、評估與資料增強使用
`uv sync --locked --extra training` 安裝套件庫提供的訓練相依套件。原生校正與匯出所需套件
由上述建置流程提供。

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

工作者依串流的 `model_key` 從已發布的模型目錄查出權重檔與 SHA-256。
下載 `models/pt/best_yolo26n.pt` 等檔案後，仍須在模型目錄發布才能選用。

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
# 可依模型尺寸覆寫。未列出的模型使用上方全域預設值。
YOLO_WORKER_CAMERAS_PER_ENGINE_BY_MODEL=yolo26n=8,yolo26s=6,yolo26m=4,yolo26l=2,yolo26x=1
YOLO_WORKER_DEVICES=cuda:0,cuda:0
YOLO_WORKER_QUEUE_SIZE=64
YOLO_WORKER_RESULT_QUEUE_SIZE=8
YOLO_WORKER_RING_SLOTS=2
YOLO_WORKER_RING_SLOT_CLEANUP_SECONDS=120
YOLO_WORKER_BATCH_SIZE=8
YOLO_WORKER_BATCH_WAIT_MS=10
YOLO_WORKER_PRECISION=f16
# 使用 INT8 時，YOLO_WORKER_ARTIFACT_MANIFEST 將已發布的權重對應至引擎檔。

# 每台攝影機各有一個容量有限的共享記憶體影格環形緩衝區及專用結果佇列。
# Docker 的 YOLO_WORKER_SHM_SIZE 至少應涵蓋
# camera_count * YOLO_WORKER_RING_SLOTS * maximum_frame_bytes。

MEDIA_PUBLISH_RTSP_BASE_URL='rtsp://127.0.0.1:8554'
MEDIA_PUBLIC_HLS_BASE_URL='/hazard/media'
MEDIA_PUBLIC_WEBRTC_BASE_URL='/hazard/media/webrtc'
MEDIA_PUBLISH_CLEAN_STREAM=true
MEDIA_PUBLISH_ANNOTATED_STREAM=true
# 原始畫面使用共享的最新影格與 Intel iGPU 編碼，可略過過時影格、
# 共用攝影機 RTSP 連線，並提供規律的 HLS 關鍵影格。需要原始位元流時可啟用直接轉送。
MEDIA_PUBLISH_CLEAN_SOURCE_RESTREAM=false
MEDIA_PUBLISH_CLEAN_ENCODER=h264_vaapi
MEDIA_PUBLISH_ENCODER=h264_vaapi
MEDIA_PUBLISH_VAAPI_DEVICE=/dev/dri/renderD128
# 同一支攝影機的原始、標註與預覽發布器共用觀看需求查詢結果。
MEDIA_DEMAND_CACHE_SECONDS=0.5
```

各 API 服務共用同一組 OIDC 簽發者、JWKS 網址與 API 目標對象設定。

### 4. 啟動基礎服務

資料庫模式需要 Redis、PostgreSQL 與 MediaMTX。

```bash
docker compose up -d redis postgres media-server
```

使用下列指令檢查容器狀態與日誌。

```bash
docker compose ps redis postgres media-server
docker compose logs -f redis postgres media-server
```

若主機已管理 PostgreSQL 與 Redis，可用獨立 Compose 設定只啟動 MediaMTX。

```bash
docker compose -f docker-compose.media.yml up -d
docker compose -f docker-compose.media.yml ps
docker compose -f docker-compose.media.yml logs -f media-server
```

完整的 `docker-compose.yml` 會啟動 PostgreSQL 容器。請直接設定
`POSTGRES_DB`、`POSTGRES_USER` 和 `POSTGRES_PASSWORD`。

Python 服務在主機上執行時，請在 `.env` 使用本機位址。

```dotenv
DATABASE_URL='postgresql+asyncpg://username:password@127.0.0.1/construction_hazard_detection'
REDIS_HOST='127.0.0.1'
REDIS_PORT=6379
REDIS_PASSWORD='password'
MEDIA_PUBLISH_RTSP_BASE_URL='rtsp://127.0.0.1:8554'
```

`main.py` 或 FastAPI 服務在 Docker Compose 內執行時，請使用服務名稱。

```dotenv
DATABASE_URL='postgresql+asyncpg://username:password@postgres/construction_hazard_detection'
REDIS_HOST='redis'
POSTGRES_DB='construction_hazard_detection'
POSTGRES_USER='construction_app'
POSTGRES_PASSWORD='set-a-strong-password'
REDIS_PASSWORD='set-a-strong-password'
MEDIA_PUBLISH_RTSP_BASE_URL='rtsp://media-server:8554'
```

- PostgreSQL 儲存使用者、工地、串流設定與違規紀錄。違規圖片儲存於違規紀錄服務的檔案系統。
- Redis 保存驗證與權杖快取、直播中繼資料，以及標註串流的觀看需求。
- MediaMTX 接收 `main.py` 發布的 RTSP 串流，並提供 HLS／WebRTC 播放。

若新建的空白 PostgreSQL 資料庫尚未掛載初始化結構，可手動匯入。此腳本會刪除應用程式
的資料表，僅限空白資料庫使用。

```bash
cat ./scripts/init.postgres.sql | docker exec -i postgres-container \
  psql -U username -d construction_hazard_detection
```

既有資料庫請依組織的資料庫變更管理流程處理。遷移與特權維運工具由部署端管理。

### 5. 設定 MediaMTX

`main.py` 透過 RTSP 將處理後的 H.264 串流發布到 MediaMTX，觀看端再透過 HLS 或 WebRTC
播放。Redis 負責保存直播中繼資料與觀看需求。

Docker Compose 的 MediaMTX 服務預設綁定下列本機連接埠。

```text
127.0.0.1:8554  接收 main.py 或 FFmpeg 發布的 RTSP 串流
127.0.0.1:8890  HLS 播放，對應 MediaMTX 的 8888 連接埠
127.0.0.1:8889  WebRTC／WHEP 播放
```

`main.py` 在主機執行時，可使用以下設定。

```dotenv
MEDIA_PUBLISH_RTSP_BASE_URL='rtsp://127.0.0.1:8554'
MEDIA_PUBLIC_HLS_BASE_URL='/hazard/media'
MEDIA_PUBLIC_WEBRTC_BASE_URL='/hazard/media/webrtc'
```

在主機執行 `main.py` 的帳號需要有 Intel 繪圖裝置的存取權。

```bash
sudo usermod -aG render "$USER"
# 重新登入後再啟動 main.py。
```

上述設定使用按需發布，透過 Intel 內顯的 `h264_vaapi` 編碼原始、標註與預覽畫面，
由 MediaMTX 分發給所有觀看者。攝影機原始位元流符合播放需求時，可將
`MEDIA_PUBLISH_CLEAN_SOURCE_RESTREAM` 設為 `true`，透過 FFmpeg 直接轉送原始串流。

`main.py` 在 Docker 內執行時，請改用 Compose 服務名稱。

```dotenv
MEDIA_PUBLISH_RTSP_BASE_URL='rtsp://media-server:8554'
```

Docker Compose 會將 `/dev/dri/renderD128` 映射到偵測容器。若主機 `render` 群組的 ID
與預設值 `992` 不同，請透過 `INTEL_RENDER_GID` 指定實際 ID。

串流後端會依下列格式回傳播放網址。

```text
/hazard/media/<media-path>/index.m3u8
/hazard/media/webrtc/<media-path>/whep
```

公開部署時，建議用 Nginx 反向代理 MediaMTX，並由 `examples.streaming_web` 驗證播放權限。
可參考 `examples/streaming_web/nginx.hazard-media.conf` 的路由設定。

```text
Nginx /hazard/media/*        -> MediaMTX HLS port 8888
Nginx /hazard/media/webrtc/* -> MediaMTX WebRTC port 8889
Nginx /hazard/api/media-auth -> examples.streaming_web /media-auth
```

`docker-compose.yml` 包含建議的直播緩衝設定。

```dotenv
MTX_HLSSEGMENTDURATION=2s
MTX_HLSSEGMENTCOUNT=14
MTX_HLSALWAYSREMUX=yes
MTX_HLSMUXERCLOSEAFTER=60s
```

降低 `MTX_HLSSEGMENTCOUNT` 可減少硬碟與記憶體使用量。預設 14 個片段（每段 2 秒時約
28 秒）可讓多鏡頭牆在短暫網路或發布中斷後恢復。

### 6. 啟動 API

請在儲存庫根目錄分別啟動各項服務。

```bash
uvicorn examples.db_management.app:app --host 127.0.0.1 --port 8005 --workers 2 --timeout-graceful-shutdown 10
uvicorn examples.local_notification_server.app:app --host 127.0.0.1 --port 8003 --workers 2
uvicorn examples.violation_records.app:app --host 127.0.0.1 --port 8002 --workers 2
uvicorn examples.streaming_web.app:app --host 127.0.0.1 --port 8800 --workers 2 --timeout-graceful-shutdown 10
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

同一批攝影機請擇一使用資料庫模式或 JSON 模式。

## 直播觀看

前端或 App 可使用下列網頁串流 API。

- `GET /labels`
- `GET /streams/{label}`
- `GET /metadata/stream-id/{label}/{stream_id}` 透過 SSE 傳送中繼資料。
- `WebSocket /ws/metadata-id/{label}/{stream_id}` 透過 WebSocket 傳送中繼資料。
- `GET /webrtc/ice-servers` 提供 WebRTC 的 STUN／TURN 設定。

前端使用 MediaMTX 提供的 HLS／WebRTC 網址播放影像。警示中繼資料通常只包含目前的
警示狀態。後端會將偵測框、多邊形與文字標籤繪製在標註串流中。

預設擷取每個解碼影格，原始、標註與預覽串流都沿用攝影機的影格率，包含 29.97 FPS
等小數值。預覽只降低解析度與位元率。推論獨立取用最新影格，播放影格率由來源決定。
重新連線後會再次讀取來源影格率，數值變更時重建發布器。

`MEDIA_PUBLISH_SOURCE_FPS`、`MEDIA_PUBLISH_FPS`、`MEDIA_PREVIEW_FPS` 與
`MEDIA_PUBLISH_CLEAN_FPS` 預設為 `auto`，空值也代表自動。填入數字可設定擷取取樣率
或輸出影格率。若只設定詳細畫面的影格率，預覽會沿用該值。自動模式直接轉送來源時保留
輸入時間戳記。使用未壓縮影格編碼的發布器在顯示啟動畫面或來源無法提供有效影格率時，
暫以 30 FPS 發布。

同一支攝影機的各種輸出發布器與來源轉送器共用啟動時載入的 `MediaPublishSettings`，
包含編碼器選項、GOP、位元率及裝置。重新連線時會保留這份設定。渲染器為每種輸出語言
複製一次共享來源影格。工作者派工與結果佇列使用資料庫的 `stream_id`，因此共用 RTSP
網址的攝影機各自保有獨立識別。

批次播放會先驗證整批工作階段，再以同一個 Redis 交易提交工作階段、需求租約及反向索引。
指定的工作階段不存在或不屬於目前使用者時，整批維持原狀。HLS 每次最多續訂
`STREAM_PLAYBACK_REFRESH_BATCH_SIZE` 個工作階段（預設 256），並分批清除過期的反向
索引。續訂項目移到排序末端，讓後續批次處理其他項目。節流狀態、索引與工作階段會在
同一個 Redis 交易中更新。

直播中繼資料頻道提供最新狀態快照。Redis 每次最多合併 `STREAMING_METADATA_READ_BATCH_SIZE`
筆（預設 64）。讀滿時會額外讀取最新一筆，略過積壓的舊狀態。每個訂閱管理自己的標註
租約、就緒與心跳狀態，並在觀看者斷線時關閉共用的轉發訂閱。違規事件與通知透過 outbox 派送。

模型發布統一使用 `python -m scripts.sync_huggingface_models`。流程會驗證新舊
模型檔案、重啟受影響的工作者，並發出新的驗證請求以確認實際載入版本。失敗時會還原
並驗證前一版本。INT8 資訊清單必須保留可用的前版引擎。詳見
[`MODEL_CATALOG.md`](examples/YOLO_server_api/MODEL_CATALOG.md)。

公開部署時，MediaMTX 建議放在 Nginx `auth_request` 後面，並由
`examples.streaming_web` 驗證 JWT 與工地權限。

## 儲存與保留

違規圖片存放於服務設定的 `static/` 目錄。若暫無歸檔儲存空間、NAS 或外接硬碟，
可採用下列保留方式。

- 圖片與違規紀錄保留 18 個月。
- 清理時一併刪除圖片及對應的資料庫紀錄。
- 把清理工作排在低流量時段。
- MediaMTX 的 HLS 片段用於直播緩衝，建議短期保留。

## 開發檢查

```bash
pre-commit run -a
python -m pytest -q --tb=short
```

mypy 的 pre-commit 掛鉤會檢查正式程式碼，包含 `main.py`、`src`、`examples` 與
`scripts`。測試由 `pytest` 與 `flake8` 驗證，涵蓋動態模擬物件與無效輸入等情境。

## 資料集與模型

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

## 執行行為

### 違規證據持久化與派送

本節說明 `main.py` 的 SQLite outbox。獨立 Rust 流程的持久化與派送方式見
[PostgreSQL outbox 說明](native/pipeline/POSTGRES_OUTBOX.md)。

`main.py` 的 outbox 派送器共用一個權杖提供者，並透過連線池取得憑證。每次憑證請求
只嘗試一次，事件派送負責控制重試次數與退避。攝影機佇列按先進先出順序處理，工作者
數量有限。忙碌的攝影機會排回隊尾，成功處理事件後會接著處理積壓事件。`VIOLATION_OUTBOX_CONCURRENCY`
預設為 4，`VIOLATION_OUTBOX_EVENT_TIMEOUT_SECONDS` 預設為 60 秒、最多 100 秒。
事件領取租約為 120 秒。取消時會先釋放仍在處理的租約，再關閉連線。

攝影機程序將有版本標記的證據中繼資料與圖片寫入本機 SQLite outbox。監督程序以一個
派送器處理所有已持久化的攝影機佇列，停用攝影機後仍可派送待處理事件。Outbox 也可讀取
舊版無版本標記的資料。升級工作者時請保留 outbox 的持久化儲存空間。

證據上傳與通知派送各自記錄完成狀態。上傳成功後會儲存違規 ID 並釋放圖片，再進行通知
重試。沒有訂閱者或可接收通知的裝置時，會標記為 `skipped`。暫時性錯誤會在退避後重試。
永久性錯誤或用盡重試次數的事件會隔離並保留失敗原因。隔離的事件也計入佇列的容量與筆數限制。

```bash
python -m src.delivery_outbox_cli
python -m src.delivery_outbox_cli --retry EVENT_ID
```

`VIOLATION_OUTBOX_MAX_ATTEMPTS` 預設為 20 次，`VIOLATION_OUTBOX_POLL_SECONDS` 預設為
1 秒。維運人員可在排除問題後以指令重試失敗事件。派送器透過租約避免同一事件被同時領取。
推播服務接受請求後若網路回應不明，仍採至少一次派送語意。證據以事件 UUID 避免重複，
App 內通知則使用通知中心的收件者鍵值。

排程器異常時，派送器會保留磁碟佇列與事件租約，重新建立派送資源。恢復等待時間自
`VIOLATION_OUTBOX_RECOVERY_SECONDS`（預設 1 秒）起逐次加倍，最多為
`VIOLATION_OUTBOX_MAX_RECOVERY_SECONDS`（預設 60 秒）。成功輪詢後會重設等待時間。
`health` 快照會回報執行狀態、累積失敗數、最近的錯誤類型及最近一次成功輪詢時間。關閉時
會中止恢復等待，再結束正在執行的工作。

HTTP、MCP、outbox 序列化與請求指紋均將無時區偏移的時間戳記視為 UTC。含偏移量的時間
會轉成 UTC。重試過去依伺服器本地時區記錄的事件時，請提供含偏移量的時間戳記。

HTTP 與 MCP 採用同一套追蹤偵測資料驗證規則。數值必須有限，邊界框面積須大於零，
信心分數須在 0 至 1 之間，類別 ID 須為非負整數，追蹤 ID 須為大於等於 `-1` 的整數
（`-1` 代表未分配）。超出畫面的座標會在後續裁切。無效上傳會回傳 HTTP 422。既有紀錄
仍可查閱，繪製時會略過無效偵測資料並保留原始證據。

### 身分驗證與通知

原生社群帳號綁定在同一筆交易中準備並驗證本機帳號對應資料，再呼叫 Keycloak。若遠端
結果不明或本機交易無法提交，會保留既有的共用身分並回傳 `account_reconciliation_required`。
取得新的有效證明後，可再次執行一般綁定流程，辨識既有的遠端綁定並修復本機資料。帳號查詢
與綁定共用 `VerifiedSocialIdentity`，而證明驗證、一次性交易儲存及身分代理兌換各由專用
模組處理。

BFF 更新權杖時會續租處理權，且只有在租約擁有者與工作階段版本皆一致時，才會儲存新權杖。
強制更新須取得較新的工作階段版本與不同的有效存取權杖。若其他請求已完成更新，會重用其
有效權杖。瀏覽器取消請求時會完成進行中的權杖更新與儲存，再釋放租約。程序意外終止或
權杖交換的回應遺失時，可能需要重新登入。

FCM 發送前會記錄裝置註冊版本，且只在註冊資料與擁有者皆相符時更新發送結果。成功發送會
更新診斷資訊，停用的裝置維持停用。每次取得或續訂通知租約時，會用單調時鐘計算期限。
Redis 操作最多等待 5 秒，領取的工作須在租約期限內完成。無法確認租約時會暫停新批次並
等待取消完成。FCM 已接受的請求仍採至少一次派送語意。

### 工作者與設定

YOLO 工作者分組異常重啟時，會建立新一代程序間通訊（IPC）的請求與結果佇列，再由設定
協調流程重新綁定受影響的攝影機程序。正常分組保留既有佇列。模型載入時使用檔案鎖，
載入程序終止時由作業系統自動釋放。

環境設定按用途整理為具型別的模型。用戶端與工作者在各自的生命週期中會使用已驗證的
設定快照。明確指定的建構參數優先於環境變數，例如可以用空字串覆寫環境變數設定的 Redis
密碼。無效數值、布林值及 MCP 傳輸設定會在載入時報錯。生命週期、憑證更新與驗證規則
見[設定管理說明](src/settings/README-zh-tw.md)。

## 授權

本專案採用 [AGPL-3.0 授權條款](LICENSE)。
