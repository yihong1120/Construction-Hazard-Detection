# 🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

<img width="100%" src="./assets/images/project_graphics/banner.gif" alt="AI-Driven Construction Safety Banner">

<div align="center">
   <a href="examples/YOLO_server_api">模型辨識伺服器</a> |
   <a href="examples/local_notification_server">FCM 通知管理伺服器</a>|
   <a href="examples/violation_records">違規記錄伺服器</a>|
   <a href="examples/db_management">資料管理伺服器</a>|
   <a href="examples/streaming_web">串流網頁</a> |
   <a href="examples/YOLO_data_augmentation">YOLO 數據增強</a> |
   <a href="examples/YOLO_evaluation">YOLO 評估</a> |
   <a href="examples/YOLO_train">YOLO 訓練</a>
</div>

<br>

<div align="center">
   <!-- 第一行：工具與框架 -->
   <a href="https://www.python.org/downloads/release/python-3127/">
      <img src="https://img.shields.io/badge/python-3.12.7-blue?logo=python" alt="Python 3.12.7">
   </a>
   <a href="https://github.com/ultralytics/ultralytics">
      <img src="https://img.shields.io/badge/YOLO11-ultralytics-blue?logo=yolo" alt="YOLO11">
   </a>
   <a href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html">
      <img src="https://img.shields.io/badge/HDBSCAN-sklearn-orange?logo=scikit-learn" alt="HDBSCAN sklearn">
   </a>
   <a href="https://fastapi.tiangolo.com/">
   <img src="https://img.shields.io/badge/fastapi-0.115.5-blue?logo=fastapi" alt="FastAPI 0.115.5">
   </a>
   <a href="https://redis.io/">
      <img src="https://img.shields.io/badge/Redis-7.0.15-red?logo=redis" alt="Redis 7.0.15">
   </a>
   <a href="https://www.docker.com/">
      <img src="https://img.shields.io/badge/Docker-Container-blue?logo=docker" alt="Docker">
   </a>

   <!-- 第二行：測試、品質與資料 -->

   <a href="https://github.com/pre-commit/pre-commit">
      <img src="https://img.shields.io/badge/pre--commit-4.0.1-blue?logo=pre-commit" alt="Pre-commit 4.0.1">
   </a>
   <a href="https://docs.pytest.org/en/latest/">
      <img src="https://img.shields.io/badge/pytest-8.3.4-blue?logo=pytest" alt="pytest 8.3.4">
   </a>
   <a href="https://codecov.io/github/yihong1120/Construction-Hazard-Detection" >
      <img src="https://codecov.io/github/yihong1120/Construction-Hazard-Detection/graph/badge.svg?token=E0M66BUS8D" alt="Codecov">
   </a>
   <a href="https://codebeat.co/projects/github-com-yihong1120-construction-hazard-detection-main">
      <img alt="codebeat badge" src="https://codebeat.co/badges/383396a9-e2cb-4604-8990-c1707e5870cf" />
   </a>
   <a href="https://universe.roboflow.com/object-detection-qn97p/construction-hazard-detection">
      <img src="https://app.roboflow.com/images/download-dataset-badge.svg" alt="Download Dataset from Roboflow">
   </a>
   <a href="https://huggingface.co/yihong1120/Construction-Hazard-Detection-YOLO11">
      <img src="https://img.shields.io/badge/HuggingFace-Model%20Repo-yellow?logo=huggingface" alt="Hugging Face Model Repo">
   </a>
</div>

<br>

"建築工地危險檢測系統" 是一款以人工智慧驅動的工具，旨在提升工地的安全性。該系統利用 YOLO 模型與自研演算法進行物件偵測，能夠識別以下潛在危險：

- 未佩戴安全帽的工人
- 未穿著安全背心的工人
- 靠近機械或車輛的工人
- 進入限制區域的工人，透過安全錐座標的計算和分群，限制區域將自動產生。
- 靠近電線桿的機具或車輛

等高風險情境，並推播警示。

系統支援透過網頁即時整合 AI 辨識結果，並可用 LINE、Messenger、微信、Telegram 等訊息平台推播圖文與現場影像。亦支援多語系通知與介面（繁中、簡中、英、法、泰、越、印尼）。

<br>
<br>

<div align="center">
   <img src="./assets/images/hazard-detection.png" alt="diagram" style="width: 100%;">
</div>

<br>

## 目錄

- [危險偵測範例](#危險偵測範例)
- [操作說明](#操作說明)
- [資料庫設定與管理](#資料庫設定與管理)
- [環境變數](#環境變數)
- [附加信息](#附加信息)
- [數據集信息](#數據集信息)
- [貢獻](#貢獻)
- [開發路線圖](#開發路線圖)
- [授權](#授權)

## 危險偵測範例

以下是系統實時危險偵測的範例：

<div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
  <!-- 範例 1: 未佩戴安全帽或安全背心的工人 -->
  <div style="text-align: center; flex-basis: 33%;">
    <img src="./assets/images/demo/person_did_not_wear_safety_vest.png" alt="未佩戴安全帽或安全背心的工人" style="width: 300px; height: 200px; object-fit: cover;">
    <p>未佩戴安全帽或安全背心的工人</p>
  </div>

  <!-- 範例 2: 工人接近機具或車輛 -->
  <div style="text-align: center; flex-basis: 33%;">
    <img src="./assets/images/demo/person_near_machinery.jpg" alt="工人接近機具或車輛" style="width: 300px; height: 200px; object-fit: cover;">
    <p>工人接近機具或車輛</p>
  </div>

  <!-- 範例 3: 工人在限制區域內 -->
  <div style="text-align: center; flex-basis: 33%;">
    <img src="./assets/images/demo/persons_in_restricted_zones.jpg" alt="工人在限制區域內" style="width: 300px; height: 200px; object-fit: cover;">
    <p>工人在限制區域內</p>
  </div>
</div>

## 操作說明

在運行應用程式之前，請先準備串流設定檔 `config/configuration.json`（格式示例如下），以及完成「啟動方式」與「環境變數」的設定。

```json
[
  {
    "video_url": "https://cctv1.kctmc.nat.gov.tw/6e559e58/",
    "site": "Kaohsiung",
    "stream_name": "Test",
    "model_key": "yolo11n",
    "notifications": {
      "line_token_1": "language_1",
      "line_token_2": "language_2"
    },
    "detect_with_server": true,
    "expire_date": "2024-12-31T23:59:59",
    "detection_items": {
      "detect_no_safety_vest_or_helmet": true,
      "detect_near_machinery_or_vehicle": true,
      "detect_in_restricted_area": true
    },
    "work_start_hour": 0,
    "work_end_hour": 24,
    "store_in_redis": true
  }
]
```

### ✅ 啟動方式（請依序完成）

1. **安裝 Docker（必須）**
   請先在電腦安裝 \[Docker Desktop / Docker Engine]。

2. **以 Docker 啟動 Redis**

   ```bash
   docker run -d --name my-redis -p 6379:6379 \
   redis:latest redis-server --requirepass "password"
   ```

3. **以 Docker 啟動 MySQL（Windows / Linux 通用）**

   ```bash
   docker run -d --name my-mysql -p 3306:3306 \
   -e MYSQL_ROOT_PASSWORD=ChangeMe! \
   -e MYSQL_DATABASE=construction_hazard_detection \
   -e MYSQL_USER=username \
   -e MYSQL_PASSWORD=password \
   mysql:latest \
   --default-authentication-plugin=mysql_native_password \
   --character-set-server=utf8mb4 \
   --collation-server=utf8mb4_unicode_ci
   ```

4. **匯入資料庫結構（僅首次）**

* **Windows PowerShell / CMD：**

   ```bash
   type .\scripts\init.sql | docker exec -i my-mysql \
   mysql -u username -p"password" --default-character-set=utf8mb4 construction_hazard_detection
   ```

* **Linux / macOS：**

   ```bash
   cat ./scripts/init.sql | docker exec -i my-mysql \
   mysql -u username -p"password" --default-character-set=utf8mb4 construction_hazard_detection
   ```

5. **執行以下指令安裝專案所需的套件：**

   ```bash
   pip install -r requirements.txt
   ```

6. **從 Hugging Face 下載 YOLO11 權重（存到 models/pt）**

- CLI（下載到 `./models/pt`）

    Hugging Face 模型庫：https://huggingface.co/yihong1120/Construction-Hazard-Detection-YOLO11

   ```bash
   hf download \
     yihong1120/Construction-Hazard-Detection-YOLO11 \
     --repo-type model \
     --include "models/pt/*.pt" \
       --local-dir .
   ```

- Python（效果相同）

   ```python
   from huggingface_hub import snapshot_download

   snapshot_download(
       repo_id="yihong1120/Construction-Hazard-Detection-YOLO11",
       repo_type="model",
       local_dir=".",
       local_dir_use_symlinks=False,
       allow_patterns=["models/pt/*.pt"],
   )
   ```

7. **啟動各 API（於專案根目錄執行，建議使用獨立終端分別啟動）**

* 資料管理（DB Management）

   ```bash
   uvicorn examples.db_management.app:app --host 127.0.0.1 --port 8005 --workers 2
   ```

* 通知伺服器（FCM / 本地推播）

   ```bash
   uvicorn examples.local_notification_server.app:app --host 127.0.0.1 --port 8003 --workers 2
   ```

* 串流網頁後端

   ```bash
   uvicorn examples.streaming_web.backend.app:app --host 127.0.0.1 --port 8800 --workers 2
   ```

* 違規紀錄 API

   ```bash
   uvicorn examples.violation_records.app:app --host 127.0.0.1 --port 8002 --workers 2
   ```

* YOLO 辨識 API

   ```bash
   uvicorn examples.YOLO_server_api.backend.app:app --host 127.0.0.1 --port 8000 --workers 2
   ```

8. **啟動主程式（兩種模式）**

* **資料庫模式（預設）**
  `main.py` 會每隔固定秒數輪詢資料庫的 `stream_configs`，並動態啟動/重啟/停止各串流的子行程。

   ```bash
   python main.py
   # 可選：調整輪詢秒數（預設 10 秒）
   python main.py --poll 5
   ```

* **JSON 模式（以檔案配置多串流）**
  使用 `--config` 指定 JSON 檔路徑；主程式會為 JSON 陣列中的每個串流配置各自啟動一個子行程。

   ```bash
   python main.py --config config/configuration.json
   ```

> 停止程式：按下 `Ctrl + C` 會優雅地結束所有子行程並釋放資源。
> `.env` 會在啟動時自動載入（`python-dotenv`）。

9. **前端頁面（雲端 Demo）**
   開啟：[https://visionnaire-cda17.web.app](https://visionnaire-cda17.web.app)
   登入帳密：`user` / `password`

> 進入後可在網頁設定頁修改 API 位址。


## 資料庫設定與管理

系統以 MySQL 儲存串流、工地、違規、使用者等資料。
資料庫名稱：`construction_hazard_detection`

主要表格：

* `stream_configs`：串流設定（URL、工地、模型、通知、偵測項目、工作時段…）
* `sites`：工地資訊
* `users`：系統使用者
* `violations`：違規記錄

> 詳細 SQL 結構請見 `scripts/init.sql`。首次啟動請先完成上方「匯入資料庫結構」。

## 環境變數

請於專案根目錄建立 `.env`，內容如下（可依實際環境調整）：

```dotenv
# Database
DATABASE_URL='mysql+asyncmy://username:password@127.0.0.1/construction_hazard_detection'

# API Credentials
API_USERNAME='user'
API_PASSWORD='password'

# API URLs
DETECT_API_URL="http://127.0.0.1:8000"
FCM_API_URL="http://127.0.0.1:8003"
VIOLATION_RECORD_API_URL="http://127.0.0.1:8002"
STREAMING_API_URL="http://127.0.0.1:8800"
DB_MANAGEMENT_API_URL="http://127.0.0.1:8005"

# Redis
REDIS_HOST='127.0.0.1'
REDIS_PORT=6379
REDIS_PASSWORD='password'

# Firebase
FIREBASE_CRED_PATH='examples/local_notification_server/your-firebase-adminsdk.json'
project_id='your-project-id'

# （選用）LINE / Cloudinary 憑證，如有整合請加上：
# LINE_CHANNEL_ACCESS_TOKEN='YOUR_LINE_CHANNEL_ACCESS_TOKEN'
# CLOUDINARY_CLOUD_NAME='YOUR_CLOUDINARY_CLOUD_NAME'
# CLOUDINARY_API_KEY='YOUR_CLOUD_API_KEY'
# CLOUDINARY_API_SECRET='YOUR_CLOUDINARY_API_SECRET'
```

> **注意**
>
> * 若服務部署在不同主機或容器網路，請把 `127.0.0.1` 改成對方可連線的位址。
> * MySQL 與 Redis 的帳密需與您啟動容器時的環境變數一致（例如 `MYSQL_USER`、`MYSQL_PASSWORD`、`--requirepass`）。
> * 初次啟動前請務必先完成 `scripts/init.sql` 的匯入。

## 附加信息

- 系統日誌可在 Docker 容器內部訪問，可用於調試目的。
- 如果啟用，檢測到的輸出圖像將保存到指定的輸出路徑。
- 如果檢測到危險，將在指定小時通過 LINE 消息 API 發送通知。

### 注意事項
- 確保 `Dockerfile` 存在於項目的根目錄中，並根據您的應用程序的要求進行了正確配置。

## 數據集信息
訓練此模型的主要數據集是 [Roboflow 的建築工地安全圖像數據集](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow/data)。我們已經用額外的註釋豐富了這個數據集，並在 Roboflow 上公開訪問。增強的數據集可以在這裡找到：[Roboflow 上的建築危險檢測](https://universe.roboflow.com/side-projects/construction-hazard-detection)。此數據集包括以下標籤：

- `0: '安全帽'`
- `1: '口罩'`
- `2: '無安全帽'`
- `3: '無口罩'`
- `4: '無安全背心'`
- `5: '人'`
- `6: '安全錐'`
- `7: '安全背心'`
- `8: '機械'`
- `9: '電線桿'`
- `10: '車輛'`

### 檢測模型

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50 | mAP<sup>val<br>50-95 | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------- | --------------------- | ------------------ | ------------------ | ----------------- | ----------------- |
| YOLO11n | 640                   | 58.0               | 34.2               | 2.6               | 6.5               |
| YOLO11s | 640                   | 70.1               | 44.8               | 9.4               | 21.6              |
| YOLO11m | 640                   | 73.3               | 42.6               | 20.1              | 68.0              |
| YOLO11l | 640                   | 77.3               | 54.6               | 25.3              | 86.9              |
| YOLO11x | 640                   | 82.0               | 61.7               | 56.9              | 194.9             |

<br>

我們的全面數據集確保模型能夠識別建築環境中常見的各種潛在危險。

## 授權

此項目根據 [AGPL-3.0](LICENSE.md) 授權。
