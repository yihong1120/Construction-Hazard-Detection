🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

<img width="100%" src="./assets/images/project_graphics/banner.gif" alt="AI-Driven Construction Safety Banner">

<div align="center">
   <a href="examples/YOLO_server_api">模型伺服器</a> |
   <a href="examples/streaming_web">串流網頁</a> |
   <a href="examples/YOLO_data_augmentation">YOLO 數據增強</a> |
   <a href="examples/YOLO_evaluation">YOLO 評估</a> |
   <a href="examples/YOLO_train">YOLO 訓練</a>
</div>

<br>

<div align="center">
   <!-- 第一行：工具与框架 -->
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
      <img src="https://img.shields.io/badge/fastapi-0.115.4-blue?logo=fastapi" alt="FastAPI 0.115.4">
   </a>
   <a href="https://redis.io/">
      <img src="https://img.shields.io/badge/Redis-7.0.15-red?logo=redis" alt="Redis 7.0.15">
   </a>
   <a href="https://www.docker.com/">
      <img src="https://img.shields.io/badge/Docker-Container-blue?logo=docker" alt="Docker">
   </a>

   <!-- 第二行：测试、代码质量与数据 -->
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
</div>

<br>

"建築工地危險檢測系統" 是一款以人工智慧驅動的工具，旨在提升工地的安全性。該系統利用 YOLO 模型進行物件偵測，能夠識別以下潛在危險：

- 未佩戴安全帽的工人
- 未穿著安全背心的工人
- 靠近機械或車輛的工人
- 進入限制區域的工人，透過安全錐座標的計算和分群，限制區域將自動產生。

後處理算法進一步提高了偵測的準確性。該系統專為即時部署而設計，能夠對檢測到的危險進行即時分析並發出警報。

此外，該系統可通過網頁介面即時整合 AI 辨識結果，並通過 LINE、Messenger、微信和 Telegram 等即時通訊應用程式發送通知及現場實時影像，及時提醒和通知相關人員。系統還支援多種語言，允許用戶接收通知並以其首選語言與介面互動。支援的語言包括：

- 🇹🇼 繁體中文（台灣）
- 🇨🇳 簡體中文（中國大陸）
- 🇫🇷 法文
- 🇬🇧 英文
- 🇹🇭 泰文
- 🇻🇳 越南文
- 🇮🇩 印尼文

多語言支援使該系統在全球範圍內的使用者都能方便使用，提升了不同地區的可用性。

<br>
<br>

<div align="center">
   <img src="./assets/images/hazard-detection.png" alt="diagram" style="width: 100%;">
</div>

<br>

## 內容

- [危險偵測範例](#hazard-detection-examples)
- [操作說明](#操作說明)
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

在運行應用程式之前，您需要配置系統，指定視頻流的詳細信息和其他參數，這些信息需要在 JSON 配置文件中進行設置。示例配置文件 `config/configuration.json` 應該看起來像這樣：

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
  },
  {
    "video_url": "streaming URL",
    "site": "Factory_1",
    "stream_name": "camera_1",
    "model_key": "yolo11n",
    "notifications": {
      "line_token_3": "language_3",
      "line_token_4": "language_4"
    },
    "detect_with_server": false,
    "expire_date": "No Expire Date",
    "detection_items": {
      "detect_no_safety_vest_or_helmet": true,
      "detect_near_machinery_or_vehicle": false,
      "detect_in_restricted_area": true
    },
    "work_start_hour": 0,
    "work_end_hour": 24,
    "store_in_redis": true
  }
]
```

數組中的每個對象代表一個視頻流配置，包含以下字段：

- `video_url`：視訊串流的 URL，可能包括：
   - 監控流
   - RTSP
   - 副流
   - YouTube 視頻或直播
   - Discord

- `site`：監控系統的位置（例如：建築工地、工廠）。

- `stream_name`：指派給監視器或串流的名稱（例如：「前門」、「相機1」）。

- `model_key`：機器學習模型的識別鍵（例如：「yolo11n」）。

- `notifications`：LINE 通知的 API 令牌和對應語言的列表。
   - `line_token_1`, `line_token_2` 等：這些是 LINE API 令牌。
   - `language_1`, `language_2` 等：通知的語言（例如：「en」表示英文，「zh-TW」表示繁體中文）。

   支援的通知語言包括：
   - `zh-TW`: 繁體中文
   - `zh-CN`: 簡體中文
   - `en`: 英語
   - `fr`：法語
   - `vi`：越南語
   - `id`：印尼語
   - `th`：泰語

   有關如何獲取 LINE 令牌的資訊，請參閱  [Line Notify教學](docs/zh/line_notify_guide_zh.md)。

- `detect_with_server`：布林值，指示是否使用伺服器 API 進行物件偵測。如果為 `True`，系統將使用伺服器進行物件偵測。如果為 `False`，物件偵測將在本地機器上執行。

- `expire_date`：視訊串流配置的到期日期，使用 ISO 8601 格式（例如：「2024-12-31T23:59:59」）。如果沒有到期日期，可以使用類似「無到期日期」的字串。

- `detection_items`：指定監控特定場景的安全偵測項目。每個項目都可以設定為“True”以啟用或設定為“False”以停用。可用的檢測項目有：
   - `detect_no_safety_vest_or_helmet`：偵測人員是否未配戴安全背心或頭盔。這對於在必須配備安全裝備以保護人員的場所監控其安全裝備要求的遵守情況至關重要。
   - `detect_near_machinery_or_vehicle`：偵測人員是否危險地靠近機器或車輛。這有助於防止在建築工地或工業區經常遇到的因靠近重型設備或移動車輛而引起的事故。
   - `detect_in_restricted_area`：偵測人員是否進入限製或控制區域。限制區域對於未經訓練的人員可能是危險的，或者可能包含敏感設備，因此此設定有助於控制對此類區域的存取。

- `work_start_hour`：指定係統應開始監控視訊串流的時間（24 小時格式）。這樣可以將監控限制在活躍的工作時間，從而減少定義的時間範圍之外的不必要的處理（例如，「7」表示上午 7:00）。

- `work_end_hour`：指定係統應停止監控視訊串流的時間（24 小時格式）。在此時間之後，監控將停止以最佳化資源使用（例如「18」表示下午 6:00）。

 「work_start_hour」和「work_end_hour」一起定義一天中的監控時段。對於 24 小時監控，請將“work_start_hour”設為“0”，將“work_end_hour”設為“24”。

- `store_in_redis`：一個布林值，決定是否將處理後的幀和關聯的檢測資料儲存在 Redis 中。如果為“True”，系統會將資料儲存到 Redis 資料庫以供進一步使用，例如即時監控或與其他服務整合。如果為“False”，則 Redis 中不會保存任何資料。


### 環境變數

本應用程式需要特定的環境變數進行正確的配置。這些變數應定義在專案根目錄中的 `.env` 檔案中。以下是 `.env` 檔案的範例內容：

```plaintext
DATABASE_URL='mysql+asyncmy://username:password@mysql/construction_hazard_detection'
API_USERNAME='user'
API_PASSWORD='password'
API_URL="http://yolo-server-api:6000"
REDIS_HOST='redis'
REDIS_PORT=6379
REDIS_PASSWORD='password'
LINE_CHANNEL_ACCESS_TOKEN='YOUR_LINE_CHANNEL_ACCESS_TOKEN'
CLOUDINARY_CLOUD_NAME='YOUR_CLOUDINARY_CLOUD_NAME'
CLOUDINARY_API_KEY='YOUR_CLOUD_API_KEY'
CLOUDINARY_API_SECRET='YOUR_CLOUD_API_SECRET'
```

#### 變數說明：

- `DATABASE_URL`：MySQL 資料庫的連線 URL。由 `server_api` 模組使用。
- `API_USERNAME`：用於 API 認證的使用者名稱。由 `main.py` 使用。
- `API_PASSWORD`：用於 API 認證的密碼。由 `main.py` 使用。
- `API_URL`：YOLO 伺服器 API 的 URL。由 `main.py` 使用。
- `REDIS_HOST`：Redis 伺服器的主機名稱。由 `main.py` 使用。
- `REDIS_PORT`：Redis 伺服器的埠號。由 `main.py` 使用。
- `REDIS_PASSWORD`：連線至 Redis 伺服器的密碼。由 `main.py` 使用。
- `LINE_CHANNEL_ACCESS_TOKEN`：LINE Messaging API 的存取權杖。由 `src/notifiers/line_notifier_message_api.py` 使用。
- `CLOUDINARY_CLOUD_NAME`：Cloudinary 的雲端名稱，用於媒體管理。由 `src/notifiers/line_notifier_message_api.py` 使用。
- `CLOUDINARY_API_KEY`：存取 Cloudinary 服務的 API 金鑰。由 `src/notifiers/line_notifier_message_api.py` 使用。
- `CLOUDINARY_API_SECRET`：存取 Cloudinary 服務的 API 秘鑰。由 `src/notifiers/line_notifier_message_api.py` 使用。

> **注意**：請將範例中的佔位值替換為實際的憑證與配置詳細資訊，以確保應用程式的正常運作。


<br>

<details>
   <summary>Docker</summary>

   ### 使用 Docker

   要運行危險檢測系統，您需要在機器上安裝 Docker 和 Docker Compose。按照以下步驟來啟動系統：

   1. 將存儲庫克隆到本地機器。
      ```bash
      git clone https://github.com/yihong1120/Construction-Hazard-Detection.git
      ```
   2. 進入克隆的目錄。
      ```bash
      cd Construction-Hazard-Detection
      ```
   3. 使用 Docker Compose 構建並運行服務：
      ```bash
      docker-compose build
      ```

   4. 要運行該應用程序，請使用以下命令：
      ```bash
      docker-compose up
      ```
      您可在http://localhost查看辨識結果

   5. 停止服務，使用以下命令：
      ```bash
      docker-compose down
      ```

</details>

<details>
   <summary>Python</summary>

   ### 使用 Python

   要在終端運行危險檢測系統，您需要在機器上安裝 Python。按照以下步驟來啟動系統：

   ### 1. 將專案複製至本機
   使用以下指令將專案從 GitHub 複製到本機：

   ```bash
   git clone https://github.com/yihong1120/Construction-Hazard-Detection.git
   ```

   ---

   ### 2. 進入專案目錄
   進入剛剛複製的專案目錄：

   ```bash
   cd Construction-Hazard-Detection
   ```

   ---

   ### 3. 安裝所需的套件
   執行以下指令安裝專案所需的套件：

   ```bash
   pip install -r requirements.txt
   ```

   ---

   ### 4. 安裝並啟動 MySQL 服務（如需要）

   #### Ubuntu 系統使用者：

   1. 開啟終端機，執行以下指令安裝 MySQL：
      ```bash
      sudo apt update
      sudo apt install mysql-server
      ```

   2. 安裝完成後，啟動 MySQL 服務：
      ```bash
      sudo systemctl start mysql.service
      ```

   #### 其他作業系統使用者：

   請從 [MySQL 下載頁面](https://dev.mysql.com/downloads/)下載適合您作業系統的 MySQL 並完成安裝。

   ---

   #### 初始化資料庫：

   完成 MySQL 安裝後，執行初始化腳本來建立 `construction_hazard_detection` 資料庫及 `users` 資料表：

   ```bash
   mysql -u root -p < scripts/init.sql
   ```

   執行指令後，系統會提示您輸入 MySQL 的 root 密碼。請確認 `scripts/init.sql` 檔案內已包含建立資料庫和資料表的 SQL 指令。

   ---

   ### 5. 設定 Redis 伺服器（僅限於 Streaming Web 功能）

   Redis 僅在使用 **Streaming Web** 功能時需要。請依以下步驟安裝及設定 Redis：

   #### Ubuntu 系統使用者：

   1. **安裝 Redis**：
      執行以下指令安裝 Redis：
      ```bash
      sudo apt update
      sudo apt install redis-server
      ```

   2. **設定 Redis（可選）**：
      - 如果需要進一步設定，請編輯 Redis 設定檔案：
      ```bash
      sudo vim /etc/redis/redis.conf
      ```
      - 如需啟用密碼保護，請找到以下行，取消註解並設置密碼：
      ```conf
      requirepass YourStrongPassword
      ```
      將 `YourStrongPassword` 替換為強密碼。

   3. **啟動並設置 Redis 服務開機自動啟動**：
      - 啟動 Redis 服務：
      ```bash
      sudo systemctl start redis.service
      ```
      - 設置開機自動啟動：
      ```bash
      sudo systemctl enable redis.service
      ```

   #### 其他作業系統使用者：

   請參考官方文件：[Redis 安裝指南](https://redis.io/docs/getting-started/installation/)。

   ---

   ### 6. 啟動物件偵測 API
   執行以下指令啟動物件偵測 API：

   ```bash
   uvicorn examples.YOLO_server.backend.app:sio_app --host 0.0.0.0 --port 8001
   ```

   ---

   ### 7. 使用特定配置檔案執行主應用程式
   執行以下指令啟動主應用程式，並指定配置檔案：

   ```bash
   python3 main.py --config config/configuration.json
   ```

   請將 `config/configuration.json` 替換為您的實際配置檔案路徑。

   ---

   ### 8. 啟動 Streaming Web 服務

   #### 啟動後端服務

   ##### Linux 系統使用者：
   執行以下指令啟動後端服務：

   ```bash
   uvicorn examples.streaming_web.backend.app:sio_app --host 127.0.0.1 --port 8002
   ```

   ##### Windows 系統使用者：
   使用以下 `waitress-serve` 指令啟動後端服務：

   ```cmd
   waitress-serve --host=127.0.0.1 --port=8002 "examples.streaming_web.backend.app:streaming-web-app"
   ```

   ---

   #### 設定前端服務

   請參考 `examples/YOLO_server_api/frontend/nginx.conf` 文件進行部署，並將靜態網頁檔案放置於以下目錄：

   ```plaintext
   examples/YOLO_server_api/frontend/dist
   ```

</details>

## 附加信息

- 系統日誌可在 Docker 容器內部訪問，可用於調試目的。
- 如果啟用，檢測到的輸出圖像將保存到指定的輸出路徑。
- 如果檢測到危險，將在指定小時通過 LINE 消息 API 發送通知。

### 注意事項
- 確保 `Dockerfile` 存在於項目的根目錄中，並根據您的應用程序的要求進行了正確配置。

有關 Docker 使用和命令的更多信息，請參閱 [Docker 文檔](https://docs.docker.com/)。

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
- `9: '車輛'`

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

## 待辦事項

- 新增對 WhatsApp 通知的支援。
- 修正examples/YOLO server_api/frontend手機版的UI界面。

## 貢獻

我們歡迎對此項目的貢獻。請按照以下步驟操作：
1. 分叉存儲庫。
2. 進行更改。
3. 提交一個清晰描述您改進的拉取請求。

## 授權

此項目根據 [AGPL-3.0](LICENSE.md) 授權。
