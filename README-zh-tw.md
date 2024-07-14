🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

<img width="100%"  src="./assets/images/project_graphics/banner.gif" alt="AI-Driven Construction Safety Banner">

<div align="center">
   <a href="examples/YOLOv8_server_api">模型伺服器</a> |
   <a href="examples/streaming_web">串流網頁</a> |
   <a href="examples/user_management">用戶管理</a> |
   <a href="examples/YOLOv8_data_augmentation">YOLOv8 數據增強</a> |
   <a href="examples/YOLOv8_evaluation">YOLOv8 評估</a> |
   <a href="examples/YOLOv8_train">YOLOv8 訓練</a>
</div>

<br>

<div align="center">
   <a href="https://github.com/pre-commit/pre-commit">
      <img src="https://img.shields.io/badge/pre--commit-3.7.1-blue?logo=pre-commit" alt="Pre-commit 3.7.1">
   </a>
   <a href="https://www.python.org/downloads/release/python-3124/">
      <img src="https://img.shields.io/badge/python-3.12.4-blue?logo=python" alt="Python 3.12.4">
   </a>
   <a href="https://github.com/ultralytics/ultralytics">
      <img src="https://img.shields.io/badge/YOLOv8-ultralytics-blue?logo=yolo" alt="YOLOv8">
   </a>
   <a href="https://flask.palletsprojects.com/en/3.0.x/">
      <img src="https://img.shields.io/badge/flask-3.0.3-blue?logo=flask" alt="Flask 3.0.3">
   </a>
   <a href="https://docs.pytest.org/en/8.2.x/">
      <img src="https://img.shields.io/badge/pytest-8.2.2-blue?logo=pytest" alt="pytest 8.2.2">
   </a>
</div>

<br>

"建築工地危險檢測系統" 是一款旨在提升建築工地安全的人工智慧工具。利用 YOLOv8 模型進行物體偵測，此系統能夠辨識潛在的危險，例如未戴安全帽的工人、未穿安全背心的工人、靠近機具的工人以及靠近車輛的工人。系統採用後處理演算法來提升偵測的準確性。設計用於實時環境部署，能夠即時分析並發出警報以應對偵測到的危險。

<br>
<br>

<div align="center">
   <img src="./assets/images/hazard-detection.png" alt="diagram" style="width: 100%;">
</div>

<br>

## 操作說明

在運行應用程式之前，您需要配置系統，指定視頻流的詳細信息和其他參數，這些信息需要在 YAML 配置文件中進行設置。示例配置文件 `configuration.yaml` 應該看起來像這樣：

```yaml
# 這是一個視頻配置列表
- video_url: "rtsp://example1.com/stream"  # 視頻的 URL
   image_name: "cam1"  # 圖像的名稱
   label: "label1"  # 視頻的標籤
   model_key: "yolov8n"  # 視頻使用的模型鍵
   line_token: "token1"  # 用於通知的 Line Token
   run_local: True  # 本地運行物件檢測
- video_url: "rtsp://example2.com/stream"
   image_name: "cam2"
   label: "label2"
   model_key: "yolov8n"
   line_token: "token2"
   run_local: True
```

數組中的每個對象代表一個視頻流配置，包含以下字段：

- `video_url`: 現場視頻流的 URL。這可以包括：
   - 監控流
   - RTSP
   - 副流
   - YouTube 視頻或直播
   - Discord
- `image_name`: 分配給圖像或攝影機的名稱。
- `label`: 分配給視頻流的標籤。
- `model_key`: 用於機器學習模型的鍵標識符。
- `line_token`: 用於發送通知的 LINE 訊息 API Token。
- `run_local`: 布爾值，指示是否在本地運行物件檢測。

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
      docker-compose up --build
      ```

   4. 使用特定的配置文件運行主應用程序，使用以下命令：
      ```bash
      docker-compose run main-application python main.py --config /path/in/container/configuration.yaml
      ```
      將 `/path/in/container/configuration.yaml` 替換為容器內配置文件的實際路徑。

   5. 停止服務，使用以下命令：
      ```bash
      docker-compose down
      ```

</details>

<details>
   <summary>Python</summary>

   ### 使用 Python

   要在終端運行危險檢測系統，您需要在機器上安裝 Python。按照以下步驟來啟動系統：

   1. 將存儲庫克隆到本地機器。
      ```bash
      git clone https://github.com/yihong1120/Construction-Hazard-Detection.git
      ```

   2. 進入克隆的目錄。
      ```bash
      cd Construction-Hazard-Detection
      ```

   3. 安裝所需的軟體包：
      ```bash
      pip install -r requirements.txt
      ```

   4. 安裝並啟動 MySQL 服務：
      ```bash
      sudo apt install mysql-server
      sudo systemctl start mysql.service
      ```

   5. 設置用戶帳戶和密碼。使用以下命令啟動用戶管理 API：
      ```bash
      gunicorn -w 1 -b 0.0.0.0:8000 "examples.user_management.app:user-managements-app"
      ```
      建議使用 Postman 應用程式與 API 進行互動。

   6. 要運行物體檢測 API，使用以下命令：
      ```bash
      gunicorn -w 1 -b 0.0.0.0:8001 "examples.YOLOv8_server_api.app:YOLOv8-server-api-app"
      ```

   7. 使用特定的配置文件運行主應用程序，使用以下命令：
      ```bash
      python3 main.py --config /path/to/your/configuration.yaml
      ```
      將 `/path/to/your/configuration.yaml` 替換為您的配置文件的實際路徑。

   8. 要啟動串流 Web 服務，執行以下命令：
      ```bash
      gunicorn -w 1 -k eventlet -b 127.0.0.1:8002 "examples.streaming_web.app:streaming-web-app"
      ```

</details>

## 附加信息

- 系統日誌可在 Docker 容器內部訪問，可用於調試目的。
- 如果啟用，檢測到的輸出圖像將保存到指定的輸出路徑。
- 如果檢測到危險，將在指定小時通過 LINE 消息 API 發送通知。

### 注意事項
- 確保 `Dockerfile` 存在於項目的根目錄中，並根據您的應用程序的要求進行了正確配置。
- `-p 8080:8080` 標誌將容器的 8080 端口映射到主機機器的 8080 端口，允許您通過主機的 IP 地址和端口號訪問應用程序。

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

<details>
   <summary>檢測模型</summary>

   | Model   | size<br><sup>(pixels) | mAP<sup>val<br>50 | mAP<sup>val<br>50-95 | params<br><sup>(M) | FLOPs<br><sup>(B) |
   | ------- | --------------------- | ------------------ | ------------------ | ----------------- | ----------------- |
   | YOLOv8n | 640                   | //                 | //                 | 3.2               | 8.7               |
   | YOLOv8s | 640                   | //                 | //                 | 11.2              | 28.6              |
   | YOLOv8m | 640                   | //                 | //                 | 25.9              | 78.9              |
   | YOLOv8l | 640                   | //                 | //                 | 43.7              | 165.2             |
   | YOLOv8x | 640                   | 82.9               | 60.9               | 68.2              | 257.8             |

</details>

<br>

我們的全面數據集確保模型能夠識別建築環境中常見的各種潛在危險。

## 貢獻
我們歡迎對此項目的貢獻。請按照以下步驟操作：
1. 分叉存儲庫。
2. 進行更改。
3. 提交一個清晰描述您改進的拉取請求。

## 開發路線圖
- [x] 數據收集和預處理。
- [x] 使用建築工地數據訓練 YOLOv8 模型。
- [x] 開發後處理技術以提高準確性。
- [x] 實施實時分析和警報系統。
- [x] 在模擬環境中進行測試和驗證。
- [x] 在實際建築工地進行現場測試。
- [x] 根據用戶反饋進行持續維護和更新。

## 授權
此項目根據 [AGPL-3.0](LICENSE.md) 授權。
