🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

<img src="./assets/images/project_graphics/banner.gif" alt="AI-Driven Construction Safety Banner" style="width: 100%;">

<div align="center">
    <a href="examples/Model-Server">模型伺服器</a> | <a href="examples/Stream-Web">串流網頁</a> | <a href="examples/User-Management">用戶管理</a> | <a href="examples/YOLOv8-Data-Augmentation">YOLOv8 數據增強</a> | <a href="examples/YOLOv8-Evaluation">YOLOv8 評估</a> | <a href="examples/YOLOv8-Train">YOLOv8 訓練</a>
</div>

"建築工地危險檢測系統" 是一個以人工智能驅動的工具，旨在提高建築工地的安全性。利用 YOLOv8 模型進行物體檢測，此系統能夠識別潛在的危險，如高空重物和鋼管。對訓練好的模型進行後處理以提高準確性。該系統設計用於實時環境中，提供對檢測到的危險的即時分析和警告。

## 配置

在運行應用程序之前，您需要通過在 JSON 配置文件中指定視頻流和其他參數的詳細信息來配置系統。一個示例配置文件 `configuration.json` 應該如下所示：

```json
[
    {
        "video_url": "rtsp://example1.com/stream",
        "api_url": "http://localhost:5000/detect",
        "model_key": "yolov8l",
        "line_token": "token1"
    },
    {
        "video_url": "rtsp://example2.com/stream",
        "api_url": "http://localhost:5000/detect",
        "model_key": "yolov8l",
        "line_token": "token2"
    }
]
```

數組中的每個對象代表一個視頻流配置，包含以下字段：

- `video_url`: 直播視頻流的 URL。
- `api_url`: 機器學習模型服務器 API 端點的 URL。
- `model_key`: 要使用的機器學習模型的鍵標識符。
- `line_token`: 用於發送通知的 LINE 消息 API 令牌。

## 使用方法

要運行危險檢測系統，您需要在機器上安裝 Docker 和 Docker Compose。按照以下步驟啟動系統：

1. 克隆存儲庫到您的本地機器。
```
git clone https://github.com/yihong1120/Construction-Hazard-Detection.git
```
2. 導航到克隆的目錄。
```
cd Construction-Hazard-Detection
```
3. 使用 Docker Compose 構建並運行服務：

```bash
docker-compose up --build
```

4. 要使用特定配置文件運行主應用程序，請使用以下命令：

```bash
docker-compose run main-application python main.py --config /path/in/container/configuration.json
```

將 `/path/in/container/configuration.json` 替換為容器內配置文件的實際路徑。

5. 要停止服務，請使用以下命令：

```bash
docker-compose down
```

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

我們的全面數據集確保模型能夠識別建築環境中常見的各種潛在危險。

## 附加信息

- 系統日誌可在 Docker 容器內部訪問，可用於調試目的。
- 如果啟用，檢測到的輸出圖像將保存到指定的輸出路徑。
- 如果檢測到危險，將在指定小時通過 LINE 消息 API 發送通知。

### 注意事項
- 確保 `Dockerfile` 存在於項目的根目錄中，並根據您的應用程序的要求進行了正確配置。
- `-p 8080:8080` 標誌將容器的 8080 端口映射到主機機器的 8080 端口，允許您通過主機的 IP 地址和端口號訪問應用程序。

有關 Docker 使用和命令的更多信息，請參閱 [Docker 文檔](https://docs.docker.com/)。

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
