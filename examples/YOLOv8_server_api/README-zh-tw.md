🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLOv8 Server API 範例

本部分提供一個 YOLOv8 Server API 的範例實作，旨在利用 YOLOv8 模型進行物件檢測。此指南提供有關如何使用、配置和了解該 API 功能的信息。

## 使用方式

1. **啟動伺服器：**
    ```sh
    python app.py
    ```

    或者

    ```sh
    gunicorn -w 1 -b 0.0.0.0:8000 "examples.YOLOv8_server_api.app:app"
    ```  

2. **向 API 發送請求：**
    - 可以使用 `curl`、Postman 或瀏覽器來向伺服器發送請求。
    - 使用 `curl` 的範例：
        ```sh
        curl -X POST -F "file=@path/to/your/image.jpg" http://localhost:8000/detect
        ```

## 功能

- **身份驗證**：使用身份驗證機制來保護 API。
- **快取**：通過快取檢測結果來提高性能。
- **模型下載**：自動下載和加載 YOLOv8 模型。
- **配置**：靈活的配置選項來定制 API。
- **物件檢測**：使用 YOLOv8 模型對上傳的圖像進行物件檢測。
- **錯誤處理**：強健的錯誤處理機制來管理各種情況。

## 配置

可以通過 `config.py` 文件來配置 API。以下是一些可用的主要配置選項：

- **伺服器設置**：
  - `HOST`：運行伺服器的主機名。默認為 `0.0.0.0`。
  - `PORT`：運行伺服器的端口。默認為 `8000`。

- **模型設置**：
  - `MODEL_PATH`：YOLOv8 模型文件的路徑。
  - `CONFIDENCE_THRESHOLD`：物件檢測的置信度閾值。

- **快取設置**：
  - `CACHE_ENABLED`：啟用或禁用快取。默認為 `True`。
  - `CACHE_EXPIRY`：快取過期時間（以秒為單位）。默認為 `3600`。

- **身份驗證設置**：
  - `AUTH_ENABLED`：啟用或禁用身份驗證。默認為 `True`。
  - `SECRET_KEY`：JWT 身份驗證的密鑰。

## 文件概述

- **app.py**：啟動伺服器並定義 API 端點的主應用文件。
- **auth.py**：處理身份驗證機制。
- **cache.py**：實現快取功能。
- **config.py**：包含 API 的配置設置。
- **detection.py**：使用 YOLOv8 模型進行物件檢測。
- **model_downloader.py**：處理 YOLOv8 模型的下載和加載。
- **models.py**：定義數據模型和結構。
- **security.py**：實現安全功能。

請確保檢查並調整 `config.py` 中的配置設置以適應您的具體需求。

歡迎通過在 GitHub 存儲庫提交問題或拉取請求來為本項目做出貢獻。