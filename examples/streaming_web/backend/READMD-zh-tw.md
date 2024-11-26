
🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Streaming Web 後端範例

本節提供了一個 Streaming Web 後端應用程式的範例實作，旨在促進即時攝影機畫面和更新。此指南提供了如何使用、配置和理解此應用程式功能的資訊。

## 使用方法

1. **啟動伺服器：**
    ```sh
    python app.py
    ```

    或

    ```sh
    uvicorn examples.streaming_web.backend.app:sio_app --host 127.0.0.1 --port 8000
    ```

2. **打開您的網頁瀏覽器並導航至：**
    ```sh
    http://localhost:8000
    ```

## 功能

- **即時串流**：顯示即時攝影機畫面並自動更新。
- **WebSocket 整合**：利用 WebSocket 進行高效的即時通訊。
- **動態內容加載**：自動更新攝影機圖像而無需刷新頁面。
- **響應式設計**：適應各種螢幕尺寸，提供無縫的使用者體驗。
- **可自訂佈局**：使用 CSS 調整佈局和樣式。
- **速率限制**：實施速率限制以防止 API 濫用。
- **錯誤處理**：全面的錯誤處理，應對各種異常情況。
- **靜態文件服務**：提供 HTML、CSS 和 JavaScript 等靜態文件服務。

## 配置

應用程式可以通過以下文件進行配置：

- **app.py**：啟動伺服器並定義路由、中介軟體和 WebSocket 整合的主應用程式文件。
- **routes.py**：定義網頁路由及其相應的處理程序，包括獲取標籤、處理 WebSocket 連接和處理 webhook 的 API 端點。
- **sockets.py**：管理 WebSocket 連接，處理連接、斷開和更新等事件。還包括更新圖像的背景任務。
- **utils.py**：應用程式的實用函數，包括編碼/解碼值和通過 WebSocket 發送畫面數據。
- **.env**：配置環境變數，例如 Redis 連接詳情。

## 文件概述

### app.py
應用程式的主要入口，啟動伺服器並設置路由、中介軟體和 WebSocket 整合。

- **生命週期管理**：使用 `@asynccontextmanager` 管理應用程式啟動和關閉任務，例如初始化和關閉 Redis 連接。
- **CORS 中介軟體**：配置 CORS 以允許跨域請求。
- **靜態文件**：從指定目錄提供靜態文件服務。
- **Socket.IO 整合**：初始化並配置 Socket.IO 伺服器以進行即時通訊。

### routes.py
定義各種網頁路由及其相應的請求處理程序。

- **API 端點**：
  - `/api/labels`：從 Redis 獲取可用標籤。
  - `/api/ws/labels/{label}`：WebSocket 端點，用於串流特定標籤的更新畫面。
  - `/api/ws/stream/{label}/{key}`：WebSocket 端點，用於串流單個攝影機的數據。
  - `/api/webhook`：處理傳入的 webhook 請求。
  - `/api/upload`：處理文件上傳並將其保存到指定文件夾。

### sockets.py
管理 WebSocket 連接，處理連接、斷開和更新等事件。還包括更新圖像的背景任務。

- **Socket.IO 事件**：
  - `connect`：處理新客戶端連接。
  - `disconnect`：處理客戶端斷開連接。
  - `error`：處理 WebSocket 通訊過程中的錯誤。
- **背景任務**：
  - `update_images`：定期獲取並向連接的客戶端發送更新的圖像。

### utils.py
包含應用程式中使用的各種實用函數。

- **RedisManager**：管理異步 Redis 操作，用於獲取標籤、鍵和圖像數據。
  - `get_labels`：從 Redis 鍵中獲取唯一的標籤和流名稱組合。
  - `get_keys_for_label`：檢索與給定標籤-流名稱模式匹配的 Redis 鍵。
  - `fetch_latest_frames`：獲取每個 Redis 流的最新畫面。
  - `fetch_latest_frame_for_key`：獲取特定 Redis 鍵的最新畫面和警告。
- **Utils**：包含靜態方法，用於編碼/解碼值和通過 WebSocket 發送畫面數據。
  - `encode`：將值編碼為 URL 安全的 Base64 字符串。
  - `decode`：解碼 URL 安全的 Base64 字符串。
  - `send_frames`：將最新的畫面發送給 WebSocket 客戶端。

### .env
包含配置 Redis 連接的環境變數，例如 `REDIS_HOST`、`REDIS_PORT` 和 `REDIS_PASSWORD`。

## 環境變數

應用程式使用以下環境變數進行配置：

- `REDIS_HOST`：Redis 伺服器的主機名（默認：`127.0.0.1`）。
- `REDIS_PORT`：Redis 伺服器的端口（默認：`6379`）。
- `REDIS_PASSWORD`：Redis 伺服器的密碼（默認：空）。

請確保檢查並調整相應文件中的配置設置，以適應您的具體需求。