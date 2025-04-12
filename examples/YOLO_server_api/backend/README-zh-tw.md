
🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLO 伺服器後端 (Backend)

此目錄包含一個基於 FastAPI 的後端，能透過 YOLO 模型進行物件偵測，並提供可選的模型檔案管理功能，同時結合使用者身份驗證與授權（透過 `examples.auth`）。

## 目錄

1. [概覽](#概覽)
2. [功能特點](#功能特點)
3. [資料夾結構](#資料夾結構)
4. [安裝與設定](#安裝與設定)
5. [執行伺服器](#執行伺服器)
6. [端點說明](#端點說明)
   - [偵測端點](#偵測端點)
   - [模型管理端點](#模型管理端點)
7. [身份驗證與使用者管理整合](#身份驗證與使用者管理整合)
   - [Token 取得與更新](#token-取得與更新)
8. [其他注意事項](#其他注意事項)

## 概覽

此後端主要提供以下能力：

- **YOLO 模型物件偵測**：詳見 [`detection.py`](./detection.py)。
- **模型管理**：可於執行期間上傳或更新 YOLO `.pt` 檔案，詳見 [`model_files.py`](./model_files.py) 與 [`models.py`](./models.py)。
- **FastAPI 路由**：使用 [`routers.py`](./routers.py) 進行路由切分，結構清晰。
- 參考 `examples.auth` 模組，提供：
  - **JWT 驗證機制**
  - **角色（Role）授權檢查**
  - **自訂的速率限制**（rate limiting）

## 功能特點

- **多種 YOLO 模型**：可動態管理不同 YOLO 權重檔，如 `yolo11n`, `yolo11s` 等。
- **Watchdog 監控**：若 `.pt` 檔有變動，會自動重新載入該模型。
- **非同步偵測管線**：有效率地處理上傳影像及產出結果。
- **選擇性重疊剔除**：後處理階段能移除重疊或被包覆的框。

## 資料夾結構

```
examples/YOLO_server_api/backend/
├── app.py            # 主要的 FastAPI 應用程式
├── detection.py      # 核心偵測邏輯 (影像處理、框選後處理等)
├── model_files.py    # 負責上傳/取得 .pt 模型檔的邏輯
├── models.py         # DetectionModelManager & Watchdog 相關自動重新載入
├── routers.py        # FastAPI 路由，包含偵測與模型管理
└── README.md         # 你正在閱讀的檔案
```

## 安裝與設定

1. **安裝依賴套件**
   確保已安裝所有需求 (包含 `fastapi`, `uvicorn`, `watchdog`, `sahi` 以及 YOLO 相關依賴)。
   如使用此專案的 `requirements.txt`，可執行：
   ```bash
   pip install -r requirements.txt
   ```

2. **模型檔**
   - 預設 `.pt` 檔會放在 `models/pt/` (以目前執行目錄為基準)。
   - [`DetectionModelManager`](./models.py) 會監控該目錄並於檔案變更時自動重新載入模型。

3. **(選用) 環境變數 / .env**：若有使用（例如 Redis、資料庫、JWT secret），可參考 `examples.auth`，本目錄並不直接處理。

## 執行伺服器

1. **移動至** 專案根目錄或適合的目錄。
2. **啟動** 應用程式：

   ```bash
   python examples/YOLO_server_api/backend/app.py
   ```
   或使用 uvicorn：
   ```bash
   uvicorn examples.YOLO_server_api.backend.app:app --host 127.0.0.1 --port 8000
   ```

3. **驗證**：瀏覽器開啟 <http://127.0.0.1:8000> 確認伺服器已啟動。

## 端點說明

主要端點定義於 [`routers.py`](./routers.py)，以下簡要說明：

### 偵測端點

- **`POST /detect`**
  上傳影像以進行 YOLO 偵測。

  **範例 cURL**：
  ```bash
  curl -X POST "http://127.0.0.1:8000/detect" \
       -H "Authorization: Bearer <ACCESS_TOKEN>" \
       -F "image=@/path/to/image.jpg" \
       -F "model=yolo11n"
  ```
  **回傳**：JSON 陣列，每個元素是 `[x1, y1, x2, y2, confidence, label_id]`。

### 模型管理端點

- **`POST /model_file_update`**
  上傳新的 `.pt` 檔以更新指定 YOLO 模型。

  **範例 cURL**：
  ```bash
  curl -X POST "http://127.0.0.1:8000/model_file_update" \
       -H "Authorization: Bearer <ACCESS_TOKEN_WITH_ADMIN_OR_MODEL_MANAGE_ROLE>" \
       -F "model=yolo11n" \
       -F "file=@/path/to/new_best_yolo11n.pt"
  ```

- **`POST /get_new_model`**
  若伺服器端模型有更新，回傳 base64 編碼的模型檔。

  **範例 cURL**：
  ```bash
  curl -X POST "http://127.0.0.1:8000/get_new_model" \
       -H "Content-Type: application/json" \
       -d '{
            "model": "yolo11n",
            "last_update_time": "2025-01-01T00:00:00"
           }'
  ```

## 身份驗證與使用者管理整合

- **身份驗證**：此後端使用 `examples.auth.jwt_config.jwt_access` 進行 JWT 驗證。
- **角色檢查**：部份端點（如上傳模型）需要 `admin` 或 `model_manage` 角色，透過 `examples.auth.jwt_config` 與 FastAPI 相依注入實現。
- **速率限制**：偵測端點使用 `examples.auth.cache` 中的 `custom_rate_limiter`。

若想關閉驗證或速率限制，可直接在 `routers.py` 移除相關依賴或改為自訂邏輯。

### Token 取得與更新

1. **登入 (Login)**：取得 `ACCESS_TOKEN` 與 `REFRESH_TOKEN`
   ```
   POST http://127.0.0.1:8000/login
   ```
   **請求範例 (JSON)**：
   ```json
   {
     "username": "your_username",
     "password": "your_password"
   }
   ```
   **回應範例**：
   ```json
   {
     "access_token": "<JWT_ACCESS_TOKEN>",
     "refresh_token": "<JWT_REFRESH_TOKEN>",
     "role": "user",
     "username": "your_username",
     "user_id": 1
   }
   ```

2. **使用 Access Token**：之後呼叫（如 `/detect`）時，於標頭帶入：
   ```
   Authorization: Bearer <ACCESS_TOKEN>
   ```
   在請求標頭中。

3. **Refresh Token**：當你的 `ACCESS_TOKEN` 過期時，可呼叫：
   ```
   POST http://127.0.0.1:8000/refresh
   ```
   **請求範例 (JSON)**：
   ```json
   {
     "refresh_token": "<YOUR_REFRESH_TOKEN>"
   }
   ```
   **回應範例**：
   ```json
   {
     "access_token": "<NEW_JWT_ACCESS_TOKEN>",
     "refresh_token": "<NEW_JWT_REFRESH_TOKEN>",
     "message": "Token refreshed successfully."
   }
   ```
   將新 `ACCESS_TOKEN` 放到 `Authorization`，或同步更新舊的 `REFRESH_TOKEN`。

## 其他注意事項

- **模型重新載入**：[`DetectionModelManager`](./models.py) 使用 Watchdog 監控 `.pt` 檔。若檔案被取代或更動，會自動重新載入。
- **框選後處理**：[`detection.py`](./detection.py) 範例展示如何移除重疊或被包覆的框（如 `hardhat` vs. `no_hardhat`）。
- **效能考量**：若需要高流量，可調整併發數量與 GPU 設定。程式預設 `device='cuda:0'`（若 GPU 可用）。
- **部署**：可使用 Docker，或搭配 Nginx 反向代理，或任何 ASGI 主機環境。若依賴檔案更新（`models/pt/`），記得確保容器或伺服器檔案路徑一致。
