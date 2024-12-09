
🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLO 伺服器 API 與使用者及模型管理系統

本專案提供一個完整的 YOLO 伺服器 API，結合了使用者管理系統與模型管理系統。伺服器能夠執行物件檢測功能，並提供強大的使用者及模型管理功能。

---

## 功能特點

### YOLO 伺服器 API
- **物件檢測**：使用 YOLO 模型進行影像中的物件檢測。
- **快取功能**：透過快取檢測結果提升效能。
- **錯誤處理**：強大的錯誤處理機制，提供無縫的使用體驗。
- **身份驗證**：基於角色的 JWT 身份驗證確保 API 安全性。

### 使用者管理系統
- **新增使用者**：建立新使用者帳戶並指定角色。
- **刪除使用者**：從系統中移除使用者。
- **更新使用者**：更新使用者的名稱及密碼。
- **啟用/停用帳戶**：管理使用者帳戶的啟用狀態。
- **角色管理**：支持基於角色的權限分配（`admin`、`model_manage`、`user`、`guest`）。

### 模型管理系統
- **模型上傳**：動態上傳與更新 YOLO 模型。
- **模型檢索**：下載最新版本的 YOLO 模型。
- **版本管理**：透過時間戳追蹤模型更新。
- **驗證功能**：確保上傳的模型符合系統需求。

---

## 快速開始

### 1. 安裝依賴套件
確保您已安裝 Python，然後執行以下命令安裝所需的依賴：
```sh
pip install -r requirements.txt
```

### 2. 配置應用程式
更新 `config.py` 或 `.env` 檔案中的設定：
- 設定資料庫連線 URI（如 SQLite 或 PostgreSQL）。
- 配置 JWT 秘鑰及其他環境變數。

### 3. 啟動伺服器
使用以下指令啟動伺服器：
```sh
uvicorn examples.YOLO_server_api.app:sio_app --host 127.0.0.1 --port 8000
```

### 4. 發送請求
使用工具如 `curl`、Postman 或瀏覽器與 API 互動。

---

## API 概覽

### YOLO 伺服器 API

- **`POST /detect`**：對上傳的影像執行物件檢測。
  - **參數**：
    - `file`（影像）：要檢測的影像檔案。
    - `model`（字串）：使用的 YOLO 模型（預設：`yolo11n`）。
  - **範例**：
    ```sh
    curl -X POST -F "file=@path/to/image.jpg" -F "model=yolo11n" http://localhost:8000/detect
    ```

---

### 使用者管理 API

- **`POST /add_user`**：新增使用者。
  - **參數**：
    - `username`（字串）：使用者名稱。
    - `password`（字串）：密碼。
    - `role`（字串）：角色（`admin`、`model_manage`、`user`、`guest`）。
  - **範例**：
    ```sh
    curl -X POST -H "Content-Type: application/json" \
         -d '{"username":"admin","password":"securepassword","role":"admin"}' \
         http://localhost:8000/add_user
    ```

- **`DELETE /delete_user/<username>`**：刪除使用者。
  - **參數**：
    - `username`（字串）：要刪除的使用者名稱。
  - **範例**：
    ```sh
    curl -X DELETE http://localhost:8000/delete_user/admin
    ```

- **`PUT /update_username`**：更新使用者名稱。
  - **參數**：
    - `old_username`（字串）：現有使用者名稱。
    - `new_username`（字串）：新使用者名稱。
  - **範例**：
    ```sh
    curl -X PUT -H "Content-Type: application/json" \
         -d '{"old_username":"admin","new_username":"superadmin"}' \
         http://localhost:8000/update_username
    ```

- **`PUT /update_password`**：更新使用者密碼。
  - **參數**：
    - `username`（字串）：使用者名稱。
    - `new_password`（字串）：新密碼。
  - **範例**：
    ```sh
    curl -X PUT -H "Content-Type: application/json" \
         -d '{"username":"admin","new_password":"newsecurepassword"}' \
         http://localhost:8000/update_password
    ```

- **`PUT /set_user_active_status`**：設置使用者啟用狀態。
  - **參數**：
    - `username`（字串）：使用者名稱。
    - `is_active`（布林值）：啟用狀態（`true` 或 `false`）。
  - **範例**：
    ```sh
    curl -X PUT -H "Content-Type: application/json" \
         -d '{"username":"admin","is_active":false}' \
         http://localhost:8000/set_user_active_status
    ```

---

### 模型管理 API

- **`POST /model_file_update`**：更新 YOLO 模型檔案。
  - **參數**：
    - `model`（字串）：模型名稱。
    - `file`（檔案）：模型檔案。
  - **範例**：
    ```sh
    curl -X POST -F "model=yolo11n" -F "file=@path/to/model.pt" http://localhost:8000/model_file_update
    ```

- **`POST /get_new_model`**：檢索最新的 YOLO 模型檔案。
  - **參數**：
    - `model`（字串）：模型名稱。
    - `last_update_time`（ISO 8601 字串）：上次更新時間。
  - **範例**：
    ```sh
    curl -X POST -H "Content-Type: application/json" \
         -d '{"model":"yolo11n", "last_update_time":"2023-01-01T00:00:00"}' \
         http://localhost:8000/get_new_model
    ```

---

## 配置選項

### YOLO 伺服器設定
定義於 `config.py`：
- `MODEL_PATH`：YOLO 模型檔案路徑。
- `CONFIDENCE_THRESHOLD`：物件檢測結果的信心閾值。
- `CACHE_ENABLED`：啟用/停用檢測結果快取。
- `AUTH_ENABLED`：啟用/停用 JWT 身份驗證。

### 資料庫設定
在 `app.py` 中設定 `SQLALCHEMY_DATABASE_URI` 來匹配您的資料庫：
```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'
```

---

## 檔案結構

- **`app.py`**：主要伺服器應用程式。
- **`routers.py`**：定義 YOLO 伺服器、使用者管理及模型管理的 API。
- **`auth.py`**：處理基於 JWT 的身份驗證。
- **`cache.py`**：實現快取功能以提升效能。
- **`detection.py`**：執行 YOLO 模型的物件檢測。
- **`user_operation.py`**：處理使用者管理邏輯。
- **`model_files.py`**：處理模型檔案的更新與檢索。
- **`config.py`**：應用程式的核心配置檔案。

---

## 注意事項

- **安全性**：確保將敏感的憑證（例如 JWT 秘鑰、資料庫密碼）安全地儲存在環境變數中。
- **測試**：使用 `tests/` 資料夾內的單元測試檔案來驗證實作的正確性。
