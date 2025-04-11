
🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# 串流 Web 後端

此目錄包含一個以 FastAPI 建立的串流網頁應用程式之後端邏輯，使用者可以上傳影格（圖片），並透過 WebSocket 即時取得這些影格。

下文將概述 `examples/streaming_web/backend/` 中的每個檔案，說明如何執行此應用程式，並提供在正式環境使用時的最佳實踐與建議。

## 目錄結構

```
examples/streaming_web/backend/
├── __init__.py
├── app.py
├── redis_service.py
├── routers.py
├── schemas.py
└── utils.py
```

### `app.py`
- **用途**：定義並配置主要的 FastAPI 應用程式。
- **重點**：
  - 使用自訂的 lifespan（`global_lifespan`）建立 `FastAPI` 實例，以管理應用程式的啟動與關閉流程（請參考 `examples.auth.lifespan`）。
  - 設置 CORS 中介軟體（middleware），允許跨來源的請求（在正式環境中應適度限制）。
  - 引入下列路由（routers）：
    - **身份驗證**（`auth_router`）
    - **使用者管理**（`user_management_router`）
    - **串流網頁服務**（`streaming_web_router`）
  - 提供一個 `main()` 函式，並使用 `uvicorn.run()` 啟動應用程式。

**使用方式**：

```bash
uvicorn examples.streaming_web.backend.app:app --host 127.0.0.1 --port 8000
```

或直接呼叫 `main()` 函式：

```bash
python examples/streaming_web/backend/app.py
```

### `redis_service.py`
- **用途**：與 Redis 互動，處理影格與其中繼資料的儲存與讀取。
- **主要函式**：
  1. `scan_for_labels(rds)`：掃描 Redis 中的鍵，找出不同的標籤（label）。標籤使用 base64 編碼儲存在鍵值中，並會被解碼。若標籤字串包含 `'test'`，則不納入。
  2. `get_keys_for_label(rds, label)`：取得與指定 label 對應的所有 Redis 鍵。
  3. `fetch_latest_frames(rds, last_ids)`：為 `last_ids` 中每個鍵擷取最新影格；若發現新影格，就回傳影格與相關中繼資料。
  4. `fetch_latest_frame_for_key(rds, redis_key, last_id)`：與上類似，但僅針對單一鍵（key）抓取最近的影格，忽略比 `last_id` 更舊的影格。
  5. `store_to_redis(rds, site, stream_name, frame_bytes, warnings_json, cone_polygons_json, ...)`：將上傳的影格及其中繼資料存入 Redis（使用 Redis Streams 的 `xadd`）。

**注意事項**：
- 透過 base64（`Utils.encode()`）編碼與解碼標籤與串流名稱。
- 使用 Redis Streams 時，`maxlen=10`，代表若同一串流超過 10 筆資料，舊資料會被裁剪。
- 定義了 `DELIMITER = b'--IMG--'` 作為在 WebSocket 傳輸時，JSON 中繼資料與二進位影格的分隔符。

### `routers.py`
- **用途**：定義提供上傳影格（`POST /frames`）、讀取標籤（`GET /labels`）以及 WebSocket 串流影格的 FastAPI 路由。
- **端點**：
  1. `GET /labels`：每分鐘限 60 次呼叫，透過 `scan_for_labels` 從 Redis 取得所有標籤。
  2. `POST /frames`：經過 JWT 驗證之後才能呼叫，允許使用者上傳圖片檔案及相關 JSON 資料（警告、標註多邊形等），並儲存在 Redis。
  3. `WebSocket /ws/labels/{label}`：針對單一 label 下的 **所有**串流鍵提供近即時影格推送。
  4. `WebSocket /ws/stream/{label}/{key}`：採用 **拉式**機制（pull-based）的串流；前端以 `{"action": "pull"}` 指令向伺服器索取最新影格，也支援 `{"action": "ping"}` 做 keepalive。

**WebSocket 互動方式**：
- **`/ws/labels/{label}`**：
  - 透過定期檢查 Redis，找出此 label 相關的所有鍵（keys），並持續發送最新影格給用戶端。
  - 影格的中繼資料會以 JSON 表示；原始影格則以二進位送出，並以 `DELIMITER` 作為分隔符。
  - 若找不到任何鍵，則回傳錯誤訊息並關閉連線。

- **`/ws/stream/{label}/{key}`**：
  - 透過 JSON 命令進行互動：
    - `ping` → 回傳 `pong`。
    - `pull` → 取出 Redis 中最新影格，若存在則送回給用戶端。
    - 任何未知指令 → 回傳錯誤訊息。

### `schemas.py`
- **用途**：使用 Pydantic 模型定義結構化的請求與回應格式。
- **主要模型**：
  1. `LabelListResponse`：回傳一個 JSON 格式的標籤清單。
  2. `FramePostResponse`：表示上傳影格的結果（例如 `"ok"`）與文字訊息。

藉由這些模型，路由可確保回傳格式一致並有明確的類型定義。

### `utils.py`
- **用途**：提供雜項的輔助函式，包括 base64 編碼、解碼，以及（未在主要路由中使用但可擴充）透過 WebSocket 傳送影格資料的函式。
- **主要函式**：
  - `encode(value)`：將字串做 URL-safe Base64 編碼。
  - `is_base64(value)`：檢查字串是否符合 URL-safe Base64 格式。
  - `decode(value)`：若字串為有效的 Base64，則解碼；否則回傳原字串。
  - `send_frames(websocket, label, updated_data)`：將已結構化的資料以 JSON 形式透過 `WebSocket` 傳送給用戶端，並包含一個標籤欄位以便識別。

## 執行應用程式

1. **安裝依賴套件**
   確保已安裝 FastAPI、Uvicorn、Redis（或 Python Redis 客戶端如 `redis-py`）以及其他必需的套件：
   ```bash
   pip install fastapi uvicorn redis fastapi-limiter
   ```

2. **啟動 Redis**
   您需要在本機或可連線的地方執行 Redis 伺服器，例如：
   ```bash
   redis-server
   ```
   或使用 Docker：
   ```bash
   docker run -p 6379:6379 redis
   ```

3. **啟動 FastAPI 應用程式**
   在專案根目錄下，執行：
   ```bash
   uvicorn examples.streaming_web.backend.app:app --host 127.0.0.1 --port 8000
   ```
   然後可在瀏覽器中開啟 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 查看自動生成的 OpenAPI 文件。

## 測試

單元測試與整合測試位於 `tests/` 目錄中，例如：

```bash
pytest --cov=examples.streaming_web.backend --cov-report=term-missing
```

主要測試檔：
- **`redis_service_test.py`**：驗證影格在 Redis 中存取的正確性。
- **`routers_test.py`**：檢驗各端點與 WebSocket 功能，使用 FastAPI 的 `TestClient`。

## 正式環境注意事項

1. **CORS 限制**：
   正式部署時，建議用特定網域取代 `allow_origins=['*']` 以加強安全性。

2. **速率限制（Rate Limiting）**：
   透過 FastAPI-Limiter 對部份端點進行限制。請檢視與調整次數/秒數等參數，避免效能瓶頸或資源耗用過高。

3. **身份驗證**：
   例如 `POST /frames` 這類端點使用自 `examples.auth.jwt_config` 匯入的 `jwt_access` 進行 JWT 驗證。若在正式環境使用，需確保金鑰與驗證流程更安全。

4. **安全連線**：
   若要將服務暴露在網際網路上，請配置 TLS/SSL，以確保 API 呼叫與 WebSocket 連線不易被竊聽或篡改。

5. **水平擴充**：
   - 若要提供更高吞吐量，可使用 Gunicorn + Uvicorn workers 並搭配負載平衡器。
   - 若影像量龐大，可使用 Redis Streams 分散式叢集來擴充負載能力。

## 聯繫與更多資訊

若想進一步瞭解身份驗證流程（`examples/auth`）、使用者管理，或進階的串流技術（如 WebRTC 或分段串流），請參考完整的程式庫或與開發團隊聯繫。
