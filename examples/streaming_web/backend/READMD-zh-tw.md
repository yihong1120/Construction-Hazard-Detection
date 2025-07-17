
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
  - 設置 CORS 中介軟體，允許跨來源的請求（正式環境應適度限制）。
  - 僅引入串流網頁服務（`streaming_web_router`）。
  - 提供 `main()` 函式，使用 `uvicorn.run()` 啟動應用程式。
  - 所有程式皆有型別註解與 Google-style docstring，並採英式英文註解。

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
  1. `scan_for_labels(rds: redis.Redis) -> list[str]`：掃描 Redis 鍵找出不同標籤（key 以 base64 編碼，回傳時解碼），排除含 'test'。
  2. `get_keys_for_label(rds: redis.Redis, label: str) -> list[str]`：取得指定 label 對應的所有 Redis 鍵。
  3. `fetch_latest_frames(rds: redis.Redis, last_ids: dict[str, str]) -> list[dict[str, str | bytes | int]]`：為 `last_ids` 中每個鍵擷取最新影格，回傳中繼資料與原始影格 bytes。
  4. `fetch_latest_frame_for_key(rds: redis.Redis, redis_key: str, last_id: str) -> dict[str, str | bytes | int] | None`：僅針對單一鍵抓取最新影格，忽略比 `last_id` 更舊者。
  5. `store_to_redis(...) -> None`：將上傳影格及中繼資料存入 Redis Streams（`xadd`）。

**注意事項**：
- 透過 base64（`Utils.encode()`）編碼與解碼標籤與串流名稱。
- Redis Streams 單一串流最多保留 10 筆資料（`maxlen=10`）。
- 定義 `DELIMITER = b'--IMG--'` 作為 WebSocket 傳輸時 JSON 與影格 bytes 的分隔符。
- 所有函式皆有型別註解與 Google-style docstring，並採英式英文註解。

### `routers.py`
- **用途**：定義提供上傳影格（`POST /frames`）、讀取標籤（`GET /labels`）以及 WebSocket 串流影格的 FastAPI 路由。
- **端點**：
  1. `GET /labels`：每分鐘限 60 次呼叫，透過 `scan_for_labels` 取得所有標籤，回傳 `LabelListResponse`。
  2. `POST /frames`：JWT 保護，允許上傳影格與中繼資料，回傳 `FramePostResponse`。
  3. `WebSocket /ws/labels/{label}`：推送該 label 下所有串流的最新影格（header + DELIMITER + frame bytes，二進位格式）。
  4. `WebSocket /ws/stream/{label}/{key}`：拉式串流，前端以 `{"action": "pull"}` 取得最新影格，或 `{"action": "ping"}` 保持連線。
  5. `WebSocket /ws/frames`：允許已驗證用戶透過 WebSocket 上傳影格與中繼資料。

**WebSocket 互動方式**：
- **`/ws/labels/{label}`**：
  - 持續推送該 label 下所有串流的最新影格。
  - 若無任何鍵，回傳錯誤並關閉連線。
  - 影格以二進位格式傳送：JSON header + DELIMITER + frame bytes。

- **`/ws/stream/{label}/{key}`**：
  - 僅接受 JSON 指令：`{"action": "pull"}` 或 `{"action": "ping"}`。
  - `pull` 送出最新影格（header + DELIMITER + frame bytes，二進位）。
  - `ping` 回傳 JSON `{ "action": "pong" }`。
  - 其他未知指令回傳 JSON 錯誤。

- **`/ws/frames`**：
  - 已驗證端點，允許用戶透過 WebSocket 上傳影格與中繼資料。
  - 僅接受二進位訊息：JSON header + DELIMITER + frame bytes。
  - 回傳 JSON 狀態訊息（成功或錯誤）。

所有端點與 WebSocket handler 皆有型別註解與 Google-style docstring，並採英式英文註解。

### `schemas.py`
- **用途**：使用 Pydantic 模型定義結構化的請求與回應格式。
- **主要模型**：
  1. `LabelListResponse`：回傳 JSON 格式標籤清單，完整型別註解與說明。
  2. `FramePostResponse`：回傳上傳影格結果（如 `"ok"`）與訊息，完整型別註解與說明。

這些模型確保所有端點回傳格式一致且型別明確。

### `utils.py`
- **用途**：以完整型別提示與 Google-style docstring 提供 base64 編碼/解碼與 WebSocket 傳送影格等輔助函式。
- **主要函式**：
  - `encode(value: str) -> str`：將字串做 URL-safe Base64 編碼。
  - `is_base64(value: str) -> bool`：檢查字串是否為合法 URL-safe Base64。
  - `decode(value: str) -> str`：若為合法 Base64 則解碼，否則回傳原字串。
  - `send_frames(websocket: WebSocket, label: str, updated_data: list[dict[str, str | bytes | int]]) -> None`：將資料以 JSON 形式透過 WebSocket 傳送。


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

所有單元與整合測試皆位於 `tests/` 目錄，並全面採用型別提示、Google-style docstring 與英式註解，易於維護與閱讀。

```bash
pytest --cov=examples.streaming_web.backend --cov-report=term-missing
```

主要測試檔：
- **`redis_service_test.py`**：驗證影格在 Redis 中存取的正確性。
- **`routers_test.py`**：檢驗各端點與 WebSocket 功能，使用 FastAPI 的 `TestClient`，所有測試皆有型別註解與說明。

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
