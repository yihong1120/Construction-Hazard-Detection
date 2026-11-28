🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

模型清單、預設模型與 Flutter 整合規格，請參閱[模型 API 文件](DETECTION_MODEL_API.md)。

# YOLO 推論 API

這個 FastAPI 服務用於 HTTP 和 WebSocket 測試、遠端部署物件偵測，以及管理模型檔案。在本機大量處理影格時，建議使用 `src/yolo_worker.py`。`main.py` 透過共享記憶體傳送影格，可省去 JPEG 編碼與 WebSocket 傳輸的成本。

需要讓其他程式透過獨立 API 呼叫推論時，即可使用此服務。

## 提供功能

- `POST /detect` 接受單張上傳圖片，回傳 YOLO 物件偵測結果。
- `WebSocket /ws/detect` 接收圖片位元組並回傳偵測結果。
- `POST /model_file_update` 是舊的 `.pt` 上傳端點。服務會檢查模型管理權限，並向具權限的呼叫者回傳 HTTP 409 `IMMUTABLE_MODEL_REGISTRY`。更新模型時，請先將新版本發布至 Hugging Face，再同步模型目錄中對應的項目。
- `POST /get_new_model` 驗證用戶端提供的時間戳記，並回傳已驗證的模型檔。若用戶端傳入的 `If-None-Match` 符合檔案的 ETag，服務會回傳 HTTP 304。

每個偵測結果都是一組數值，其欄位順序如下。

```text
[x1, y1, x2, y2, confidence, class_id]
```

## 檔案

- `app.py` 建立 FastAPI 應用程式，並管理服務啟動與關閉。
- `routers.py` 定義偵測及模型管理路由。
- `websocket_handlers.py` 處理選用的 WebSocket 偵測連線。
- `detection.py` 負責影像解碼、推論及偵測框後處理。
- `models.py` 依版本快取模型，並管理推論期間的模型使用租約。
- `model_files.py` 以串流方式計算模型傳輸所需的 SHA-256 雜湊值。模型檔案路徑由模型目錄提供。
- `config.py` 管理執行設定，包括推論裝置的選擇。
- `schemas.py` 定義請求與回應的資料模型。

## 執行

請從儲存庫根目錄執行下列指令。

```bash
uvicorn examples.YOLO_server_api.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1
```

每個 Uvicorn 工作程序會分別載入 PyTorch 模型。若希望同一 GPU 上只有一份模型，請使用一個工作程序。

## 設定

以下是部署設定範例。HTTP 偵測端點會直接執行身分驗證。

```dotenv
# DETECT_API_AUTH_REQUIRED=true （已停用）
DETECT_API_URL=http://127.0.0.1:8000
```

服務依模型 ID 選取模型目錄中已發布的版本，載入前會以該版本登錄的 SHA-256 雜湊值核對檔案。

## 身分驗證

HTTP 路由與需要登入的 WebSocket 共用 `examples.auth.jwt_config` 的 `jwt_access` 驗證機制。其 OIDC、部署及權杖撤銷檢查與通知、違規紀錄 API 相同。啟用 Keycloak 時，請在請求中帶入登入後取得的存取權杖。

```text
Authorization: Bearer <access-token>
```

`/detect` 的流量限制只檢查請求配額，無須舊登入機制的 Redis `jti_list`。WebSocket 不會自動登錄權杖識別碼。服務查詢部署設定時，會將對外 WSS 位址對應至 HTTPS，因此反向代理須保留對外的主機名稱及通訊協定。若使用獨立 WSS 網域，請先為該網域登錄部署設定。同一台主機上的流程可使用 `YOLO_WS_ALLOW_LOCALHOST_BYPASS`。若所有連線都須驗證，請將其設為 `false`。

舊的上傳端點要求 `admin` 或 `model_manage` 角色。具備角色後仍須依不可變更的模型目錄規則發布新版本。

## 效能注意事項

- 本機的多攝影機處理流程建議使用 `src/` 的 YOLO 工作程序。
- 透過 API 在程序之間傳送影像時，影像編碼與解碼會占用 CPU 時間及記憶體頻寬。
- WebSocket 模式適合相容性測試。本機追求高吞吐量時，使用共享記憶體的工作程序較快。
