🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

模型清單、預設模型與 Flutter 聯調契約：[模型 API 文件](DETECTION_MODEL_API.md)。

# YOLO Server API

可選的 FastAPI detector service，用於 HTTP/WebSocket 測試、遠端 detector 部署與模型檔案
管理。目前本機高吞吐主路徑仍建議使用 `src/yolo_worker.py`，由 `main.py` 透過 shared
memory 傳送 frame，避免 JPEG 與 WebSocket 額外成本。

當你刻意需要用 API 邊界隔離推論時，才使用此服務。

## 提供功能

- `POST /detect`：上傳單張圖片並回傳 YOLO detections。
- `WebSocket /ws/detect`：接收 image bytes 並回傳 detections。
- `POST /model_file_update`：上傳替換用 `.pt` 模型檔。
- `POST /get_new_model`：當 client 模型時間戳過舊時回傳更新模型。

detection 結果格式：

```text
[x1, y1, x2, y2, confidence, class_id]
```

## 檔案

- `app.py`：FastAPI application 與 lifespan。
- `routers.py`：detection 與 model-management routes。
- `websocket_handlers.py`：可選 WebSocket detection path。
- `detection.py`：image decode、inference 與 bounding-box post-processing。
- `models.py`：model manager 與 file watcher。
- `model_files.py`：模型傳輸使用的串流 SHA-256 checksum；檔案路徑與上傳由 `routers.py` 的 registry 流程處理。
- `config.py`：runtime settings，包含 device selection。
- `schemas.py`：request 與 response models。

## 執行

從 repo 根目錄：

```bash
uvicorn examples.YOLO_server_api.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1
```

除非你刻意要重複載入模型，否則一個 GPU model instance 建議只用一個 worker。多個
Uvicorn workers 不會共享同一個 PyTorch model。

## 設定

常見環境變數：

```dotenv
DETECT_API_AUTH_REQUIRED=true
DETECT_SERVER_MODEL_KEYS=yolo26n,yolo26s
DETECT_API_URL=http://127.0.0.1:8000
YOLO_MODEL_DIR=models/pt
```

模型檔案遵循專案命名規則：

```text
models/pt/best_<model_key>.pt
```

例如 `model=yolo26n` 會載入 `models/pt/best_yolo26n.pt`。

## 驗證

HTTP routes 與需登入的 WebSocket 共用 `examples.auth.jwt_config.jwt_access`
驗證核心，與通知、違規紀錄 API 使用相同的 OIDC、部署及撤銷檢查。啟用
Keycloak 時，請使用登入取得的 access token：

```text
Authorization: Bearer <access-token>
```

`/detect` 的限流只處理配額，不再要求舊登入系統的 Redis `jti_list`。
WebSocket 不再自動登錄 jti，驗證流程與 HTTP API 共用。
WSS 的公開來源會以 HTTPS 對應到已登錄的 deployment；反向代理需保留正確
Host 與外部 scheme。不同 WSS 網域需先有對應的部署設定。
既有同機管線的 `YOLO_WS_ALLOW_LOCALHOST_BYPASS` 選項仍保留；若所有連線都
必須驗證，請設為 `false`。

模型上傳 endpoint 需要模型管理權限。

## 效能注意事項

- 多攝影機主流程請優先使用 `src/` 的本機 YOLO workers。
- API path 會在 process 邊界進行圖片 encode/decode，會消耗 CPU 與記憶體頻寬。
- WebSocket mode 適合相容性測試，但不是最快的本機路徑。
