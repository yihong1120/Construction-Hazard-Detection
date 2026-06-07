🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLO Server API Backend

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
- `model_files.py`：model upload 與 retrieval helper。
- `config.py`：runtime settings，包含 device selection。
- `schemas.py`：request 與 response models。

## 執行

從 repo 根目錄：

```bash
uvicorn examples.YOLO_server_api.backend.app:app \
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

routes 使用 `examples.auth` 的共用 JWT dependency。請先透過 database management API
登入，再帶入：

```text
Authorization: Bearer <access-token>
```

模型上傳 endpoint 需要模型管理權限。

## 效能注意事項

- 多攝影機主流程請優先使用 `src/` 的本機 YOLO workers。
- API path 會在 process 邊界進行圖片 encode/decode，會消耗 CPU 與記憶體頻寬。
- WebSocket mode 適合相容性測試，但不是最快的本機路徑。
