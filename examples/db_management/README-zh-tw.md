🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Database Management Backend

用於管理使用者、群組、功能權限、工地與 stream 設定的 FastAPI 服務。`main.py` 會讀取
這裡的啟用 stream 設定，因此這個 backend 是直播偵測 runtime 的 control plane。

## 職責

- 使用者登入與 JWT refresh。
- 管理使用者、待審核註冊、角色與群組。
- 依群組管理功能權限。
- 管理工地與 user-site access。
- 管理 stream 設定，包含攝影機 URL、stream name、model key、偵測選項、工作時間與
  live publishing flags。

## 檔案

- `app.py`：FastAPI application。
- `deps.py`：JWT、role 與 site-permission dependencies。
- `routers/`：auth、users、groups、features、sites、streams routes。
- `schemas/`：Pydantic request 與 response models。
- `services/`：async SQLAlchemy service layer。

## 執行

從 repo 根目錄：

```bash
uvicorn examples.db_management.app:app \
  --host 127.0.0.1 \
  --port 8005 \
  --workers 2
```

OpenAPI docs：`http://127.0.0.1:8005/docs`。

## 必要設定

```dotenv
DATABASE_URL=postgresql+asyncpg://username:password@127.0.0.1/construction_hazard_detection
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=password
JWT_SECRET_KEY=replace-with-a-long-random-secret
```

所有 API service 必須使用同一個 `JWT_SECRET_KEY`。

## Stream 設定與 Runtime

啟用的 stream rows 會驅動主偵測流程：

```text
database stream_configs -> main.py -> src/stream_processor.py
```

重要欄位包含：

- source URL 與 stream display name；
- site label 與 stream ID；
- `model_key`，對應 `models/pt/best_<model_key>.pt`；
- detection item switches 與 warning thresholds；
- working-hour schedule；
- clean 與 annotated MediaMTX publishing options。

多攝影機部署建議使用 database mode，而不是本機 JSON config，這樣更新設定不需要修改
`main.py`。
