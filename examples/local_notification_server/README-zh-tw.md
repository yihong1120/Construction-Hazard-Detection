🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Local Notification Server

Firebase Cloud Messaging (FCM) 的 FastAPI 服務。stream processor 偵測到違規後會呼叫此
服務，由它解析工地訂閱、使用者 token 與語言化通知文字。

## 職責

- 儲存與刪除 device FCM tokens。
- 依 user 與 site 追蹤 token language。
- 依語言分組發送違規通知。
- 移除 Firebase 回報的 invalid tokens。
- 分批讀取 Redis token，避免一次載入所有 token 到記憶體。

## 檔案

- `app.py`：FastAPI application 與 Firebase initialisation lifespan。
- `routers.py`：token 與 notification endpoints。
- `services.py`：Redis token lookup、site subscription 與 send orchestration。
- `fcm_service.py`：Firebase Admin SDK wrapper。
- `lang_config.py`：支援語言與 warning translations。
- `schemas.py`：request 與 response models。

## 執行

```bash
uvicorn examples.local_notification_server.app:app \
  --host 127.0.0.1 \
  --port 8003 \
  --workers 2
```

## 必要設定

```dotenv
FIREBASE_CRED_PATH=/secure/path/firebase-service-account.json
FCM_API_URL=http://127.0.0.1:8003
DATABASE_URL=postgresql+asyncpg://username:password@127.0.0.1/construction_hazard_detection
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=password
JWT_SECRET_KEY=replace-with-a-long-random-secret
```

Firebase service-account JSON 請放在 repo 外。

## Endpoints

- `POST /store_token`：儲存使用者 device token 與語言。
- `DELETE /delete_token`：移除 device token。
- `POST /send_fcm_notification`：對訂閱使用者發送一筆工地警示。
- `GET /site_notification_users/{site}`：列出工地通知訂閱者。
- `PUT /site_notification_users/{site}`：更新工地訂閱者。

## Runtime 注意事項

- 支援語言定義在 `lang_config.py`。
- 不支援的 device language 會被拒絕，不會靜默轉成其他語言。
- Redis 只用於小型 token 與 subscription 狀態。
- 違規圖片建議使用 violation records service 產生的公開圖片 URL。
