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
FCM_TOKEN_ENCRYPTION_KEY=replace-with-a-fernet-key
```

Firebase service-account JSON 請放在 repo 外。
`FCM_TOKEN_ENCRYPTION_KEY` 用於加密 DB 內的 device token。若未設定，服務會由
`JWT_SECRET_KEY` 派生 key，但正式環境建議使用獨立 Fernet key：

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## Endpoints

- `POST /store_token`：建立加密後的 DB token 紀錄，並刷新 Redis 發送 cache。
  API 不回 raw token，也不回 token hash。
- `DELETE /delete_token`：在 DB 停用 device token，並從 Redis cache 移除。
- `POST /send_fcm_notification`：對訂閱使用者發送一筆工地警示。
- `GET /notifications?status=unread&type=violation&page=1&page_size=20`：
  列出目前使用者的 app 內通知。
- `GET /notifications/unread_count`：取得未讀 badge 數量。
- `PATCH /notifications/{id}/read`：將單筆通知標為已讀。
- `PATCH /notifications/read_all`：將目前使用者所有通知標為已讀。
- `DELETE /notifications/{id}`：刪除單筆通知。
- `GET /notifications/site_preferences`：列出目前使用者可管理的工地通知訂閱設定。
- `PUT /notifications/site_preferences`：更新工地通知訂閱設定。

`/send_fcm_notification` 可帶 `deep_link`；服務會把同一個 deep link 寫入 FCM
`data.deep_link` 與 notification record。若未提供，違規通知會預設使用
`/violations?violation_id={id}`。

## Runtime 注意事項

- 支援語言定義在 `lang_config.py`。
- 不支援的 device language 會被拒絕，不會靜默轉成其他語言。
- DB 是 FCM token 的 source of truth；Redis 只作為短期發送 cache 與
  subscription 狀態。
- log 只能記錄 `token_hash`，不要記錄 raw FCM token。
- 面向使用者的 API response 不應暴露 raw FCM token 或 token hash；
  token 診斷請使用 `/notifications/device-status`。
- 違規圖片建議使用 violation records service 產生的公開圖片 URL。
