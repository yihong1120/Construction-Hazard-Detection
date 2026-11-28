🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# 本機通知服務

這個 FastAPI 服務使用 Firebase Cloud Messaging（FCM）傳送推播。串流處理程式送出違規事件後，服務會查詢工地訂閱者及其裝置權杖，並依各裝置的語言產生通知文字。

## 職責

- 儲存及刪除裝置的 FCM 權杖。
- 依使用者與工地追蹤各裝置權杖對應的語言。
- 依語言分組傳送違規通知。
- 移除 Firebase 回報為無效的權杖。
- 分批處理 Redis 的權杖資料，避免一次將所有權杖載入記憶體。

## 檔案

- `app.py` 建立 FastAPI 應用程式，並在服務啟動時初始化 Firebase。
- `routers.py` 定義 HTTP 端點及其相依服務。
- `notification_delivery_service.py` 處理權杖註冊、重複資料排除及 FCM 發送流程。
- `notification_centre_service.py` 負責通知中心的查詢與狀態更新。
- `site_preference_service.py` 依使用者權限處理各工地的通知偏好。
- `services.py` 負責資料持久化、Redis 快取、翻譯，以及具並行數量限制的 FCM 批次發送。
- `fcm_service.py` 透過 Firebase Admin SDK 傳送訊息。
- `lang_config.py` 定義支援的語言及各語言的警示文字。
- `schemas.py` 定義請求與回應的資料模型。

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
FIREBASE_PROJECT_ID=your-firebase-project-id
FCM_API_URL=http://127.0.0.1:8003
DATABASE_URL=postgresql+asyncpg://username:password@127.0.0.1/construction_hazard_detection
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=password
OIDC_ENABLED=true
OIDC_ISSUER_URL=https://sso.example.com/realms/visionnaire
OIDC_JWKS_URL=https://sso.example.com/realms/visionnaire/protocol/openid-connect/certs
OIDC_AUDIENCE=visionnaire-api
OIDC_ACCOUNT_URL=https://sso.example.com/realms/visionnaire/account
FCM_TOKEN_ENCRYPTION_KEY=replace-with-a-fernet-key
```

## 容器部署

請從儲存庫根目錄建置，將共用的 `examples/auth`、`examples/db_management` 和 `src` 套件納入映像檔。

```bash
docker build -f examples/local_notification_server/Dockerfile -t local-notification-server .
```

啟動容器時，請將 Firebase 服務帳戶的 JSON 憑證以機密檔案掛載至 `/run/secrets/firebase-service-account.json`。若使用其他路徑，請設定 `FIREBASE_CRED_PATH` 指向實際位置。建置過程不會從建置目錄複製憑證。

請將 Firebase 服務帳戶的 JSON 憑證存放在儲存庫外。服務使用 `FCM_TOKEN_ENCRYPTION_KEY` 加密資料庫中的裝置權杖。這是必填設定，請使用有效的 Fernet 金鑰。

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## API 端點

- `PUT /devices` 建立或更新資料庫中已加密的裝置權杖，並更新 Redis 發送快取。服務依 Bearer 權杖識別使用者。JSON 必填欄位為 `device_token`、`device_lang` 和 `platform`。`device_lang` 須使用服務支援的標準 BCP 47 語言代碼，`platform` 可填 `web`、`ios` 或 `android`。註冊回應不含原始權杖及其雜湊值。
- `DELETE /devices` 在資料庫停用目前使用者的指定裝置權杖，並從 Redis 快取中移除。
- `POST /send_fcm_notification` 將一筆工地警示發給訂閱者。事件來源須提供 `type`、`title`、`deep_link` 和 `metadata`，完整指定通知內容及點選後開啟的位置。
- `GET /notifications?status=unread&type=violation&page_size=20&cursor=<next_cursor>` 列出目前使用者在應用程式內收到的通知。
- `GET /notifications/unread_count` 回傳未讀通知數，供介面顯示。
- `PATCH /notifications/{id}/read` 將一筆通知標為已讀。
- `PATCH /notifications/read_all` 將目前使用者的所有通知標為已讀。
- `DELETE /notifications/{id}` 刪除一筆通知。
- `GET /notifications/site_preferences` 列出目前使用者可管理的工地通知訂閱設定。
- `PUT /notifications/site_preferences` 更新工地通知訂閱設定。

`/send_fcm_notification` 會將必填欄位 `deep_link` 和 `type` 原樣寫入 FCM 訊息資料及通知紀錄。產生事件的程式須提供通知規格要求的所有欄位。

## 執行注意事項

- `lang_config.py` 定義支援的語言代碼。
- 裝置註冊若使用服務不支援的語言，服務會拒絕請求。
- FCM 權杖以資料庫紀錄為準。Redis 保存短期發送快取及訂閱狀態。
- 日誌只能使用 `token_hash` 識別權杖，不得記錄原始 FCM 權杖。
- 裝置註冊回應會省略權杖及其雜湊值。登入後可透過 `/notifications/device-status` 查詢自己的裝置狀態；診斷資料以雜湊值識別裝置，不會回傳原始權杖。
- 傳送違規圖片時，請使用違規紀錄服務產生的公開圖片網址。

裝置權杖的所有權由 PostgreSQL 記錄。同一裝置改用其他帳號註冊時，服務會刪除舊帳號的發送快取、中繼資料及索引項目。每次發送前，服務會以一次批次查詢，比對快取版本與資料庫中的權杖所有者、更新時間。只有收件資料過期時才會重新解密權杖及重建快取。即使 Redis 曾中斷、部分寫入延後，服務仍能修復快取，不會因舊的就緒標記而沿用過期資料。
