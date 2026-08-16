# Web Backend-for-Frontend

Web BFF 是獨立的程式碼模組，由 `examples.db_management.app` 掛載在共同的
FastAPI process（port `8005`），並由 Nginx 以 `/bff/` 對外提供。它負責 Web
opaque session cookie、CSRF、JWT refresh，以及只允許固定 service 的 HTTP
proxy。

啟動方式：

```bash
uvicorn examples.db_management.app:app \
  --host 127.0.0.1 \
  --port 8005 \
  --workers 4 \
  --timeout-graceful-shutdown 10
```

公開路徑：

- `POST /bff/auth/login`
- `GET /bff/auth/session`
- `GET /bff/auth/csrf`
- `POST /bff/auth/logout`
- `/bff/{allowlisted-service}/*`

正式 service 名稱為 `chat`、`db_management`、`detection`、`fcm`、`files`、
`streaming`、`streaming_web` 與 `violations`。例如，站點列表使用
`GET /bff/db_management/list_sites`。

所有變更狀態的 request 都必須送允許的 `Origin` 與 `X-CSRF-Token`。

## 推播裝置註冊

Flutter Web 在 BFF 登入成功後，先以 `GET /bff/auth/csrf` 取得 CSRF token，
再使用下列路徑註冊 Firebase 裝置 token：

```text
PUT /bff/fcm/devices
X-CSRF-Token: <csrf-token>
```

BFF 會在伺服器端注入 bearer token；瀏覽器程式不得讀取或自行傳送 access token。
JSON body 只能包含 `device_token`、`device_lang` 與 `platform`（`web`）。

直播播放不再由 BFF 提供 media-session endpoint。Flutter Web 與 Native 共用
db_management 的 playback facade：

```text
POST   /hazard/api/db_management/api/playback/sessions
POST   /hazard/api/db_management/api/playback/walls
POST   /hazard/api/db_management/api/playback/sessions/renew
DELETE /hazard/api/db_management/api/playback/sessions/{id}
```

保留獨立 package 可維持清楚的安全邊界，同時不增加另一個 process、DB pool 與
部署單元。
