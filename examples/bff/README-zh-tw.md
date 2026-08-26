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

違規列表與詳情回應中的 `image_url`、`thumbnail_url` 會由 BFF 改寫為
`/bff/violations/...`，使證據圖片維持走同源、以 Cookie 驗證的路徑，而不會讓
瀏覽器直接存取違規服務的內部根路由。

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

直播播放不使用獨立的 BFF playback service。Flutter Web 經由既有 allow-listed
`db_management` proxy；Native 則以 public API root 呼叫同一個 facade：

```text
Web:    POST   /bff/db_management/api/playback/sessions
Web:    POST   /bff/db_management/api/playback/walls
Web:    POST   /bff/db_management/api/playback/sessions/renew
Web:    DELETE /bff/db_management/api/playback/sessions/{id}

Native: POST   /hazard/api/db_management/api/playback/sessions
Native: POST   /hazard/api/db_management/api/playback/walls
Native: POST   /hazard/api/db_management/api/playback/sessions/renew
Native: DELETE /hazard/api/db_management/api/playback/sessions/{id}
```

即使 response 含 Native 格式的 `renew_endpoint`，Web 仍必須維持 BFF path 續租。

保留獨立 package 可維持清楚的安全邊界，同時不增加另一個 process、DB pool 與
部署單元。
