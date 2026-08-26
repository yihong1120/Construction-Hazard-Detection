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
- `GET /bff/auth/oidc/login?return_to=/...`（Keycloak 登入切換後）
- `GET /bff/auth/oidc/callback`（僅供 Keycloak callback）
- `GET /bff/auth/account`（開啟 Keycloak Account Console）
- `/bff/{allowlisted-service}/*`

部署提供 `https://<app-host>/login` 作為主 App 的 OIDC 登入入口；它會安全地導向
`/bff/auth/oidc/login`，再由 Keycloak 完成帳密與 MFA（如有設定）。

正式 service 名稱為 `chat`、`db_management`、`detection`、`fcm`、`files`、
`streaming`、`streaming_web` 與 `violations`。例如，站點列表使用
`GET /bff/db_management/list_sites`。

違規列表與詳情回應中的 `image_url`、`thumbnail_url` 會由 BFF 改寫為
`/bff/violations/...`，使證據圖片維持走同源、以 Cookie 驗證的路徑，而不會讓
瀏覽器直接存取違規服務的內部根路由。

所有變更狀態的 request 都必須送允許的 `Origin` 與 `X-CSRF-Token`。

## Keycloak / OpenID Connect 單一登入

設定 `OIDC_ENABLED=true` 後，Web 前端的登入按鈕應導向：

```text
GET /bff/auth/oidc/login?return_to=/violations
```

BFF 會產生一次性的 `state`、S256 PKCE verifier，並把 Keycloak access/refresh token
加密保存在 Redis；瀏覽器只會收到既有的 HttpOnly session cookie。callback 會先驗證
Keycloak access token 的簽章、issuer 和 `OIDC_AUDIENCE`，再以 Keycloak `sub` 對應本機
`user_identities(provider=keycloak)`；本機的 tenant、角色、群組、site 與 feature 權限仍是
唯一授權來源。

`OIDC_AUDIENCE` 必須是 Visionnaire 專用的 Keycloak audience（建議
`visionnaire-api`）。不要把它加到 Open WebUI client 的 token；Open WebUI 與
Visionnaire 必須使用不同 client，否則 Open WebUI token 可能被誤用於 Visionnaire API。

完成帳號切換後，帳號設定中的「變更密碼」請連到 `/bff/auth/account`。開啟
`OIDC_PASSWORDS_MANAGED_EXTERNALLY=true` 時，舊的 Visionnaire 密碼變更與重設 API
會回傳 `409 password_managed_by_identity_provider` 與 Keycloak Account Console URL，不再
寫入本機 `password_hash`。同一開關也會停用舊的帳密、Google、Apple 與 legacy refresh
登入路徑；登入按鈕必須改用 `/bff/auth/oidc/login`。

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
