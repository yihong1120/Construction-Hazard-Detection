🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# 網頁前端專用後端（BFF）

網頁 BFF 由 `examples.db_management.app` 掛載於共同的 FastAPI 行程（連接埠 `8005`），透過 Nginx 的
`/bff/` 對外提供服務。此模組管理內容不透明的工作階段 Cookie、CSRF 檢查、JWT 更新，以及只轉送至允許服務的 HTTP 代理。

啟動方式：

```bash
uvicorn examples.db_management.app:app \
  --host 127.0.0.1 \
  --port 8005 \
  --workers 4 \
  --timeout-graceful-shutdown 10
```

公開路徑：

- `GET /bff/auth/session`
- `GET /bff/auth/csrf`
- `POST /bff/auth/logout`
- `GET /bff/auth/oidc/logout?state=...`（僅供登出後立即在瀏覽器最上層重新導向）
- `GET /bff/auth/oidc/login?return_to=/...`
- `GET /bff/auth/oidc/callback`（僅供 Keycloak 回呼）
- `GET /bff/auth/account`（開啟 Keycloak Account Console）
- `/bff/{allowlisted-service}/*`

部署提供 `https://<app-host>/login` 作為主應用程式的 OIDC 登入入口；它會安全地導向
`/bff/auth/oidc/login`，再由 Keycloak 完成帳密、hCaptcha 與 MFA（如有設定）。

正式服務名稱為 `chat`、`db_management`、`detection`、`fcm`、`files`、
`streaming`、`streaming_web` 與 `violations`。例如，工地清單使用
`GET /bff/db_management/list_sites`。

`GET /bff/auth/session` 會驗證目前的存取權杖，並重新查詢本機身分、
個人資料與功能權限。即使工作階段建立於權限異動前，重新整理頁面後也會反映
新增或撤銷的功能。登入當時的 Redis 快照不作為授權依據；查詢工作階段摘要
也不會覆寫其他請求正在更新的加密權杖。

違規列表與詳情回應中的 `image_url`、`thumbnail_url` 會由 BFF 改寫為
`/bff/violations/...`，使證據圖片維持走同源、以 Cookie 驗證的路徑，而不會讓
瀏覽器直接存取違規服務的內部根路由。

所有變更狀態的請求都必須送允許的 `Origin` 與 `X-CSRF-Token`。

## Keycloak / OpenID Connect 單一登入

本部署採用 OIDC 登入。網頁前端的登入按鈕導向下列路徑。

```text
GET /bff/auth/oidc/login?return_to=/violations
```

BFF 會產生一次性的 `state` 與 S256 PKCE 驗證碼，並把 Keycloak 存取與更新權杖
加密保存在 Redis；瀏覽器只會收到既有的 HttpOnly 工作階段 Cookie。回呼端點會先驗證
Keycloak 存取權杖的簽章、簽發者和 `OIDC_AUDIENCE`，再以 Keycloak `sub` 對應本機
`user_identities(provider=keycloak)`；本機的租戶、角色、群組、工地與功能權限仍是
唯一授權來源。

Web BFF 只請求一般更新權杖，**不請求** `offline_access`。每次更新權杖都會延長
同一個 Keycloak 線上單一登入工作階段，因此 Visionnaire 與 Keycloak Account Console 在同一個
瀏覽器中共用登入狀態。單一登入達到閒置或最長期限時，兩者都會要求重新驗證。

Web 登出一律先以 CSRF 保護呼叫 `POST /bff/auth/logout`。OIDC 工作階段的回應會帶有一次性、
不含權杖的 `global_logout_url`；前端必須用瀏覽器最上層頁面開啟該網址。後端會清除 BFF Cookie，
再以 Keycloak 的 HttpOnly SSO Cookie 啟動用戶端登出程序（RP-initiated logout），
結束該瀏覽器的單一登入工作階段。Flutter Web 應使用回傳的網址，不自行拼接登出網址、讀取權杖或在背景追蹤重新導向。

`OIDC_AUDIENCE` 必須指定 Visionnaire 專屬的權杖對象（建議
`visionnaire-api`）。不要把它加到 Open WebUI 用戶端的權杖；Open WebUI 與
Visionnaire 必須使用不同用戶端，否則 Open WebUI 權杖可能被誤用於 Visionnaire API。

需要保留 Web 的 Google／Apple 品牌按鈕時，只能導向
`/bff/auth/oidc/login?return_to=/&idp_hint=google` 或 `idp_hint=apple`。BFF 只轉送
允許的身分提供者提示給 Keycloak。第三方社群權杖由 Keycloak 處理。

帳號設定中的「變更密碼」連到 `/bff/auth/account`。登入按鈕使用
`/bff/auth/oidc/login`；本機帳密、直接 Google／Apple 登入及本機權杖更新
端點已移除。hCaptcha 由 Keycloak 登入佈景主題與伺服器端提供者處理。

## 推播裝置註冊

Flutter Web 在 BFF 登入成功後，先以 `GET /bff/auth/csrf` 取得 CSRF 權杖，
再使用下列路徑註冊 Firebase 裝置權杖：

```text
PUT /bff/fcm/devices
X-CSRF-Token: <csrf-token>
```

BFF 會在伺服器端注入 Bearer 權杖；瀏覽器程式不得讀取或自行傳送存取權杖。
JSON 本文只能包含 `device_token`、`device_lang` 與 `platform`（`web`）。

Flutter Web 透過 BFF 允許的 `db_management` 代理呼叫直播播放介面。原生應用程式
從公開 API 呼叫同一組端點。

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

即使回應中的 `renew_endpoint` 使用原生應用程式的路徑，網頁端仍透過 BFF 路徑續租。

BFF 採用獨立套件，共用既有行程、資料庫連線池及部署單元。

OIDC 權杖交換只有在 HTTP 400／401／403 明確回覆 `invalid_grant` 時，才將
該筆授權視為永久失效。限流保留 429；連線、伺服器、回應格式或用戶端設定
錯誤回傳 503。暫時失敗會保留工作階段，並轉送上游的 `Retry-After`。
寫回更新權杖時會檢查鎖擁有者及工作階段版本。
