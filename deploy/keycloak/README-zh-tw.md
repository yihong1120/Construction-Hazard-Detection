# Keycloak 部署

此 Compose 設定使用主機既有的 PostgreSQL instance，但 Keycloak 必須使用獨立的
`keycloak` database；絕不可使用 Visionnaire 的 `construction_hazard_detection`
database 或其資料表。

主機 PostgreSQL 若僅綁定 `127.0.0.1`，此設定以 host network 執行 Keycloak，並且將
Keycloak HTTP 僅綁定在 `127.0.0.1:8081`。外部流量只能經由現有 Nginx 的
`/keycloak/` HTTPS proxy 進入。

部署前請在 `.env` 設定：

- `KEYCLOAK_ADMIN_USERNAME`、`KEYCLOAK_ADMIN_PASSWORD`
- `KEYCLOAK_DB_USERNAME`、`KEYCLOAK_DB_PASSWORD`
- `KEYCLOAK_USER_LINKER_CLIENT_SECRET`
- 所有 `OIDC_*` 值
- `HCAPTCHA_SITE_KEY`、`HCAPTCHA_SECRET_KEY`

資料庫帳號至少必須能連線至獨立的 `keycloak` database。建立 database 的權限可只由
PostgreSQL 管理者在初次部署時使用；不需要授予 Keycloak 或 Visionnaire 應用程式額外
的 superuser 權限。

## 登入防護

此映像檔包含自訂的 `visionnaire-hcaptcha` Keycloak Authenticator 與
`visionnaire` login theme。容器啟動後會透過 Keycloak Admin API 建立並指定
`visionnaire-browser` flow：既有 Keycloak SSO cookie 可直接續用；新的帳密登入
則一定依序通過「帳密 → hCaptcha」驗證。hCaptcha secret 只存在容器環境變數，
不會寫入 realm JSON、Keycloak 管理設定或 Flutter 前端。

Web BFF 必須使用一般 OIDC refresh token，不得請求 `offline_access`。一般 refresh 會延長
線上 SSO session，讓 Visionnaire 與 Account Console 共用同一次登入；offline token 的用途
是無人背景工作，會在瀏覽器 SSO 已失效後仍保持 API session，造成使用者開啟 Account Console
時被意外要求再次登入。

Provider 對 hCaptcha 的 `siteverify` 會帶入 server-side secret、一次性 challenge token
與預期 site key；外部服務異常、token 無效或 site key 不相符時會 fail closed，絕不
略過真人驗證。網域限制必須在 hCaptcha Dashboard 的 sitekey **Domain Allowlist** 設定
`mooo.com`（會涵蓋 `changdar-server.mooo.com`）；不可依賴 `siteverify` 回傳的
`hostname`，因為該值是瀏覽器衍生的統計資訊。

既有 Visionnaire 使用者可能沒有 email、名字或姓氏。為使 OIDC 切換後仍可直接登入，
部署會停用 Keycloak 的 `VERIFY_PROFILE` required action；使用者可在登入後的 Keycloak
Account Console 自行補齊個人資料，但不會在登入流程中被強制要求。

Realm 密碼政策以 `forceExpiredPasswordChange(180)` 將密碼有效期設為 180 天；期限從
Keycloak 密碼 credential 最近一次建立或更新時計算，到期後才在登入流程要求變更。
建立帳號與管理員重設密碼仍使用非暫時密碼，不會額外觸發首次登入強制變更。

Keycloak realm 的 `browserSecurityHeaders.contentSecurityPolicy` 必須在
`frame-src` 明確允許 `https://hcaptcha.com` 與 `https://*.hcaptcha.com`；這是
hCaptcha iframe 實際使用的回應標頭，不能只寫在 login theme 的
`theme.properties`。

### 帳密驗證

帳密登入使用 Keycloak 的標準 `auth-username-password-form`，之後執行 hCaptcha。
舊 Visionnaire 密碼搬移 provider 已移除；啟動腳本會清除既有 browser forms flow
中殘留的搬移 execution，並重新啟用標準帳密驗證器。

## Google 與 Apple 社群登入

Web BFF 的 Google／Apple 仍由 Keycloak Identity Broker 處理。啟動時會將 Identity
Provider Redirector 放在 browser flow 的帳密表單之前：Google／Apple 按鈕走第三方登入；
帳密分支才會進入 hCaptcha。

provider 預設停用。設定 `KEYCLOAK_GOOGLE_ENABLED=true` 或
`KEYCLOAK_APPLE_ENABLED=true` 時，必須同時提供其 client ID 與 secret；否則啟動程序
會停用該 provider，避免 UI 出現無法使用的按鈕。完整的外部回呼 URL、Apple JWT secret
輪替與三平台驗收程序見
[Keycloak 社群登入部署規格](../../docs/zh/keycloak_social_login.md)。

Flutter iOS／Android 可另外使用官方 Google／Apple SDK，但不是直接取得 Visionnaire
JWT，也不使用 Keycloak 已淘汰的 external Token Exchange v1。Visionnaire API 先驗證
provider assertion 和 nonce，Keycloak custom authenticator 再透過 loopback HMAC 消耗
PKCE-bound one-use proof，最終仍回到標準 Authorization Code + PKCE。完整 API、連結
交易與前端規格在
[原生社群憑證交換規格](../../docs/zh/native_social_exchange.md)。

## Flutter Native client

啟動程序也會建立 `visionnaire-mobile` public OIDC client。它只允許 Authorization
Code + PKCE（S256），停用 implicit、password/direct grant 與 client secret，callback
固定為 `com.changdar.visionnaire:/oauthredirect`；access token 會有
`visionnaire-api` audience。Flutter app 不得持有 client secret 或 hCaptcha secret。
