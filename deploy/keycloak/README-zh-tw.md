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

Provider 對 hCaptcha 的 `siteverify` 會帶入 server-side secret、一次性 challenge token
與預期 site key；外部服務異常、token 無效或 site key 不相符時會 fail closed，絕不
略過真人驗證。網域限制必須在 hCaptcha Dashboard 的 sitekey **Domain Allowlist** 設定
`mooo.com`（會涵蓋 `changdar-server.mooo.com`）；不可依賴 `siteverify` 回傳的
`hostname`，因為該值是瀏覽器衍生的統計資訊。

既有 Visionnaire 使用者可能沒有 email、名字或姓氏。為使 OIDC 切換後仍可直接登入，
部署會停用 Keycloak 的 `VERIFY_PROFILE` required action；使用者可在登入後的 Keycloak
Account Console 自行補齊個人資料，但不會在登入流程中被強制要求。

Keycloak realm 的 `browserSecurityHeaders.contentSecurityPolicy` 必須在
`frame-src` 明確允許 `https://hcaptcha.com` 與 `https://*.hcaptcha.com`；這是
hCaptcha iframe 實際使用的回應標頭，不能只寫在 login theme 的
`theme.properties`。

### 舊 Visionnaire 密碼的一次性遷移

遷移期間，Keycloak 密碼政策必須與既有 Visionnaire 的最小長度相容（目前為
8 字元）。若在此期間提高最小長度，原本合法但較短的密碼無法寫入 Keycloak，
使用者就會被錯誤地拒絕登入。所有帳號完成遷移並通知使用者更新密碼後，才可再
提高 Keycloak 的密碼政策。

既有帳號的 Argon2 hash 不可直接複製到 Keycloak database。遷移期間設定：

```dotenv
LEGACY_PASSWORD_MIGRATION_ENABLED=true
LEGACY_PASSWORD_MIGRATION_TTL_SECONDS=30
```

使用者第一次以原 Visionnaire 密碼登入時，Keycloak 先驗證自己的 credential；失敗才會由
custom form 透過 loopback HMAC 向 Visionnaire 驗證既有 hash。驗證成功後，Keycloak 立即以
同一份密碼建立自己的 credential，並以一個 30 秒、一次性的 migration token 回呼停用
`users.password_hash`。明文密碼、舊 hash、migration token 都不會寫入 Keycloak log、Redis
或 HTTP response。

此橋接只接受 `127.0.0.1` 的 Keycloak container，並以與 native-social HMAC 不可互換的
domain-separated signature 保護；它不會恢復 Visionnaire 的舊登入 API。所有已遷移的帳號
後續只使用 Keycloak 密碼。確認 active 密碼帳號都已遷移後，設定
`LEGACY_PASSWORD_MIGRATION_ENABLED=false` 並重新部署 Keycloak 與 db-management。

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
