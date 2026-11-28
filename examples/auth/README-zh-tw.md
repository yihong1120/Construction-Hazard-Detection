🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# 共用認證

API 服務透過 `jwt_config.py` 與 `oidc.py` 驗證 OIDC 存取權杖。
`oidc_identity.py` 將已驗證的身分提供者主體識別碼對應到本機租戶、角色、群組、
工地與功能權限。HTTP、WebSocket 與播放請求共用部署綁定及 Redis 撤銷檢查。

## 設定

各服務需一致設定 `OIDC_ENABLED=true`、`OIDC_ISSUER_URL`、`OIDC_JWKS_URL`、
`OIDC_AUDIENCE`、`OIDC_IDENTITY_PROVIDER` 與 `OIDC_ACCOUNT_URL`。
BFF 的 Authorization Code 設定見 `.env.example`。未設定 OIDC 驗證器的
服務會拒絕需要認證的請求。

BFF 必須設定獨立的 `BFF_TOKEN_ENCRYPTION_KEY`，加密 Redis 內的權杖。
網頁端只取得內容不透明的 HttpOnly 工作階段 Cookie。原生應用程式透過 Keycloak 的
Authorization Code + PKCE 登入。背景工作程序使用專用的服務帳號，以用戶端憑證授權。

## 帳號管理

Keycloak 負責密碼、帳號復原與社群登入。管理員建立帳號及重設密碼使用 Keycloak
Admin API，仍需通過本機業務權限檢查。既有帳號必須已有不可變的 Keycloak 主體識別碼連結。後端不會依使用者名稱自動建立連結。

本機帳密登入、JWT 簽發、本機權杖更新授權、密碼復原與一次性密碼移轉橋接已移除。
部署此後端至既有資料庫時需套用 `scripts/migrate_remove_local_passwords.sql`；
新資料庫使用更新後的初始化 SQL。

Redis 儲存 BFF 工作階段、媒體存取授權、權杖撤銷紀錄及流量限制資料。
更新權杖的請求會送至設定的身分提供者。

## 跨儲存庫的驗證變更

變更登入與工作階段行為時，請同步檢查 `ai-gateway` 與
`visionnaire-flutter-dev`。本服務採用 Keycloak 與 OIDC。閘道的本機密碼簽發、
密碼復原權杖資料表及身分移轉程序屬於閘道本身。

原生社群登入的起始請求與 API 配額共用 `redis_counters.py`。單一 Lua 操作會
遞增計數器，並在缺少到期時間時補上設定，同時保留原計數時段。跨專案更新時，
請一併執行驗證與已停用路由的迴歸測試。

閘道將帳號、工作階段及身分提供者的核心程式放在 `examples.identity`。
本服務使用 `examples.auth`，由 Keycloak 管理登入。兩邊共用登入、權杖更新、
Cookie 及部署契約，計數器行為須與閘道的 `examples.identity.redis_counters` 一致。
