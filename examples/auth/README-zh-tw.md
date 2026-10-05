# 共用認證

API service 透過 `jwt_config.py` 與 `oidc.py` 驗證 OIDC access token。
`oidc_identity.py` 將已驗證的 provider subject 對應到本機 tenant、角色、群組、
工地與功能權限。HTTP、WebSocket 與播放請求共用部署綁定及 Redis 撤銷檢查。

## 設定

各服務需一致設定 `OIDC_ENABLED=true`、`OIDC_ISSUER_URL`、`OIDC_JWKS_URL`、
`OIDC_AUDIENCE`、`OIDC_IDENTITY_PROVIDER` 與 `OIDC_ACCOUNT_URL`。
BFF 的 Authorization Code 設定見 `.env.example`。未配置 OIDC verifier 的
服務會拒絕需要認證的請求。

BFF 必須設定獨立的 `BFF_TOKEN_ENCRYPTION_KEY`，加密 Redis 內的 token。
Web 只取得 HttpOnly opaque session cookie；原生端透過 Keycloak 的
Authorization Code + PKCE 登入。Worker 使用專用 service account 與
client-credentials grant。

## 帳號管理

Keycloak 負責密碼、帳號復原與社群登入。管理員建立帳號及重設密碼使用 Keycloak
Admin API，仍需通過本機業務權限檢查。既有帳號必須已有不可變的 Keycloak subject
連結，後端不再透過 username 自動補建連結。

本機帳密登入、JWT 簽發、本機 refresh grant、忘記密碼與一次性密碼搬移橋接已移除。
部署此後端至既有資料庫時需套用 `scripts/migrate_remove_local_passwords.sql`；
新資料庫使用更新後的初始化 SQL。

Redis 保存 BFF session、media capability、token 撤銷與限流資料。
Token refresh 只向設定的 identity provider 送出。
