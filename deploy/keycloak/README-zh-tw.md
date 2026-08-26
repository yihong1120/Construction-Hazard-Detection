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

Provider 對 hCaptcha 的 `siteverify` 回應會同時驗證成功狀態、site key 與公開
hostname；外部服務異常或設定不完整時會 fail closed，絕不略過真人驗證。

## Flutter Native client

啟動程序也會建立 `visionnaire-mobile` public OIDC client。它只允許 Authorization
Code + PKCE（S256），停用 implicit、password/direct grant 與 client secret，callback
固定為 `com.changdar.visionnaire:/oauthredirect`；access token 會有
`visionnaire-api` audience。Flutter app 不得持有 client secret 或 hCaptcha secret。
