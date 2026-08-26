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

資料庫帳號至少必須能連線至獨立的 `keycloak` database。建立 database 的權限可只由
PostgreSQL 管理者在初次部署時使用；不需要授予 Keycloak 或 Visionnaire 應用程式額外
的 superuser 權限。
