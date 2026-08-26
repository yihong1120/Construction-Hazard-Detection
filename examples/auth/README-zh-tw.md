🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Authentication And Authorisation

`examples/` 內 FastAPI 服務共用的驗證與授權模組。這裡提供 JWT 驗證、Redis-backed
token/session cache、密碼雜湊、資料庫 session，以及 user/site models。

此目錄是共用 library，不是獨立 application。

## 主要檔案

- `config.py`：從環境變數讀取設定。
- `database.py`：async SQLAlchemy engine、session factory 與 base model。
- `models.py`：唯一的 SQLAlchemy ORM 定義模組，在 `global_lifespan`
  建立 metadata 前載入。
- `jwt_config.py`：JWT access 與 refresh dependencies。
- `cache.py`：Redis user cache、token cache 與輕量 rate helper。
- `redis_pool.py`：HTTP 與 WebSocket handlers 共用的 async Redis pool。
- `user_service.py`：user/site access helper 與 cache invalidation。
- `token_cleanup.py`：清理過期 token cache entries。
- `security.py` 與 `jwt_scheduler.py`：可選的 secret 產生工具。

## 必要設定

```dotenv
DATABASE_URL=postgresql+asyncpg://username:password@127.0.0.1/construction_hazard_detection
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=password
JWT_SECRET_KEY=replace-with-a-long-random-secret
```

同一套部署中的所有 API service 必須使用同一個穩定、高熵的 `JWT_SECRET_KEY`。更換 key
會立刻讓現有 access token 與 refresh token 失效，因此只能作為有計畫的安全操作。

## 密碼

專案使用 `pwdlib[argon2]` 雜湊與驗證使用者密碼。不要儲存明文密碼，也不要用可逆加密
取代 Argon2。

## Keycloak / OpenID Connect 遷移

部署 Open WebUI 時，建議以 Keycloak 作為唯一帳號與密碼來源，Visionnaire、Open WebUI
各自是 OIDC client；不要讓 Open WebUI 讀取 Visionnaire 的 `users.password_hash` 或共用
Visionnaire 的 JWT secret。

Visionnaire 在 `OIDC_ENABLED=true` 時會接受 Keycloak 的 RS256 access token，並驗證固定
issuer、JWKS、Visionnaire 專用 audience。驗證後必須以 `UserIdentity` 的
`provider=keycloak`、`provider_user_id=<Keycloak sub>` 連到本機使用者；不會以 email 或
username 自動比對。這使串流、違規、通知和管理 API 仍沿用本機 tenant/group/site 權限，
但共用 Keycloak 登入狀態。

建議切換順序：

1. 建立 Keycloak realm、`visionnaire-web`（confidential）和 `open-webui`（依 Open WebUI
   OIDC 文件設定）兩個 client；為 Visionnaire client 加入 `visionnaire-api` audience
   mapper，勿加入 Open WebUI client。
2. 先在 Keycloak 建立既有帳號與初始密碼。是否可保留原密碼取決於 Keycloak 的 credential
   import 與現有 Argon2 參數是否經過實測相容；無法驗證時，使用 Keycloak 的 required
   password update/reset，而不是複製或降級雜湊。
3. 以 service-account（僅授予目標 realm 的 `view-users`）先執行：

   ```bash
   uv run python scripts/keycloak_link_users.py \
     --server-url https://sso.example.com \
     --realm visionnaire \
     --client-id visionnaire-user-linker \
     --client-secret "$KEYCLOAK_LINKER_SECRET"
   ```

   預設是 dry-run。確認每個 username 對應後再加 `--apply`；工具只寫入 Keycloak UUID
   對應，不會讀取或搬移密碼。
4. 讓所有服務套用相同 `OIDC_*` 值並開啟 BFF OIDC login。確認 Visionnaire 與 Open WebUI
   都可登入後，才設 `OIDC_PASSWORDS_MANAGED_EXTERNALLY=true`，把 Visionnaire 的改密碼、
   忘記密碼與管理者重設導至 Keycloak Account Console／Admin Console。此開關也會拒絕
   舊的帳密、Google、Apple 登入與 legacy refresh token，避免本機 password hash 成為第二個
   可登入的 credential authority；既有 session 會要求重新透過 Keycloak 登入。

Keycloak 的 realm issuer 一般是
`https://<sso-host>/realms/visionnaire`，其 JWKS endpoint 為
`.../protocol/openid-connect/certs`。Open WebUI 使用同一 issuer 的
`/.well-known/openid-configuration`，callback 是
`https://<open-webui-host>/oauth/oidc/callback`。詳見 [Open WebUI SSO
文件](https://docs.openwebui.com/features/authentication-access/auth/sso/) 與
[Keycloak OIDC 文件](https://www.keycloak.org/securing-apps/oidc-layers)。

## Redis 用途

Redis 儲存小型驗證與授權狀態：

- user cache entries；
- refresh-token references；
- access-token `jti` lists；
- effective site access cache；
- 共用 cache helpers 使用的 Lua script 狀態。

Redis 不儲存直播影像 frame。

## 被哪些服務使用

- `examples/db_management/`
- `examples/local_notification_server/`
- `examples/streaming_web/`
- `examples/violation_records/`
- `examples/YOLO_server_api/`

請直接啟動上述服務；它們會透過 lifespan 與 dependency wiring 匯入此目錄。
