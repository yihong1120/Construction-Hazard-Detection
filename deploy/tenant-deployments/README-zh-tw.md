# Tenant 與 Deployment 管理

`tenants` 與 `deployments` 是 Deployment Registry 的唯一權威設定來源。原生 App
先以公司一次性啟用碼取得 `deployment_id`，再讀取並驗證 Registry 的 Ed25519 簽章
設定檔。`api_base_url` 是 canonical HTTPS **API root**（例如
`https://api.example.com/hazard/api`），不含個別服務路徑（如 `/db_management`）、
query、fragment 或帳密；route prefix 不能成為另一個可由 client 操控的 tenant
選擇來源。

受權限保護的管理 API：

- `POST /admin/tenants`
- `PATCH /admin/tenants/{tenant_id}`
- `POST /admin/deployments`
- `PATCH /admin/deployments/{deployment_id}`
- `GET /admin/deployments`

只有 platform super-admin 可以使用這些端點。一般使用者、Flutter app 與 FCM
裝置註冊都不接收 `tenant_id` 或 `user_id` 作為選擇依據。

登入、refresh 與 BFF session 回應都會帶有 deployment contract：

```json
{
  "deployment": {
    "deployment_id": "<UUID>",
    "tenant_id": "<UUID>",
    "config_revision": 7
  }
}
```

若任一值不一致，client 不得建立／保留 session，必須清除 token，重新以已驗證的
Registry 設定檔登入。App 不得從 deep link、query 或 header 接受 tenant、deployment
或 API root 覆寫值。

## 變更順序

1. 透過管理 API 建立或更新 deployment；`api_base_url`、`tenant_id`、`status`
   的實際變更會將 `config_revision` 加一。
2. 由受信任 operator 建立、撤銷或交付一次性公司啟用碼；App 以 code 交換
   `deployment_id`，再取得 Registry 設定檔。
3. 讓使用者以新的簽章設定檔重新登入。舊 access token、refresh token 與 BFF
   session 會因 issuer/audience、deployment 或 revision 不符而收到明確的 `409`
   `deployment_configuration_changed`，而不是被靜默接受。

反向代理必須將外部 HTTPS scheme 正確交給 ASGI。伺服器只依實際抵達的 HTTPS
origin 查詢 `deployments`，不信任 client 提供的 tenant/deployment header。

## Signed Deployment Registry

受管理的原生 App 可先用一次性公司啟用碼交換 deployment ID：

```text
POST /hazard/api/deployment-registry/v1/enrollments/exchange
```

Request body 必須為 `Content-Type: application/json`，且僅能有
`enrollment_code`。`Accept` 與 `Cache-Control` 可由 client 或 edge 加入，但不是
Registry 驗證 response 的必要條件。此 endpoint 不接受 Bearer token、Cookie、CSRF
或 refresh token；它以 Redis 限制 IP 與 code verifier
的嘗試次數，並在 PostgreSQL transaction／row lock 中將 code 標記為已使用。資料庫
只保存由 `DEPLOYMENT_ENROLLMENT_CODE_PEPPER` HMAC 的 verifier，絕不保存或記錄原始
啟用碼。成功回應固定為：

```json
{"deployment_id":"<uuid>"}
```

接著 App 在登入前使用：

```text
GET /hazard/api/deployment-registry/v1/deployments/{deployment_id}
```

Flutter build define 的值為：

```text
DEPLOYMENT_REGISTRY_URL=https://<host>/hazard/api/deployment-registry
```

Registry 是獨立 FastAPI router 與簽章 service，不經過 BFF、登入、Cookie、
CSRF 或 JWT middleware。成功時只會回傳九個固定欄位：

```json
{
  "schema_version": 1,
  "deployment_id": "<uuid>",
  "tenant_id": "<uuid>",
  "api_base_url": "https://<host>/hazard/api",
  "config_revision": 1,
  "issued_at": 1735689600,
  "expires_at": 1735693200,
  "key_id": "registry-ed25519-2026-01",
  "signature": "<base64url-ed25519-signature>"
}
```

設定 `DEPLOYMENT_REGISTRY_ED25519_PRIVATE_KEY` 時，只能從 KMS、Secret Manager
或受保護的 runtime 環境變數注入 Ed25519 PKCS#8 PEM；不可放入 Git、App 設定
或回傳 JSON。`DEPLOYMENT_REGISTRY_TTL_SECONDS` 必須介於 1 與 86400；私鑰缺失、
格式錯誤或 TTL 不合法時 endpoint 會 fail closed 回 `503`，不會產生未簽章文件。
`DEPLOYMENT_ENROLLMENT_CODE_PEPPER` 也必須從相同受保護的 secret store 注入，且
必須至少 32 bytes；未設定、Redis 或資料庫不可用時，exchange 同樣只回無敏感資訊的
`503`。啟用碼到期、撤銷或已兌換回 `410`；不存在／格式無效回 `403`；超過限制回
`429`。

建立 code 使用受信任的私有 operator tooling；migration 與該 tooling 刻意不隨
application source 發布。原始 code 只能在 commit 成功後透過核准的公司管道交付，
不可貼入 ticket、log 或 source control。

## 已登入的裝置邀請管理

tenant admin 與 platform super-admin 可為**目前登入 deployment**建立、列出與撤銷
裝置邀請。tenant 與 deployment 一律從 deployment-bound JWT 或 BFF session 推導；
request body 不可包含 `tenant_id`、`deployment_id`、API/Registry URL 或簽章金鑰。

原生 App 使用 Bearer token 呼叫 API root 下的：

```text
POST   /hazard/api/db_management/deployment-enrollment-codes
GET    /hazard/api/db_management/deployment-enrollment-codes
DELETE /hazard/api/db_management/deployment-enrollment-codes/{id}
```

POST body 僅能是 `{"expires_in_minutes":30}`，範圍為 1–1440。成功建立時，回應會
**唯一一次**包含 `enrollment_code`；之後的 GET、audit、錯誤回應與資料庫皆不含原碼
或 verifier。列表只會回傳 canonical lower-case UUID、`expires_at` 與
`active`／`redeemed`／`expired`／`revoked` 狀態。DELETE 為 idempotent `204`。

Web 不直接持有 Bearer token，而是呼叫：

```text
/bff/db_management/deployment-enrollment-codes
```

BFF 以 HttpOnly session 保存 server-side token，並對 POST 與 DELETE 強制受信任
Origin 與 `X-CSRF-Token`。BFF 不會把 access/refresh token 回傳瀏覽器。

套用 `20260821_add_deployment_enrollment_code_management_postgres.sql` 後，
`deployment_enrollment_codes.public_id` 是給管理 API 使用的公開 UUID；
`deployment_enrollment_code_audit_logs` 僅記錄 created/revoked 的 actor、tenant、
deployment 與時間，不保存啟用碼或雜湊。
App 必須以內建或安全配置的公開金鑰 key ring，依 `key_id` 找出對應公開金鑰並驗證
signature。簽署 bytes 必須是以下七個欄位依 key 字典序、UTF-8、無空白的 JSON（不含
`key_id` 或 `signature`，也不是 HTTP body）：

```json
{"api_base_url":"...","config_revision":1,"deployment_id":"...","expires_at":123,"issued_at":123,"schema_version":1,"tenant_id":"..."}
```

將
`nginx.deployment-registry.conf.example` 的 HTTP/HTTPS location 放入對應的
server block（HTTPS location 必須在通用 `/hazard/api/` fallback 前）。本機正式
設定可套用相同內容至 `deploy/bff/nginx.changdar-server.complete.conf`。此路由只
允許 HTTPS（HTTP 請求直接失敗、不轉址），並示範由 edge 加上
`Cache-Control: no-store`。client 不可將該 header 視為 Registry response 有效性的
必要條件。

## 本機開發直連

正式環境必須透過註冊的 HTTPS API root。僅在本機需要直接呼叫
`http://127.0.0.1:8005/login` 或 `/refresh` 時，才可在受保護的本機 `.env` 設定：

```dotenv
LOCAL_DEVELOPMENT_AUTH_ENABLED=true
LOCAL_DEVELOPMENT_DEPLOYMENT_ID=<已註冊 deployment UUID>
```

此例外不接受任意 origin 或 tenant：socket peer 與 `Host` 都必須是 loopback
(`127.0.0.1`、`::1` 或 `localhost`)，且 deployment 永遠取自伺服器環境變數。它不
會接受任意 origin 或 tenant。
