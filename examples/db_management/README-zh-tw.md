🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Database Management Backend

用於管理使用者、群組、功能權限、工地與 stream 設定的 FastAPI 服務。`main.py` 會讀取
這裡的啟用 stream 設定，因此這個 backend 是直播偵測 runtime 的 control plane。

## 職責

- 驗證 OIDC access token，並管理 Keycloak identity。
- 供原生 App 以公司一次性啟用碼選定 deployment，並在登入前取得 Ed25519 簽章
  deployment 設定的匿名 Registry。
- 管理使用者、待審核註冊、角色與群組。
- 依群組管理功能權限。
- 管理工地與 user-site access。
- 管理 stream 設定，包含攝影機 URL、stream name、model key、偵測選項、工作時間與
  live publishing flags。

## 檔案

- `app.py`：FastAPI application。
- `deps.py`：JWT、role 與 site-permission dependencies。
- `routers/`：auth、users、groups、features、sites、streams routes。
- `schemas/`：Pydantic request 與 response models。
- `services/`：async SQLAlchemy service layer。

## 執行

從 repo 根目錄：

```bash
uvicorn examples.db_management.app:app \
  --host 127.0.0.1 \
  --port 8005 \
  --workers 2 \
  --timeout-graceful-shutdown 10
```

OpenAPI docs：`http://127.0.0.1:8005/docs`。

## 必要設定

```dotenv
DATABASE_URL=postgresql+asyncpg://username:password@127.0.0.1/construction_hazard_detection
DB_POOL_SIZE=2
DB_MAX_OVERFLOW=1
DB_POOL_TIMEOUT_SECONDS=10
DB_POOL_RECYCLE_SECONDS=1800
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=password
OIDC_ENABLED=true
OIDC_ISSUER_URL=https://sso.example.com/realms/visionnaire
OIDC_JWKS_URL=https://sso.example.com/realms/visionnaire/protocol/openid-connect/certs
OIDC_AUDIENCE=visionnaire-api
OIDC_ACCOUNT_URL=https://sso.example.com/realms/visionnaire/account
HCAPTCHA_ENABLED=true
HCAPTCHA_SECRET_KEY=replace-with-your-hcaptcha-secret
HCAPTCHA_SITE_KEY=3e5cc8c8-0e36-4316-8416-63f0e4c635d0
HCAPTCHA_BYPASS_KEY=local-script-only-random-secret
BFF_TOKEN_ENCRYPTION_KEY=replace-with-an-independent-random-secret
BFF_SESSION_COOKIE_NAME=__Host-vn_session
BFF_SESSION_COOKIE_SECURE=true
BFF_SESSION_TTL_SECONDS=2592000
MEDIA_SESSION_TTL_SECONDS=600
PLAYBACK_STREAMING_API_URL=http://127.0.0.1:8800
CORS_ALLOWED_ORIGINS=https://changdar-server.mooo.com,https://visionnaire-cda17.web.app,http://localhost:3000,http://127.0.0.1:3000,http://localhost:5000,http://127.0.0.1:5000,http://localhost:8080,http://127.0.0.1:8080
DEPLOYMENT_API_BASE_PATH=/hazard/api
DEPLOYMENT_REGISTRY_ED25519_PRIVATE_KEY=load-from-secret-manager-only
DEPLOYMENT_REGISTRY_KEY_ID=registry-ed25519-2026-01
DEPLOYMENT_REGISTRY_TTL_SECONDS=86400
```

各 API service 共用同一組 OIDC issuer、JWKS URL 與 API audience 設定。

## 登入

Web 使用 `GET /bff/auth/oidc/login` 的 Keycloak Authorization Code + PKCE
流程；原生端直接使用 Keycloak 的 Authorization Code + PKCE。
API 只接受 OIDC access token。管理員建立帳號、重設密碼及更新帳號仍透過
Keycloak Admin API；個人密碼與帳號復原使用 `/bff/auth/account`。

本機帳密登入、直接 Google／Apple 登入、本機 JWT／refresh 簽發及密碼搬移橋接
已移除。既有資料庫需套用 `scripts/migrate_remove_local_passwords.sql`。

## Native OAuth 與 Unified Playback

Flutter Web BFF 路由由掛載於同一 process 的 `examples/bff` 模組提供。此服務也
保留 Native OAuth，並提供 Flutter Web/iOS/Android 共用的 playback facade。

Native 使用 Keycloak 的 Authorization Code + PKCE，並以 `GET /me` 取得本機業務權限。

原生 App 輸入公司一次性啟用碼後，先呼叫：

```text
POST /hazard/api/deployment-registry/v1/enrollments/exchange
```

成功回應只含 `deployment_id`；接著在登入前呼叫獨立的：

```text
GET /hazard/api/deployment-registry/v1/deployments/{deployment_id}
```

此 router 不在 `/db_management` 或 `/bff` 下，不讀取 Authorization、Cookie、CSRF
或 refresh token，只會回傳一份最長 24 小時的九欄位 Ed25519 簽章文件。Nginx 必須
將公開路徑轉送到同一個 8005 process 的 `/deployment-registry/` router；完整部署
契約與可提交的 Nginx 片段見 `deploy/tenant-deployments/README-zh-tw.md`。

tenant admin 與 super-admin 可建立、列出或撤銷目前登入 deployment 的一次性裝置
邀請：

```text
POST   /hazard/api/db_management/deployment-enrollment-codes
GET    /hazard/api/db_management/deployment-enrollment-codes
DELETE /hazard/api/db_management/deployment-enrollment-codes/{id}
```

POST body 僅接受 `{"expires_in_minutes":30}`（1–1440）。新 code 只在 POST 成功
回應中出現一次；GET 和 audit 均不回傳 code 或 verifier。Web 必須透過
`/bff/db_management/deployment-enrollment-codes` 呼叫，BFF 會以 HttpOnly session
保存 server-side token，並為 POST／DELETE 驗證 Origin 與 CSRF token。

Native 直播使用既有 `/hazard/api/db_management/` base path 下的 playback facade；
Flutter Web 則使用 authenticated BFF proxy：

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

單鏡頭回傳 `mode: "single"` 與 `hls_url`；多鏡頭牆回傳
`mode: "multi_stream"`、`layout: "responsive"` 與
`items[*].preview_hls_url`。`hls_url`/`preview_hls_url` 會指向 stable playback
playlist，內含短效 `mt` media token；streaming_web 會在回傳 playlist 時把 segment
URL 補上同一個 `mt`，播放器不用再分 Web Cookie 或 Native Bearer header。

多鏡頭牆可以只送 site，或送明確的鏡頭清單：

```json
{
  "site": "Site A",
  "cameras": ["Cam 1", "Cam 2"],
  "profile": "overlay"
}
```

多鏡頭牆固定使用獨立的低碼率 `preview` rendition，不再是 detail HLS path 的
別名；`profile` 只決定畫面模式：要後端繪製辨識結果時送 `"overlay"`，關閉
「顯示辨識結果」時送 `"clean"`。單鏡頭 session 固定使用 detail rendition。

`POST /api/playback/sessions/renew` 的 body 為 `{"id":"..."}`。它只延長原本
media capability 的 TTL，`hls_url` 與 `items[*].preview_hls_url` 不會改變；前端
續租成功後不可重建播放器。

舊的 Web/Native media-session 公開 API 已移除；Flutter 不需要先建立 Cookie
或 Bearer media session。多鏡頭牆最多 24 個不重複鏡頭，不支援 wildcard。
即時影像只會回傳已開啟 `recognition_enabled` 的設定。

```dotenv
GOOGLE_WEB_CLIENT_ID=860473757501-c1gtkrqr4lsa52vgoq7vclprm8atjvtv.apps.googleusercontent.com
GOOGLE_IOS_CLIENT_ID=860473757501-s53qldp7i294qbg1ia8aq822oa0rudj2.apps.googleusercontent.com
GOOGLE_ANDROID_CLIENT_ID=860473757501-088t4flpgv0kdds6pu4a5m1fntamf1ht.apps.googleusercontent.com
APPLE_TEAM_ID=5DU8R27949
APPLE_KEY_ID=NGC4QBS7ZY
APPLE_SERVICE_ID=com.changdar.visionnaire.signin
APPLE_BUNDLE_ID=com.changdar.visionnaire
APPLE_REDIRECT_URI=https://changdar-server.mooo.com/hazard/api/db_management/auth/apple/callback
APPLE_PRIVATE_KEY_PATH=config/secrets/apple/AuthKey_NGC4QBS7ZY.p8
```

## Stream 設定與 Runtime

開啟辨識且位於設定工作時段內的 stream rows 會驅動主偵測流程：

```text
database stream_configs -> main.py -> src/stream_processor.py
```

重要欄位包含：

- source URL 與 stream display name；
- site label 與 stream ID；
- `model_key`，對應 `models/pt/best_<model_key>.pt`；
- `recognition_enabled`：關閉時仍保存攝影機設定，但不會啟動擷取、辨識或違規處理；
- detection item switches 與 warning thresholds；
- working-hour schedule；
- clean 與 annotated MediaMTX publishing options。

多攝影機部署建議使用 database mode，而不是本機 JSON config，這樣更新設定不需要修改
`main.py`。

## 已退役路由

讀取串流設定統一使用 `GET /sites/{site_id}/stream-config`。
`/list_stream_configs`、`/auth/verify-email`、`/auth/resend-verification` 已移除。
Email 驗證與郵件發送由 Keycloak realm 管理；啟用此流程前應設定 SMTP 與驗證政策。
移除本機驗證不會自動啟用既有帳號。
