🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# 資料庫管理後端

請將範例中的網域、位址與識別碼替換成自己部署環境的值。

此 FastAPI 服務管理使用者、群組、功能權限、工地與串流設定。Python 攝影機服務
`main.py` 從資料庫讀取啟用的串流設定，以執行即時辨識。

## 職責

- 驗證 OIDC 存取權杖，並管理 Keycloak 身分。
- 提供匿名部署登錄服務。原生應用程式以公司的一次性啟用碼選擇部署，並在登入前取得
  Ed25519 簽章的部署設定。
- 管理使用者、待審核註冊、角色與群組。
- 依群組管理功能權限。
- 管理工地及使用者對工地的存取權限。
- 管理串流設定，包括攝影機網址、串流名稱、模型識別碼、偵測選項、工作時間及
  即時影像發布選項。

## 檔案

- `app.py`：FastAPI 應用程式。
- `deps.py`：JWT、角色與工地權限的相依檢查。
- `routers/`：驗證、使用者、群組、功能、工地及串流的路由。
- `schemas/`：Pydantic 請求與回應模型。
- `services/`：非同步 SQLAlchemy 服務層。

Open WebUI 帳號橋接分成 `schemas/openwebui.py`（資料格式）、
`openwebui_deps.py`（OIDC 身分與管理員驗證）、
`services/openwebui_services.py`（跨應用帳號同步）及
`routers/openwebui.py`（HTTP 路由）。一般使用者管理與 Open WebUI
共用 `services/user_action_services.py` 的帳號異動流程，服務層不直接呼叫路由。

串流設定沿用 `schemas/stream_config.py` 的資料格式。
`routers/streams.py` 負責 HTTP 路由與回應標頭；
`services/stream_management_services.py` 負責工地／群組範圍、模型選項、
配額檢查及批次更新交易；`services/stream_config_services.py` 保留資料存取
與回應資料組裝。對外 API 沿用既有的路徑及請求與回應格式。

網站 IP 管理使用 `schemas/website_access.py` 定義資料格式，
`website_access_models.py` 定義資料表，`services/website_access.py` 處理
白名單變更、稽核查詢與 Nginx 套用狀態判讀；
`routers/website_access.py` 處理 HTTP 路由、管理員驗證與來源 IP 擷取。

## 執行

從儲存庫根目錄執行下列指令。

```bash
uvicorn examples.db_management.app:app \
  --host 127.0.0.1 \
  --port 8005 \
  --workers 2 \
  --timeout-graceful-shutdown 10
```

OpenAPI 文件位於 `http://127.0.0.1:8005/docs`。

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
HCAPTCHA_SITE_KEY=replace-with-your-hcaptcha-site-key
HCAPTCHA_BYPASS_KEY=replace-with-a-local-random-secret
BFF_TOKEN_ENCRYPTION_KEY=replace-with-an-independent-random-secret
BFF_SESSION_COOKIE_NAME=__Host-vn_session
BFF_SESSION_COOKIE_SECURE=true
BFF_SESSION_TTL_SECONDS=2592000
MEDIA_SESSION_TTL_SECONDS=600
PLAYBACK_STREAMING_API_URL=http://127.0.0.1:8800
CORS_ALLOWED_ORIGINS=https://app.example.org,https://web.example.org,http://localhost:3000,http://127.0.0.1:3000,http://localhost:5000,http://127.0.0.1:5000,http://localhost:8080,http://127.0.0.1:8080
DEPLOYMENT_API_BASE_PATH=/hazard/api
DEPLOYMENT_REGISTRY_ED25519_PRIVATE_KEY=<private-key-from-secret-manager>
DEPLOYMENT_REGISTRY_KEY_ID=registry-ed25519-2026-01
DEPLOYMENT_REGISTRY_TTL_SECONDS=86400
```

各 API 服務共用 OIDC 簽發者、JWKS 網址與 API 權杖的預期對象。

## 登入

Web 使用 `GET /bff/auth/oidc/login` 的 Keycloak Authorization Code + PKCE
流程；原生端直接使用 Keycloak 的 Authorization Code + PKCE。
API 只接受 OIDC 存取權杖。管理員建立帳號、重設密碼及更新帳號仍透過
Keycloak Admin API；個人密碼與帳號復原使用 `/bff/auth/account`。

本機帳密登入、直接 Google／Apple 登入、本機 JWT 與更新權杖的簽發及密碼搬移橋接
已移除。既有資料庫需套用 `scripts/migrate_remove_local_passwords.sql`。

## 原生應用程式 OAuth 與共用播放介面

Flutter Web BFF 路由由掛載於同一行程的 `examples/bff` 模組提供。此服務也
保留原生應用程式的 OAuth，並提供 Flutter Web、iOS 與 Android 共用的播放介面。

原生應用程式使用 Keycloak 的 Authorization Code + PKCE，並以 `GET /me` 取得本機業務權限。

原生應用程式輸入公司一次性啟用碼後，先呼叫：

```text
POST /hazard/api/deployment-registry/v1/enrollments/exchange
```

成功回應只含 `deployment_id`；接著在登入前呼叫獨立的：

```text
GET /hazard/api/deployment-registry/v1/deployments/{deployment_id}
```

此路由獨立於 `/db_management` 或 `/bff`，不讀取 Authorization 標頭、Cookie、CSRF 權杖或更新權杖，只會回傳一份最長 24 小時的九欄位 Ed25519 簽章文件。Nginx 必須
將公開路徑轉送到同一個行程（連接埠 8005）的 `/deployment-registry/` 路由。
詳見[部署契約與 Nginx 設定](../../deploy/tenant-deployments/README-zh-tw.md)。

租戶管理員及全站管理員可建立、列出或撤銷目前登入部署的一次性裝置邀請。

```text
POST   /hazard/api/db_management/deployment-enrollment-codes
GET    /hazard/api/db_management/deployment-enrollment-codes
DELETE /hazard/api/db_management/deployment-enrollment-codes/{id}
```

POST 本文僅接受 `{"expires_in_minutes":30}`（1 至 1440 分鐘）。新的邀請碼只在 POST 成功回應中出現一次，GET 與稽核紀錄均不提供邀請碼或驗證值。網頁端必須透過
`/bff/db_management/deployment-enrollment-codes` 呼叫，BFF 會以 HttpOnly Cookie 識別工作階段，將權杖保存在伺服器端，
並為 POST 與 DELETE 驗證 Origin 及 CSRF 權杖。

原生應用程式使用 `/hazard/api/db_management/` 下的播放介面。Flutter Web 則透過驗證使用者身分的 BFF 代理存取。

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
`items[*].preview_hls_url`。兩個網址皆指向穩定的播放清單，內含短效媒體權杖 `mt`。
`streaming_web` 回傳清單時會將相同的 `mt` 加到片段網址。網頁與原生播放器皆可直接使用。

多鏡頭牆可只傳送工地，也可指定鏡頭清單。

```json
{
  "site": "Site A",
  "cameras": ["Cam 1", "Cam 2"],
  "profile": "overlay"
}
```

多鏡頭牆固定使用獨立的低位元率 `preview` 版本。
`profile` 決定畫面模式：要後端繪製辨識結果時送 `"overlay"`，關閉
「顯示辨識結果」時送 `"clean"`。單鏡頭工作階段固定使用詳細畫質版本。

`POST /api/playback/sessions/renew` 的請求本文為 `{"id":"..."}`。它只延長原本
媒體存取授權的有效時間，`hls_url` 與 `items[*].preview_hls_url` 不會改變；前端
續租成功後不可重建播放器。

舊的 Web/Native 媒體工作階段公開 API 已移除；Flutter 不需要先建立 Cookie
或 Bearer 媒體工作階段。公開播放介面的 `cameras` 清單可指定最多 16 個
不重複的鏡頭，不支援萬用字元。省略清單時使用工地的預設選擇，另受上游配額限制。
明確指定清單時，一律適用 16 個鏡頭的上限。
即時影像只會回傳已開啟 `recognition_enabled` 的設定。

```dotenv
GOOGLE_WEB_CLIENT_ID=your-web-client-id.apps.googleusercontent.com
GOOGLE_IOS_CLIENT_ID=your-ios-client-id.apps.googleusercontent.com
GOOGLE_ANDROID_CLIENT_ID=your-android-client-id.apps.googleusercontent.com
APPLE_TEAM_ID=YOURTEAMID
APPLE_KEY_ID=YOUR_KEY_ID
APPLE_SERVICE_ID=com.example.visionnaire.signin
APPLE_BUNDLE_ID=com.example.visionnaire
APPLE_REDIRECT_URI=https://app.example.org/hazard/api/db_management/auth/apple/callback
APPLE_PRIVATE_KEY_PATH=config/secrets/apple/AuthKey_YOUR_KEY_ID.p8
```

## 串流設定與執行流程

已開啟辨識的串流設定會驅動 Python 偵測流程，並依工作時段處理警示。

```text
database stream_configs -> main.py -> src/stream_processor.py
```

重要欄位包含：

- 來源網址與串流顯示名稱。
- 工地名稱與串流 ID。
- `model_key` 指定模型目錄中已發布且支援 `stream` 的模型。
- `recognition_enabled`：關閉時仍保存攝影機設定，但不會啟動擷取、辨識或違規處理。
- 偵測項目開關與警示門檻。
- 工作時段。
- MediaMTX 發布原始串流與標註串流的選項。

多攝影機部署使用資料庫模式時，可以直接更新設定，無須修改 `main.py`。

## 已退役路由

讀取串流設定統一使用 `GET /sites/{site_id}/stream-config`。
`/list_stream_configs`、`/auth/verify-email`、`/auth/resend-verification` 已移除。
電子郵件驗證與郵件發送由 Keycloak realm 管理；啟用此流程前應設定 SMTP 與驗證政策。
移除本機驗證不會自動啟用既有帳號。

## 網站 IP 存取管理

全站管理員可透過 `/admin/website-access` 管理主網站、AI、Wiki 各自的
IPv4／IPv6 白名單、名稱、啟用狀態及變更紀錄。網站政策與工地資料分開。
API、版本衝突、防止鎖定管理員，以及 Nginx 同步與部署步驟請見
[網站存取管理](../../deploy/website_access/README-zh-tw.md)。
