🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Database Management Backend

用於管理使用者、群組、功能權限、工地與 stream 設定的 FastAPI 服務。`main.py` 會讀取
這裡的啟用 stream 設定，因此這個 backend 是直播偵測 runtime 的 control plane。

## 職責

- 使用者可用 username 或 e-mail 搭配密碼登入，並支援 JWT refresh。
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
JWT_SECRET_KEY=replace-with-a-long-random-secret
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
CORS_ALLOWED_ORIGINS=https://changdar-server.mooo.com,http://localhost:3000,http://127.0.0.1:3000,http://localhost:5000,http://127.0.0.1:5000,http://localhost:8080,http://127.0.0.1:8080
BREVO_API_KEY=replace-with-your-brevo-api-key
MAIL_FROM=verified-sender@example.com
MAIL_FROM_NAME=Visionnaire
APP_PUBLIC_URL=https://changdar-server.mooo.com
PASSWORD_RESET_TOKEN_TTL_SECONDS=1800
```

所有 API service 必須使用同一個 `JWT_SECRET_KEY`。
`HCAPTCHA_SECRET_KEY` 只能放在後端環境變數，不要提交到 git。
`HCAPTCHA_BYPASS_KEY` 只給可信任的後端腳本使用，不能放到瀏覽器前端。
`BREVO_API_KEY` 只能放在後端環境變數，不要提交到 git。
`BFF_TOKEN_ENCRYPTION_KEY` 用來加密後端保存的 access/refresh token，不可提供給
任何 client。Flutter Web 只會拿到 opaque HttpOnly session cookie；Native 使用
OAuth Authorization Code + PKCE。
`CORS_ALLOWED_ORIGINS` 必須列出允許帶 cookie 的 Web origin；credentialed
request 不可以使用 `*`。
若使用 Docker Compose，`PLAYBACK_STREAMING_API_URL` 應設為
`http://streaming-web-backend:8000`；若所有服務直接跑在 host 上，才使用
`http://127.0.0.1:8800`。

## 登入

`POST /auth/login` 使用單一帳號欄位，接受 username 或 e-mail：

```json
{
  "identifier": "user@example.com",
  "password": "password",
  "hcaptcha_token": "frontend-hcaptcha-token"
}
```

`POST /auth/google` 與 `POST /auth/apple` 接收 provider login tokens；當綁定的
本地使用者為 active 時，回傳格式與帳密登入相同。新的 provider 帳號會建立為 pending
user，需由管理員核准後才能取得本地 JWT。
若 provider token 的 e-mail 已存在於既有帳號，後端不會自動合併，會回傳
`account_link_required`，請使用者先用原本方式登入後再綁定。

已登入使用者可管理登入方式：

- `GET /auth/identities`
- `POST /auth/identities/google/link`
- `POST /auth/identities/apple/link`
- `DELETE /auth/identities/{identity_id}`

解除綁定時，若該 user 沒有密碼且只剩一個 provider identity，後端會回
`last_login_method`，避免帳號失去所有登入方式。

## Native OAuth 與 Unified Playback

Flutter Web BFF 路由由掛載於同一 process 的 `examples/bff` 模組提供。此服務也
保留 Native OAuth，並提供 Flutter Web/iOS/Android 共用的 playback facade。

Native 使用 `GET /hazard/api/db_management/oauth/authorize`、
`POST /hazard/api/db_management/oauth/token`、
`GET /hazard/api/db_management/me` 與
`POST /hazard/api/db_management/oauth/revoke`。只接受 S256 PKCE 與設定檔列出的
client/redirect 配對。
Access token 有效 15 分鐘；refresh token 每次旋轉並保存 family reuse-detection。

Flutter Web/iOS/Android 直播統一呼叫既有
`/hazard/api/db_management/` base path 下的 playback facade：

```text
POST   /hazard/api/db_management/api/playback/sessions
POST   /hazard/api/db_management/api/playback/walls
POST   /hazard/api/db_management/api/playback/sessions/renew
DELETE /hazard/api/db_management/api/playback/sessions/{id}
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

若已套用舊版 `user_identities` migration，請再執行：

```bash
psql "$DATABASE_URL" \
  -f scripts/migrations/20260621_upgrade_user_identities_login_methods.sql
```

## 忘記密碼

`POST /password/forgot` 不論 e-mail 是否存在都回傳：

```json
{
  "message": "If the email exists, a reset link has been sent."
}
```

如果 e-mail 對應到使用者，後端會產生一次性 raw token，只把
SHA-256 hash 寫入 Redis，TTL 使用 `PASSWORD_RESET_TOKEN_TTL_SECONDS`，
再透過 Brevo 寄出
`APP_PUBLIC_URL/reset_password?token={raw_reset_token}`。

`POST /password/reset` 接收 URL 上的 raw token 與新密碼。重設成功後會刪除
Redis reset token，並移除該使用者既有 JWT session cache。

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
