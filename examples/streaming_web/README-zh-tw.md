🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# 網頁串流後端

FastAPI 服務負責列出可觀看的串流，提供直播中繼資料、MediaMTX 存取授權和 WebRTC ICE 設定。影像由 MediaMTX 透過 HLS 或 WebRTC 傳送。

## 執行時的職責

```text
main.py / src/stream_processor.py
        |
        +--> 將原始及標註後的 H.264 串流發布至 MediaMTX
        +--> 將精簡的警示中繼資料寫入 Redis

streaming_web
        |
        +--> 從 PostgreSQL 取得可觀看的工地與串流
        +--> 回傳 MediaMTX 的 HLS／WebRTC 播放網址
        +--> 透過 SSE／WebSocket 傳送中繼資料
        +--> 使用 /media-auth 驗證請求，再由 Nginx 轉送至 MediaMTX
```

## 檔案

- `app.py`：FastAPI 應用程式設定。
- `routers.py`：宣告 HTTP 與 WebSocket 路由，注入所需的服務。
- `application_services.py`：為每個請求組合所需的應用服務。
- `streaming_api_service.py`：媒體授權與播放 API 政策。
- `playback_service.py`：播放工作階段、播放清單與反映觀看需求的發布租約。
- `stream_catalog_service.py`：依權限列出工地與串流。
- `streaming_metadata_service.py`：在授權後建立 SSE 或 WebSocket 回應。
- `streaming_metadata_handlers.py`：共用的 SSE 和 WebSocket 中繼資料傳輸邏輯。
- `redis_service.py` 與 `metadata_keys.py`：精簡中繼資料在 Redis 中的鍵值與格式。
- `media_paths.py`：MediaMTX 路徑、公開網址及觀看需求的鍵值。
- `overlay_renderer.py`：具快取功能的多語疊圖繪製工具。
- `webrtc_service.py`：產生 STUN／TURN ICE 伺服器設定。
- `schemas.py`：定義並驗證 HTTP 與 Redis 的資料格式。
- `nginx.hazard-media.conf`：Nginx 的媒體代理與授權設定。

## 執行

```bash
uvicorn examples.streaming_web.app:app \
  --host 127.0.0.1 \
  --port 8800 \
  --workers 2 \
  --timeout-graceful-shutdown 10
```

## API 端點

- `GET /labels`：列出登入者可存取的工地標籤。
- `GET /streams/{label}`：取得串流 ID 與 HLS／WebRTC 播放網址。
- `GET /metadata/stream-id/{label}/{stream_id}`：透過 SSE 傳送警示中繼資料。
- `WebSocket /ws/metadata-id/{label}/{stream_id}`：透過 WebSocket 傳送中繼資料。
- `GET /overlay-languages`：列出後端支援的疊圖語言。
- `POST /stream-playback`：建立單一攝影機的播放項目，供資料管理服務的播放介面呼叫。
- `POST /stream-playback/batch`：為攝影機牆建立最多 24 個播放項目。播放清單與攝影機牆只納入已啟用 `recognition_enabled` 的串流。
- `GET /stream-playback/sessions/{id}/index.m3u8`：透過固定網址讀取當前原始或標註畫面的 MediaMTX 播放清單，並將 `mt` 加入各片段網址。
- `GET /webrtc/ice-servers`：取得 WebRTC 用戶端的 STUN／TURN 設定。
- `GET /media-auth`：供 Nginx 的 `auth_request` 驗證 MediaMTX 路徑。

建立播放授權時會檢查使用者狀態和工地權限。短效授權可從 Redis 撤銷，讀取 HLS 片段時無須查詢 PostgreSQL。

## 必要設定

```dotenv
STREAMING_API_URL=http://127.0.0.1:8800
MEDIA_PUBLIC_HLS_BASE_URL=/hazard/media
MEDIA_PUBLIC_WEBRTC_BASE_URL=/hazard/media/webrtc
MEDIA_INTERNAL_HLS_BASE_URL=http://media-server:8888
MEDIA_OVERLAY_ALLOWED_LANGUAGES=zh-TW,en,zh-CN,ja,vi,id,fr,th
MEDIA_DEFAULT_OVERLAY_LANGUAGE=zh-TW
DATABASE_URL=postgresql+asyncpg://username:password@127.0.0.1/construction_hazard_detection
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=password
OIDC_ENABLED=true
OIDC_ISSUER_URL=https://sso.example.com/realms/visionnaire
OIDC_JWKS_URL=https://sso.example.com/realms/visionnaire/protocol/openid-connect/certs
OIDC_AUDIENCE=visionnaire-api
OIDC_ACCOUNT_URL=https://sso.example.com/realms/visionnaire/account
```

各 API 服務共用同一組 OIDC 簽發者、JWKS 網址與 API audience 設定。

## 直播影像路徑

MediaMTX 分別發布預覽與詳細畫質的 HLS 路徑。預覽由 `MEDIA_PREVIEW_*` 控制，預設為 640×360、15 FPS、500–700 kb/s。選定單一攝影機後才使用詳細畫質。後端在發布前畫上標註，並控制影格發布間隔。

Redis 保存精簡中繼資料及疊圖的需求、就緒狀態。SSE 與 WebSocket 傳送中繼資料，瀏覽器及行動裝置可直接播放串流，省去額外的 JPEG 解碼。

## MediaMTX 與 Nginx

公開的 MediaMTX 網址可交由 Nginx 的 `auth_request` 驗證。Nginx 會將媒體專用的 Bearer 權杖或 Cookie，以及 `X-Original-URI`，送到下列端點。

```text
GET /media-auth
```

後端從 Redis 讀取資料管理服務建立的播放授權，檢查指定工地、攝影機、畫面模式及使用者目前的工地權限，再由 Nginx 轉送請求至 MediaMTX。HLS 網址須帶短效媒體權杖 `mt`。主要 API 權杖、`token=` 查詢參數與舊的媒體工作階段 Cookie 均無法用於此端點。

若觀眾只需觀看即時影像，可使用較短的 HLS 播放視窗。

```dotenv
MTX_HLSSEGMENTDURATION=2s
MTX_HLSSEGMENTCOUNT=14
MTX_HLSALWAYSREMUX=no
MTX_HLSMUXERCLOSEAFTER=60s
```

## WebRTC ICE

區域網路測試可先使用 STUN。

```dotenv
STREAMING_WEBRTC_STUN_URLS=stun:stun.l.google.com:19302
```

網際網路觀看端可配置 TURN。以下範例使用 coturn REST 憑證。

```dotenv
STREAMING_WEBRTC_TURN_URLS=turn:example.com:3478?transport=udp,turn:example.com:3478?transport=tcp
STREAMING_WEBRTC_TURN_SHARED_SECRET=<same-secret-as-coturn>
STREAMING_WEBRTC_TURN_TTL_SECONDS=600
```

TURN 透過 UDP／TCP 中繼 ICE 流量，連線不經 HTTPS 反向代理。請為 coturn 設定共用密鑰、防火牆規則及適當的中繼連接埠範圍。

## 測試

```bash
pytest tests/examples/streaming_web -q
```

疊圖語言設定由 `OverlayLanguagePolicy` 共用：
`MEDIA_OVERLAY_ALLOWED_LANGUAGES` 與 `MEDIA_DEFAULT_OVERLAY_LANGUAGE` 都會將
`zh`、`zh_Hant`、`en-US` 等已知別名轉為標準代碼，並移除重複項目。
空清單、未知語言與不在允許清單中的預設語言會被拒絕，避免 API 與串流發布端
採用不同設定。串流服務啟動時驗證，發布端在每路串流的生命週期內保留
同一份不可變政策；修改環境設定後須重啟相關服務與串流。
此政策不更動來源、發布或預覽 FPS。
