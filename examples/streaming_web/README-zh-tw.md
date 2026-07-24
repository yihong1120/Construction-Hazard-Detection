🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Streaming Web Backend

用於 live-view metadata、stream discovery、MediaMTX 存取授權與 WebRTC ICE 設定的
FastAPI 服務。它不透過 Redis 或 WebSocket 傳送 video frame；影像播放由 MediaMTX 透過
HLS 或 WebRTC 提供。

## Runtime 角色

```text
main.py / src/stream_processor.py
        |
        +--> 發布 clean 與 annotated H.264 streams 到 MediaMTX
        +--> 將 compact warning metadata 寫入 Redis

streaming_web
        |
        +--> 從 PostgreSQL 列出 labels 與 streams
        +--> 回傳 MediaMTX HLS/WebRTC playback URLs
        +--> 透過 SSE/WebSocket 傳送 compact metadata
        +--> Nginx proxy MediaMTX 前先驗證 /media-auth
```

## 檔案

- `app.py`：FastAPI application 設定。
- `routers.py`：stream listing、metadata、media auth、overlay 與 ICE routes。
- `ws_handlers.py`：metadata-only WebSocket 與 SSE generators。
- `redis_service.py`：讀取 compact live warning metadata。
- `media_paths.py`：編碼與解碼 MediaMTX path names。
- `overlay_renderer.py`：支援多語 overlay rendering helpers。
- `webrtc_service.py`：STUN/TURN ICE-server response builder。
- `schemas.py`：response models。
- `utils.py`：小型 encoding 與 WebSocket helpers。

## 執行

```bash
uvicorn examples.streaming_web.app:app \
  --host 127.0.0.1 \
  --port 8800 \
  --workers 2
```

## Endpoints

- `GET /labels`：登入使用者可存取的 site labels。
- `GET /streams/{label}`：stream IDs 與 HLS/WebRTC playback URLs。
- `GET /metadata/stream-id/{label}/{stream_id}`：SSE warning metadata。
- `WebSocket /ws/metadata-id/{label}/{stream_id}`：metadata-only WebSocket。
- `GET /overlay-languages`：可用的後端 overlay languages。
- `POST /stream-playback`：底層單鏡頭 playback primitive，由 db_management
  playback facade 呼叫。
- `POST /stream-playback/batch`：為多鏡頭牆建立最多 24 個 playback items。
- 播放清單與播放牆只會回傳已開啟 `recognition_enabled` 的串流；關閉的串流不會
  佔用播放牆站位。
- 播放使用短效、可撤銷的 Redis capability；建立 capability 時會驗證使用者狀態
  與工地權限，因此 HLS segment 授權不會查詢 PostgreSQL。
- `GET /stream-playback/sessions/{id}/index.m3u8`：stable playlist endpoint，會
  依目前 clean/overlay 狀態取 MediaMTX playlist，並將 `mt` 補到 fragment URL。
- `POST /streams/{label}/{stream_id}/playback`：選擇 playback language 與 overlay mode。
- `GET /webrtc/ice-servers`：WebRTC clients 使用的 STUN/TURN settings。
- `GET /media-auth`：給 Nginx `auth_request` 驗證 MediaMTX paths。

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
JWT_SECRET_KEY=replace-with-a-long-random-secret
```

`JWT_SECRET_KEY` 必須和 database-management、violation-record services 共用。

## 直播影像路徑

- MediaMTX 負責 HLS 與 WebRTC。
- detail 與 preview HLS path 分開發布。preview 預設由 `MEDIA_PREVIEW_*` 限制為
  640x360、15 FPS、500–700 kb/s；使用者選定的單鏡頭才使用 detail rendition。
- Annotated overlay 會在發布到 MediaMTX 前畫好。
- Redis 只保存 compact metadata 與 overlay demand/ready keys。
- SSE/WebSocket payload 是小型 metadata，不是影像 frame。

這樣可以避免 browser 與 mobile clients 額外解碼 JPEG payload，同時讓後端控制 frame pacing
與 overlay rendering。

## MediaMTX 與 Nginx

公開 MediaMTX URL 建議放在 Nginx `auth_request` 後面。Nginx 將獨立的 media
bearer/cookie 與 `X-Original-URI` 轉送至：

```text
GET /media-auth
```

backend 會從 Redis 解析 db_management playback facade 建立的 opaque media
capability，強制限制指定的 site/camera/profile，並重新檢查目前 user/site 權限，
再讓 Nginx proxy request 到 MediaMTX。HLS URL 必須帶短效 `mt` media token；
主 access token、`token=` query、舊 media-session cookie 都不再接受。

除非需要 rewind，HLS live window 請保持短：

```dotenv
MTX_HLSSEGMENTDURATION=2s
MTX_HLSSEGMENTCOUNT=14
MTX_HLSALWAYSREMUX=no
MTX_HLSMUXERCLOSEAFTER=60s
```

## WebRTC ICE

LAN 測試時 STUN 可能已足夠：

```dotenv
STREAMING_WEBRTC_STUN_URLS=stun:stun.l.google.com:19302
```

Internet viewers 建議設定 TURN。若使用 coturn REST credentials：

```dotenv
STREAMING_WEBRTC_TURN_URLS=turn:example.com:3478?transport=udp,turn:example.com:3478?transport=tcp
STREAMING_WEBRTC_TURN_SHARED_SECRET=<same-secret-as-coturn>
STREAMING_WEBRTC_TURN_TTL_SECONDS=600
```

TURN relay traffic 使用 ICE over UDP/TCP，不會經過 HTTPS reverse proxy。請使用 shared
secret、firewall rules 與較窄的 relay port range 保護 coturn。

## 測試

```bash
pytest tests/examples/streaming_web -q
```
