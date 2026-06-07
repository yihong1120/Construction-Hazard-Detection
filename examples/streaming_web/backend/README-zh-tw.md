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

streaming_web/backend
        |
        +--> 從 PostgreSQL 列出 labels 與 streams
        +--> 回傳 MediaMTX HLS/WebRTC playback URLs
        +--> 透過 SSE/WebSocket 傳送 compact metadata
        +--> Nginx proxy MediaMTX 前先驗證 /media-auth
```

## 檔案

- `app.py`：FastAPI application 與 CORS 設定。
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
uvicorn examples.streaming_web.backend.app:app \
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
- `POST /streams/{label}/{stream_id}/playback`：選擇 playback language 與 overlay mode。
- `GET /webrtc/ice-servers`：WebRTC clients 使用的 STUN/TURN settings。
- `GET /media-auth`：給 Nginx `auth_request` 驗證 MediaMTX paths。

## 必要設定

```dotenv
STREAMING_API_URL=http://127.0.0.1:8800
MEDIA_PUBLIC_HLS_BASE_URL=/hazard/media
MEDIA_PUBLIC_WEBRTC_BASE_URL=/hazard/media/webrtc
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
- Annotated overlay 會在發布到 MediaMTX 前畫好。
- Redis 只保存 compact metadata 與 overlay demand/ready keys。
- SSE/WebSocket payload 是小型 metadata，不是影像 frame。

這樣可以避免 browser 與 mobile clients 額外解碼 JPEG payload，同時讓後端控制 frame pacing
與 overlay rendering。

## MediaMTX 與 Nginx

公開 MediaMTX URL 建議放在 Nginx `auth_request` 後面。Nginx 將 `Authorization`、
cookies 與 `X-Original-URI` 轉送至：

```text
GET /media-auth
```

backend 會先驗證 JWT 與 site access，再讓 Nginx proxy HLS 或 WebRTC request 到
MediaMTX。

除非需要 rewind，HLS live window 請保持短：

```dotenv
MTX_HLSSEGMENTDURATION=2s
MTX_HLSSEGMENTCOUNT=7
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
pytest tests/examples/streaming_web/backend -q
```
