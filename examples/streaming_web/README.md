🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Streaming Web Backend

FastAPI service for live-view metadata, stream discovery, MediaMTX access
authorisation, and WebRTC ICE settings. It does not stream video frames through
Redis or WebSocket. Video playback is served by MediaMTX through HLS or WebRTC.

## Runtime Role

```text
main.py / src/stream_processor.py
        |
        +--> publishes clean and annotated H.264 streams to MediaMTX
        +--> writes compact warning metadata to Redis

streaming_web
        |
        +--> lists labels and streams from PostgreSQL
        +--> returns MediaMTX HLS/WebRTC playback URLs
        +--> sends compact metadata through SSE/WebSocket
        +--> validates /media-auth requests before Nginx proxies MediaMTX
```

## Files

- `app.py`: FastAPI application and CORS setup.
- `routers.py`: stream listing, metadata, media auth, overlay, and ICE routes.
- `ws_handlers.py`: metadata-only WebSocket and SSE generators.
- `redis_service.py`: reads compact live warning metadata.
- `media_paths.py`: encodes and decodes MediaMTX path names.
- `overlay_renderer.py`: language-aware overlay rendering helpers.
- `webrtc_service.py`: STUN/TURN ICE-server response builder.
- `schemas.py`: response models.
- `utils.py`: small encoding and WebSocket helpers.

## Run

```bash
uvicorn examples.streaming_web.app:app \
  --host 127.0.0.1 \
  --port 8800 \
  --workers 2
```

## Endpoints

- `GET /labels`: labels for sites the authenticated user can access.
- `GET /streams/{label}`: stream IDs and HLS/WebRTC playback URLs.
- `GET /metadata/stream-id/{label}/{stream_id}`: SSE warning metadata.
- `WebSocket /ws/metadata-id/{label}/{stream_id}`: metadata-only WebSocket.
- `GET /overlay-languages`: available backend overlay languages.
- `POST /streams/{label}/{stream_id}/playback`: language and overlay-mode
  playback selection.
- `GET /webrtc/ice-servers`: STUN/TURN settings for WebRTC clients.
- `GET /media-auth`: Nginx `auth_request` endpoint for MediaMTX paths.

## Required Settings

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

The same `JWT_SECRET_KEY` must be shared with the database-management and
violation-record services.

## Live Video Path

- MediaMTX serves HLS and WebRTC.
- Annotated overlays are drawn before publishing to MediaMTX.
- Redis keeps compact metadata and overlay demand/ready keys only.
- SSE/WebSocket payloads are small metadata updates; they are not image frames.

This keeps browser and mobile clients from decoding extra JPEG payloads while
the backend controls frame pacing and overlay rendering.

## MediaMTX And Nginx

Public MediaMTX URLs should be protected by Nginx `auth_request`.
Nginx forwards `Authorization`, cookies, and `X-Original-URI` to:

```text
GET /media-auth
```

The backend validates the JWT and site access before Nginx proxies the HLS or
WebRTC request to MediaMTX.

Keep the HLS live window short unless rewind is required:

```dotenv
MTX_HLSSEGMENTDURATION=2s
MTX_HLSSEGMENTCOUNT=7
MTX_HLSALWAYSREMUX=no
MTX_HLSMUXERCLOSEAFTER=60s
```

## WebRTC ICE

For LAN-only testing, STUN may be enough:

```dotenv
STREAMING_WEBRTC_STUN_URLS=stun:stun.l.google.com:19302
```

For internet viewers, configure TURN. With coturn REST credentials:

```dotenv
STREAMING_WEBRTC_TURN_URLS=turn:example.com:3478?transport=udp,turn:example.com:3478?transport=tcp
STREAMING_WEBRTC_TURN_SHARED_SECRET=<same-secret-as-coturn>
STREAMING_WEBRTC_TURN_TTL_SECONDS=600
```

TURN relay traffic uses ICE over UDP/TCP and does not pass through the HTTPS
reverse proxy. Protect coturn with a shared secret, firewall rules, and a narrow
relay port range.

## Testing

```bash
pytest tests/examples/streaming_web -q
```
