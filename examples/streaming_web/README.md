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

- `app.py`: FastAPI application setup.
- `routers.py`: HTTP and WebSocket declarations with dependency injection only.
- `application_services.py`: request-scoped application-service composition.
- `streaming_api_service.py`: media authorisation and playback API policy.
- `playback_service.py`: playback sessions, playlists, and producer demand leases.
- `stream_catalog_service.py`: authorised site and stream catalogues.
- `streaming_metadata_service.py`: authorised SSE and WebSocket response setup.
- `streaming_metadata_handlers.py`: shared metadata SSE and WebSocket transports.
- `redis_service.py` and `metadata_keys.py`: compact metadata Redis contract.
- `media_paths.py`: canonical MediaMTX paths, public URLs, and demand keys.
- `overlay_renderer.py`: localised, cached overlay rendering helpers.
- `webrtc_service.py`: STUN/TURN ICE-server response builder.
- `schemas.py`: validated HTTP and Redis payload contracts.
- `nginx.hazard-media.conf`: Nginx media proxy and authorisation configuration.

## Run

```bash
uvicorn examples.streaming_web.app:app \
  --host 127.0.0.1 \
  --port 8800 \
  --workers 2 \
  --timeout-graceful-shutdown 10
```

## Endpoints

- `GET /labels`: labels for sites the authenticated user can access.
- `GET /streams/{label}`: stream IDs and HLS/WebRTC playback URLs.
- `GET /metadata/stream-id/{label}/{stream_id}`: SSE warning metadata.
- `WebSocket /ws/metadata-id/{label}/{stream_id}`: metadata-only WebSocket.
- `GET /overlay-languages`: available backend overlay languages.
- `POST /stream-playback`: low-level single-camera playback primitive called
  by the db_management playback facade.
- `POST /stream-playback/batch`: creates up to 24 playback items for a
  multi-camera wall.
- Playback listings and walls include only streams with `recognition_enabled`
  enabled; disabled streams do not reserve a wall tile.
- Media playback uses a short-lived, revocable Redis capability. Its user
  status and site access are checked when the capability is issued, so HLS
  segment authorisation does not query PostgreSQL.
- `GET /stream-playback/sessions/{id}/index.m3u8`: stable playlist endpoint
  that fetches the current MediaMTX playlist and appends `mt` to fragment URLs.
- `GET /webrtc/ice-servers`: STUN/TURN settings for WebRTC clients.
- `GET /media-auth`: Nginx `auth_request` endpoint for MediaMTX paths.

## Required Settings

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

The same `JWT_SECRET_KEY` must be shared with the database-management and
violation-record services.

## Live Video Path

- MediaMTX serves HLS and WebRTC.
- Detail and preview HLS paths are published independently. Preview is capped
  by `MEDIA_PREVIEW_*` (640x360, 15 FPS, 500–700 kb/s by default), while a
  selected camera retains the detail rendition.
- Annotated overlays are drawn before publishing to MediaMTX.
- Redis keeps compact metadata and overlay demand/ready keys only.
- SSE/WebSocket payloads are small metadata updates; they are not image frames.

This keeps browser and mobile clients from decoding extra JPEG payloads while
the backend controls frame pacing and overlay rendering.

## MediaMTX And Nginx

Public MediaMTX URLs should be protected by Nginx `auth_request`.
Nginx forwards the dedicated media bearer/cookie and `X-Original-URI` to:

```text
GET /media-auth
```

The backend resolves the opaque media capability created by the db_management
playback facade from Redis, enforces its exact site/camera/profile scope, and
rechecks current user/site access before Nginx proxies the request to MediaMTX.
HLS URLs must carry the short-lived `mt` media token. Main access tokens,
`token=` query strings, and old media-session cookies are no longer accepted.

Keep the HLS live window short unless rewind is required:

```dotenv
MTX_HLSSEGMENTDURATION=2s
MTX_HLSSEGMENTCOUNT=14
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
