🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Streaming Web Frontend

Vite-based web client for browsing sites, selecting cameras, and viewing
MediaMTX live streams with compact warning metadata from the streaming backend.

## Pages

- `public/index.html`: label/site list.
- `public/label.html`: stream list for one label.
- `public/camera.html`: live camera view.
- `public/config.html`: viewer configuration page.

## Run Locally

```bash
cd examples/streaming_web/frontend
npm install
npm run dev
```

The Vite development server listens on `http://127.0.0.1:8888`.

By default, `/api` is proxied to `http://127.0.0.1:8800`. Override it when the
backend runs elsewhere:

```bash
STREAMING_WEB_BACKEND_URL=http://127.0.0.1:8800 npm run dev
```

## Build

```bash
npm run build
```

Built files are written to `examples/streaming_web/frontend/dist`.

## Runtime Expectations

- Video playback URLs come from `GET /streams/{label}`.
- HLS/WebRTC media is served by MediaMTX, not by WebSocket frame streaming.
- Warning state comes from SSE or metadata-only WebSocket endpoints.
- API calls require the same JWT used by the backend services.

## Files

- `public/js/index.js`: fetches labels.
- `public/js/label.js`: fetches streams for one label.
- `public/js/camera.js`: plays HLS/WebRTC and subscribes to metadata.
- `public/js/config.js`: stores viewer preferences.
- `public/css/styles.css`: shared styling.
- `vite.config.js`: dev server, proxy, and build inputs.
