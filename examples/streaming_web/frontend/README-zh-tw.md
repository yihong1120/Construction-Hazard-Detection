🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Streaming Web Frontend

Vite-based web client，用於瀏覽工地、選擇攝影機，以及觀看 MediaMTX live streams，同時
從 streaming backend 接收 compact warning metadata。

## 頁面

- `public/index.html`：label/site list。
- `public/label.html`：單一 label 的 stream list。
- `public/camera.html`：live camera view。
- `public/config.html`：viewer configuration page。

## 本機執行

```bash
cd examples/streaming_web/frontend
npm install
npm run dev
```

Vite development server 會在 `http://127.0.0.1:8888`。

預設 `/api` 會 proxy 到 `http://127.0.0.1:8800`。若 backend 在其他位置：

```bash
STREAMING_WEB_BACKEND_URL=http://127.0.0.1:8800 npm run dev
```

## Build

```bash
npm run build
```

build 結果會輸出到 `examples/streaming_web/frontend/dist`。

## Runtime 預期

- video playback URLs 來自 `GET /streams/{label}`。
- HLS/WebRTC media 由 MediaMTX 提供，不透過 WebSocket frame streaming。
- warning state 來自 SSE 或 metadata-only WebSocket endpoints。
- API calls 需要和 backend services 相同的 JWT。

## 檔案

- `public/js/index.js`：取得 labels。
- `public/js/label.js`：取得單一 label 的 streams。
- `public/js/camera.js`：播放 HLS/WebRTC 並訂閱 metadata。
- `public/js/config.js`：儲存 viewer preferences。
- `public/css/styles.css`：共用樣式。
- `vite.config.js`：dev server、proxy 與 build inputs。
