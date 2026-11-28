🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# 核心程式模組

`src/` 是 `main.py` 使用的正式執行模組。這裡負責擷取影像、執行 YOLO 推論、
產生安全警示、發布直播媒體、上傳違規紀錄，以及發送通知。

## 主流程

```text
stream_capture.py
    -> yolo_detector.py / yolo_worker.py
    -> danger_detector.py
    -> stream_processor.py
    -> media_stream_publisher.py / media_restreamer.py
    -> violation_sender.py
    -> notifiers/*
```

## 主要模組

- `stream_processor.py`：協調單一攝影機的工作，依觀看需求發布影像，並負責清理中繼資料和關閉資源。
- `stream_detection.py`：執行擷取與推論迴圈，產生警示事件及通知。
- `stream_overlay_frames.py`：管理唯讀影格與各語言的疊圖快取。
- `stream_capture.py`：從 RTSP、HTTP 或本機檔案擷取影格，限制緩衝量並處理來源重連。
- `yolo_worker.py`：提供跨程序的工作者與用戶端。影格存入 POSIX 共享記憶體，佇列只傳遞中繼資料。工作者載入一次模型，即可跨攝影機進行批次辨識。
- `yolo_detector.py`：提供串流處理程序使用的辨識介面，將影格送至共用 YOLO 工作者並追蹤辨識結果。
- `danger_detector.py`：將辨識結果轉成工地安全警示與管制區域的多邊形。
- `media_stream_publisher.py`：透過 FFmpeg 將原始影像或標註後的影像發布到 MediaMTX。
- `media_restreamer.py`：將來源串流直接轉送到 MediaMTX，無須等待辨識完成。
- `violation_sender.py`：將違規圖片及中繼資料上傳至違規紀錄 API。

## 共用工具

- `utils.py`：權杖、Redis、幾何運算、編碼等共用工具。
- `warning_types.py`：警示資料的型別。
- `model_fetcher.py`：模型下載與更新。
- `monitor_logger.py`：日誌設定。
- `stream_viewer.py`：用於人工檢查串流的 OpenCV 檢視器。

## 通知服務

`notifiers/` 支援 FCM、LINE Messaging API、Messenger、Telegram、WeChat 與 HTTP 廣播。FCM 使用經 OIDC 驗證的後端，其他平台使用各自的憑證。偵測與違規判斷由 `stream_detection.py` 處理。

執行 `uv sync --locked --extra social-notifications` 即可安裝多平台通知套件。可以直接呼叫通知服務，也可以透過 MCP 通知工具使用。攝影機工作程序使用 FCM。LINE 的設定請參閱 [Messaging API 指南](../docs/zh/line_notify_guide_zh.md)。

## 執行注意事項

- Redis 保存精簡的中繼資料、驗證快取、通知權杖快取及疊圖的協調狀態。
- MediaMTX 提供 HLS 與 WebRTC 直播。
- YOLO 工作者的佇列限制等待中的推論請求數。佇列已滿時，呼叫端會逾時或略過過期工作，以控制記憶體用量。
- 只在擷取、共享記憶體與 FFmpeg 交接時複製影格。

## 背景工作程序的身分驗證

`main.py` 會先確認 Redis 可用，再啟動攝影機和 GPU 程序。請將 `REDIS_PASSWORD` 設為伺服器密碼。Redis 在建立連線時驗證，後續讀寫沿用該連線。

工作程序使用 `WORKER_OIDC_CLIENT_ID` 與 `WORKER_OIDC_CLIENT_SECRET`，透過用戶端憑證取得並更新權杖。將 `WORKER_OIDC_TOKEN_ENDPOINT` 設為 realm 的權杖端點，或提供 `OIDC_ISSUER_URL`。除了本機迴送位址，端點一律使用 HTTPS。

每個租戶的工作程序使用各自的 Keycloak 機密用戶端。啟用用戶端驗證及服務帳號，關閉瀏覽器登入與直接密碼流程，並透過 audience mapper 加入 API 的 audience（通常是 `visionnaire-api`）。請為工作程序單獨配置用戶端，與 BFF、帳號連結工具分開。

將服務帳號的 Keycloak 使用者 ID（權杖中的 `sub`）透過 `UserIdentity` 連結至啟用中的本機工作帳號，並使用設定的身分提供者（通常是 `keycloak`）。依工作需求指派租戶、工地、群組與功能權限。驗證沿用 API 的授權規則，無須 realm 管理員角色。缺少本機身分對應時，即使持有用戶端密鑰也無法通過授權。既有 realm 須明確建立此用戶端，重新匯入 realm 不會更新既有用戶端。密鑰僅存放在執行環境。

成功取得權杖只能確認用戶端設定有效。完成身分連結後，應向目標 API 發出一次請求來驗證權限。Redis 連線失敗或缺少工作程序憑證時，啟動流程會在建立攝影機程序前停止。
