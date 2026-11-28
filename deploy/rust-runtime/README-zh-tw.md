🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# Construction Rust 執行環境預覽版

這個獨立發行版本包含攝影機流程和預覽 HTTP API，兩者都由 Rust 執行檔直接啟動。請用隔離的資料庫、Redis 和身分驗證 realm 驗證。PostgreSQL、Redis、Keycloak、MediaMTX 與反向代理仍是基礎服務。

| 元件 | 預覽版範圍 |
| --- | --- |
| `construction-pipeline` | CPU 重播或 TensorRT 推論、追蹤、危害分析、標註、串流發布及 PostgreSQL outbox |
| `construction-api` | 健康與就緒檢查、證據上傳與存取、工地清單、WebRTC ICE 設定、SSE 與 WebSocket 即時中繼資料，以及可選的媒體授權及 TensorRT HTTP／WebSocket 推論 |
| 可選移植功能 | OIDC BFF、單鏡頭與多鏡頭播放、帳號與工地管理、通知中心及 FCM、PostgreSQL 違規歷史與審核 |

systemd 範例直接執行這兩個程式，並使用隔離的資料庫 schema、連接埠、帳號和路徑。可在還原的應用程式資料庫副本上驗證可選移植功能。離線開發及主機管理工具不在這個執行環境的範圍內。

執行檔由 `native/` 的 Rust 原始碼建置，連結原生 SDK 函式庫。`native/postprocess-core/` 提供共用的追蹤與危害演算法，`native/stream-contract/` 提供共用的 Redis 中繼資料鍵值、識別碼編碼及標註色盤。`main.py` 及既有服務使用的 PyO3 相容層位於 `bindings/python/`，另行建置。

| Python 入口 | 此版本對應的 Rust 程式 |
| --- | --- |
| `main.py` 和 `src/` 的攝影機模組 | `native/pipeline/` 由 JSON 或 PostgreSQL 讀取攝影機設定，套用個別上下班時間，並透過 PostgreSQL outbox 保存事件。 |
| `examples/streaming_web/` 和 `examples/db_management/routers/playback.py` | `native/api/src/streaming/` 提供工地查詢、ICE、即時中繼資料、HLS 工作階段、觀看租約與 Flutter 播放 API。 |
| `examples/db_management/` | `native/api/src/accounts.rs` 查詢管理員所屬群組內的帳號、群組與工地，並管理工地成員。 |
| `examples/bff/` | `native/api/src/bff/` 處理 OIDC 登入、工作階段更新、CSRF 防護、登出及 API 轉送。 |
| `examples/local_notification_server/` | `native/api/src/notifications/` 處理加密的裝置登錄、通知中心、工地偏好及 FCM HTTP v1 派送。 |
| `examples/YOLO_server_api/` | `native/api/src/detection/` 提供設定中的引擎模型及圖片推論，並限制同時受理的工作數。 |
| `examples/violation_records/` | `native/api/src/violations/` 使用既有歷史紀錄與審核資料表。基本預覽版的證據 API 使用隔離 schema。 |

## 建置發行版本

準備 Linux、Rust 1.93.1、C 編譯器、`pkg-config` 及 FFmpeg 開發函式庫。Debian 或 Ubuntu 使用 `libavformat-dev`、`libavcodec-dev`、`libavutil-dev` 和 `libswscale-dev`。開發及執行時函式庫須使用相同主版本。非系統 SDK 可指定 `FFMPEG_ROOT`。需要多語標註時請安裝 Noto 字型。

攝影機流程透過程序內的 libavformat 探測來源尺寸。攝影機設定與目錄快取都沒有尺寸資料時才會探測。

從儲存庫根目錄執行，並指定新的發行目錄。

```bash
deploy/rust-runtime/build.sh /tmp/construction-rust-preview-cpu
```

需要 GPU 推論時，在輸出目錄前加上 `--tensorrt`。設定 `TRT_INCLUDE`、`TRT_LIB`、`CUDA_INCLUDE` 和 `CUDA_LIB` 指向相容的 SDK。若 OpenCV 安裝於非系統路徑，請設定 `OPENCV_ROOT`。建置連結階段與執行環境都須能找到動態函式庫與其相依套件。使用非系統 SDK 時，可在建置環境與服務環境檔設定 `LD_LIBRARY_PATH`。請使用相容於部署 GPU 和 TensorRT 版本的現有引擎。CPU 版可執行危害分析或重播已有的辨識結果。

```bash
deploy/rust-runtime/build.sh --tensorrt /tmp/construction-rust-preview-gpu
```

新目錄中包含 `bin/`、位於 `config/` 的 API 設定範例、`docs/pipeline/` 的攝影機說明及 `systemd/` 服務範例。Cargo 使用既有鎖檔建置兩個執行檔，成功後才發布發行目錄。

## 在本機啟動 API

將 `config/api.example.json` 複製至私有設定檔。讓預覽 API 繼續監聽本機迴路位址，並設定隔離 realm 的簽發者、JWKS 網址及受眾。依授權對象設定 `sub`、`azp`、權限及可存取的工地。透過程序環境提供 `RUST_API_DATABASE_URL` 與 `RUST_API_REDIS_URL`。初始化資料庫 schema 時須提供資料庫連線，該角色也要能建立預覽 schema。

```bash
release=/tmp/construction-rust-preview-cpu
api_config=/path/to/private-api.json
"$release/bin/construction-api" --config "$api_config" --init-db
"$release/bin/construction-api" --config "$api_config"
```

另外開啟終端機測試下列端點。存取證據需要符合設定受眾及授權的已簽章存取權杖。

```bash
curl --fail http://127.0.0.1:18800/health
curl --fail http://127.0.0.1:18800/ready
```

觀看即時中繼資料時，為攝影機所在的工地授予 `stream_read`。`GET /labels` 可查詢該授權涵蓋的工地。讓 API 的 `metadata_prefix` 和攝影機流程的 `redis.prefix` 相同。網址使用工地名稱與攝影機名稱的 URL-safe Base64 編碼，保留尾端補位。`/metadata/stream-id/{site}/{stream_id}` 使用 SSE，`/ws/metadata-id/{site}/{stream_id}` 使用 WebSocket。同一台鏡頭的觀看者共用 Redis 讀取器，API 重用現有 Redis 連線查詢訂閱狀態。網路較慢的用戶端接收最新資料。

Flutter 選擇 `overlay` 與觀看語言，直接播放 PostgreSQL 目錄管理的鏡頭所輸出的後端標註影片。CPU 解碼時設定 `capture.host_yuv: true`，讓後端直接在編碼器的 YUV 平面繪圖。Rust 播放 API 的多鏡頭牆最多支援 32 路。

```bash
curl --no-buffer --fail -H "Authorization: Bearer $PREVIEW_ACCESS_TOKEN" \
  'http://127.0.0.1:18800/metadata/stream-id/demo/Y2FtZXJhLTE='
```

使用 `GET /webrtc/ice-servers` 時，將 `STREAMING_WEBRTC_STUN_URLS` 設成逗號分隔的 STUN 網址。要發行有期限的 TURN 憑證，再設定 `STREAMING_WEBRTC_TURN_URLS`、`STREAMING_WEBRTC_TURN_SHARED_SECRET` 和 `STREAMING_WEBRTC_TURN_TTL_SECONDS`。這個端點也需要 `stream_read`。

WebSocket 握手接受 `Authorization: Bearer` 標頭，瀏覽器可改用 `token` 查詢參數。請設定反向代理及存取紀錄省略這些路由的查詢字串。`WS_MAX_SESSION_SECONDS` 控制 WebSocket 最長連線時間，預設為 1800 秒。服務會重新驗證連線中的權杖。

### 驗證移植後的 API

以發行包中的 `config/api.migration.example.json` 為範本。將範例值換成隔離的 Keycloak realm，以及還原後 PostgreSQL 中啟用的租戶與部署。身分透過 `user_identities.provider_user_id` 繫結。請同時設定 PostgreSQL 角色及工地範圍，並提供對應的 API 授權。先執行一次 `--init-db` 建立 `rust_api` 預覽證據 schema。移植設定使用既有的應用程式資料表，請先完成其 schema 版本更新。

- 啟用 `account_management` 並設定 `deployment_scope`，即可查詢所屬群組的帳號和工地，以及增減工地成員。清單使用 `account_read`，成員修改使用 `account_manage`。
- 在 Keycloak 註冊 `/bff/auth/oidc/callback` 回呼網址。公開 API 根網址須與部署資料列相符。將 `/bff/` 路由到 Rust API，透過環境變數提供用戶端密鑰和至少 32 位元組的獨立工作階段加密密鑰。每個已啟用的上游都可指向 Rust 監聽位址，BFF 會帶入已簽章的使用者權杖。
- 若要啟用以個人額度使用公司助理的聊天入口，將 `bff.upstreams.company_agent_chat` 指向介接服務的本機回環位址，例如 `http://127.0.0.1:28082/api/visionnaire/chat`。前端可於 `/bff/chat/v1` 操作聊天室、訊息、附件、工作狀態、取消及成果下載，沿用現有工作階段和 CSRF 驗證。介接服務需驗證此存取權杖的簽發者和受眾。原聊天 API 繼續使用 `bff.upstreams.chat`。
- 設定 `media_auth` 和 `playback`。將播放清單路由到 Rust API，媒體路由透過反向代理呼叫 `/media-auth`。讓 `playback.demand_prefix` 和 `metadata_prefix` 與攝影機流程使用相同 Redis 前綴。設定 `deployment_scope` 後，鏡頭清單、新播放與連線中的中繼資料會重新確認 PostgreSQL 工地成員資格。已發行的媒體授權有期限，續期需通過驗證。
- 應用程式資料表位於其他 schema 時，請設定 `RUST_API_DATABASE_SCHEMA`。PostgreSQL 會先搜尋指定的 schema，再搜尋 `public`。請依啟用的路由授予資料庫角色必要的存取權限。
- 設定 `violation_history` 可啟用歷史、統計、證據圖片、回饋與含稽核的審核。將 `write_uploads` 設為 `true` 時，Rust `/upload` 會寫入部署的 `violations` 資料表。使用現有的 `evidence_write_operations` 日誌及已發布或封存模型目錄，並讓 API 服務帳號和 systemd 單元可寫入 `media_root` 的絕對路徑。攝影機工作者需要 `evidence_write`，已繫結且啟用的 PostgreSQL 帳號還需具備工地和模型權限。
- 用 `notifications.token_key_env` 指定現有的 Fernet 密鑰環境變數。要啟用派送，在 `notifications` 加入 `"fcm": {"service_account_path_env": "GOOGLE_APPLICATION_CREDENTIALS"}`，並讓該變數指向隔離 Firebase 服務帳號檔案。通知中心和裝置使用 `notification_read`。[Firebase HTTP v1 文件](https://firebase.google.com/docs/cloud-messaging/send/v1-api)

帳號建立、核准、重設密碼、角色變更、建立與刪除群組或工地，以及模型管理仍由既有服務處理。

### 派送違規通知

在 API 設定的 `grants[].permissions` 為攝影機工作者加入 `notification_dispatch`，並在 `grants[].sites` 列出目標工地。工作者的 Bearer 存取權杖須符合該授權項目的 `subject`（`sub`）與 `client_id`（`azp`）。

以 JSON 呼叫 `POST /send_fcm_notification`。填入工地的 `site_id`，以及該工地成功呼叫 `POST /upload` 後回傳的 `violation_id`。兩者均須為 JSON 正整數。`site` 須與 PostgreSQL 目前的工地名稱及呼叫端授權相符。重試時使用相同的兩個 ID，API 會依 PostgreSQL 派送收據略過 FCM 已接受的派送。

```bash
curl --fail -H "Authorization: Bearer $PREVIEW_ACCESS_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"site":"demo","site_id":1,"violation_id":123,"stream_name":"Camera 1",
       "type":"violation","title":"Safety violation alert",
       "body":{"warning_no_hardhat":{"count":1}},
       "deep_link":"/violations?violation_id=123","metadata":{"violation_id":123}}' \
  http://127.0.0.1:18800/send_fcm_notification
```

若要測試登入者自己已註冊的裝置，以具 `notification_read` 權限的存取權杖呼叫 `POST /notifications/test`。此請求的本文留空。

攝影機派送程式會從上傳回應的 `violation_id` 讀取 JSON 正整數。啟用通知派送時，通知端點須回傳 `outcome`，值為 `completed`、`skipped`、`retryable` 或 `permanent_failure`。

### 可選的 TensorRT 圖片推論 API

使用 `--tensorrt` 建置的版本時，可在 API 設定加入 `detection`，指定 CUDA `device`、選用的 `default_model_id` 及 `models`。每個模型包含 `id`、`display_name`、`version`、`engine` 的 `.engine` 絕對路徑、`sha256`、`imgsz`、`classes` 和 `allowed_subjects`。類別包含 `id`、`code`、`display_name` 與 `#RRGGBB` 的 `color`。呼叫端的權杖主體須有 `image_detection` 權限，並列在該模型的授權清單內。設定模型後，預覽版提供 `GET /models`、`POST /detect`、`POST /v2/detect` 和 `/ws/detect`。

WebSocket 用戶端可使用 `X-Model-Key` 及選用的 `X-Model-Version` 標頭選擇模型，也可用 `{"model_key":"preview-model"}` 作為首則文字訊息。收到含 `"status":"ready"` 的訊息後，每次以一則二進位訊息傳送 JPEG 或 PNG。HTTP 與 WebSocket 共用有容量上限的推論佇列，攝影機流程使用自己的模型工作者。

預覽版 `/v2/detect` 支援物件模式，使用 `view=objects` 及 `compute_regions=false`。上傳 JPEG 或 PNG，並指定已設定的模型 ID。

```bash
curl --fail -H "Authorization: Bearer $PREVIEW_ACCESS_TOKEN" \
  -F 'image=@/path/to/test.jpg' -F 'model=preview-model' \
  -F 'view=objects' -F 'compute_regions=false' \
  http://127.0.0.1:18800/v2/detect
```

## 啟動攝影機流程

依發行包的 `docs/pipeline/README-zh-tw.md` 或儲存庫的 `native/pipeline/README-zh-tw.md` 準備獨立攝影機 JSON 設定，使用獨立的絕對 `state_dir` 路徑及隔離的 PostgreSQL 資料庫。若 `outbox.url_env` 指定 `RUST_OUTBOX_DATABASE_URL`，請用這個環境變數傳入連線。啟動擷取前先驗證設定並檢查 outbox。

```bash
release=/tmp/construction-rust-preview-gpu
pipeline_config=/path/to/private-pipeline.json
"$release/bin/construction-pipeline" validate --config "$pipeline_config"
"$release/bin/construction-pipeline" outbox --config "$pipeline_config"
"$release/bin/construction-pipeline" run --config "$pipeline_config"
```

要隔離驗證證據派送，可將 `delivery.upload_url` 指向 Rust `/upload`，設定 `token_url` 與憑證環境變數名稱，並為工作者與所屬工地授權 `evidence_write`。要驗證通知與歷史紀錄使用相同 ID，啟用 `violation_history.write_uploads`，並讓 `delivery.notification_url` 指向 Rust 的 `/send_fcm_notification`。未指定通知端點時，攝影機流程在上傳成功後即可完成 outbox 事件。

## systemd 範例

服務範例以 `/opt/construction-rust-preview/current` 指定發行版本，並使用 `chd-rust-preview-api`、`chd-rust-preview-pipeline` 兩個服務帳號與位於 `/etc/construction-rust-preview/` 的私有設定及環境檔。啟動 API 服務前先初始化資料庫 schema。攝影機狀態與輸出放在 `/var/lib/construction-rust-preview-pipeline`，並讓對應的服務帳號能讀取模型。TensorRT 版也需要服務帳號能存取 GPU 裝置及原生函式庫。先透過隔離的服務單元驗證發行版本，再調整正式路由。
