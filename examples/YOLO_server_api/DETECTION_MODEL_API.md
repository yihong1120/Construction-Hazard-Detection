# 模型目錄與偵測 API

YOLO 服務提供 `GET /models`、`POST /detect` 與 `WS /ws/detect`。
Web 經由 `/bff/detection/`，native 經由部署設定的 detection API 前綴。
Web 使用 BFF session，native 使用 Bearer JWT。

模型清單、預設模型與權限來自 PostgreSQL 發布目錄。清單不載入模型，
依 tenant、role、enabled 與 capability 過濾，回應包含不可變 `version`
及每個類別的 `id`、穩定 `code`、顯示名稱與顏色。沒有可用的預設模型時
`default_model_id` 為 `null`；沒有候選模型時 `models` 為空陣列。
回應使用 `Cache-Control: private, no-store`。

`POST /detect` 接受 multipart `image`、`model` 與可選的 `model_version`。
成功回應維持 `[[x1,y1,x2,y2,confidence,class_id], ...]`。
後處理依所選版本的語意 code 建立互斥關係，不假設固定數字 ID；沒有
`helmet/no_helmet` 或 `safety_vest/no_safety_vest` 成對定義的類別不套用該規則。

不存在、停用、版本不符或載入失敗的模型回傳 409 `MODEL_UNAVAILABLE`；
無權限回傳 403。模型目錄不可用回傳 503 `MODEL_REGISTRY_UNAVAILABLE`。
輸入過大回傳 413；共用工作額度用盡回傳 503 `INFERENCE_BUSY`。

HTTP 與 WebSocket 共用有界 runtime。圖片解碼、模型載入、同步推論與
後處理均在其工作池執行；同一模型序列推論。不可變版本快取使用 LRU，
載入中與使用中模型都計入容量，只有沒有租約的模型可淘汰。
WebSocket 握有輕量版本選擇資訊，每幀才取得模型租約；閒置連線不釘住快取。

設定：

- `YOLO_API_INFERENCE_CONCURRENCY`：工作執行緒，預設 4。
- `YOLO_API_QUEUE_CAPACITY`：執行中及排隊工作總數，預設 8。
  保留 `DETECT_INGRESS_CONCURRENCY` 作為相同設定的舊名稱。
- `MAX_LOADED_MODELS`：載入中及已載入模型上限，預設 3。
- `DETECT_MAX_UPLOAD_BYTES`：HTTP 與 WebSocket 共用的壓縮影像上限，預設 20 MiB。
- `WS_MAX_SESSION_SECONDS`：整段 WebSocket 會話期限，包含認證、選模、
  閒置收幀及推論等待。到期後關閉連線並取消待執行工作。

取消排隊工作會釋放額度；已進入原生函式的執行緒無法強制中止，因此
額度與模型租約保留到原生工作實際結束，避免取消導致超額執行。
應用程式關閉時取消排隊工作、等待執行中的工作並清理模型與暫存 artifact。
舊檔名快取、檔案監聽與 `MODEL_VARIANTS`／`DEFAULT_MODEL_ID` 等設定已移除。
模型發布、復原與權重完整性規則見 [MODEL_CATALOG.md](MODEL_CATALOG.md)。

測試以替身驗證 API、排程、取消與類別語意，沒有執行正式 GPU 模型或 Flutter 聯調。

同一版本的排程與工作執行緒額度分開管理：同模型等待者先在事件迴圈排隊，
不占用原生工作執行緒；其他模型可使用剩餘額度。取消等待者會立即釋放排隊
額度，已進入原生工作的請求則保留資源直到完成。應用關閉時會取消等待者。
本機認證略過分支同樣驗證 `x-model-version`，版本不符不會改載入目前版本。

Image headers and decoded dimensions are validated in the bounded inference
worker before acquiring a model lease. `DETECT_MAX_IMAGE_PIXELS` defaults to
67,108,864 pixels. Invalid or undecodable input returns HTTP 400 `INVALID_IMAGE`;
a header exceeding the pixel limit returns `IMAGE_DIMENSIONS_EXCEEDED` before
native decode. WebSocket detection uses the same error code and `http_status`
field and closes invalid-input sessions with code 1003. Overlay responses reuse
the decoded image dimensions instead of decoding the request a second time.

`DETECT_MAX_DECODED_BYTES` limits active decoded BGR inputs across API inference
workers (default 256 MiB). After parsing the image header, the runtime reserves
width × height × 3 bytes before native decode and retains the reservation through
model loading and inference, including cancellation of the caller. An image
that cannot fit alone returns HTTP 413 `IMAGE_MEMORY_LIMIT_EXCEEDED`; a temporary
aggregate shortage returns HTTP 503 `INFERENCE_MEMORY_BUSY`. WebSocket input
size rejection uses close code 1009. This budget covers active decoded inputs;
model memory, internal copies/tensors and GPU allocations require separate
capacity planning. Source, publish and preview FPS remain unchanged.

Overlay geometry has its own process-owned thread pool: two native workers and
at most eight admitted jobs, including queued jobs. Exhaustion returns HTTP 503
`OVERLAY_BUSY`. Caller cancellation marks queued work to skip and leaves its
capacity reserved until the queue actually drains; running work holds admission
until native completion. Application shutdown stops admission, cancels queued
work and drains native workers. Rendering receives one typed work record rather
than forwarding its fields through the scheduling layer. Source FPS is unchanged.

Model downloads use a verified artifact record carrying path, SHA256, model ID
and version. Verification caches the checksum by file identity, size, mtime and
ctime; a changed file must pass validation again. Conditional 304 responses reuse
the same verified checksum, without a second full file read. Permission checks
still precede artifact access.

HTTP 與 WebSocket 的共用推論入口會在排程及模型載入前檢查壓縮影像大小。
`python -m examples.YOLO_server_api.app --host 0.0.0.0 --port 6000 --workers 1`
會將同一個 `DETECT_MAX_UPLOAD_BYTES` 設為 Uvicorn 的 `ws_max_size`；Docker
與 dev runner 也使用此設定。若自行以 `uvicorn` 啟動，需另外傳入相同的
`--ws-max-size`，使傳輸層與應用層上限一致。

Overlay 與證據影像處理共用 `NativeWorkPool` 實作，各自持有獨立的有界工作池。
取消請求不會提早釋放仍在執行的原生工作額度。歷史與即時 overlay 共用框的
端點裁切規則，先將兩端限制於影像範圍，再計算正規化寬高。
