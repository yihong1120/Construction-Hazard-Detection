# Construction Hazard Detection MCP

[English](README.md) | [繁體中文](README-zh-tw.md)

MCP 是 AI 客戶端存取偵測、違規紀錄、模型與多平台通知的選用入口。
它沿用現有 API、OIDC 授權與工地權限，不另建帳號或資料庫。

## 安裝與啟動

從 repo 根目錄執行：

```bash
uv sync --extra mcp
MCP_TRANSPORT=stdio uv run --no-sync python -m examples.mcp_server.main
```

僅需查詢、上傳、FCM、模型下載與幾何工具時，不需安裝 GPU 或影像套件。
本機推論及危害分析另需 `uv sync --extra mcp --extra yolo`；
串流截圖另需 `--extra streaming` 或 `--extra yolo`。
重型依賴在呼叫相關工具時才載入。

HTTP 模式：

```bash
MCP_TRANSPORT=streamable-http uv run --no-sync python -m examples.mcp_server.main
```

預設網址 `http://127.0.0.1:8092/mcp`。另支援 `sse` transport，路徑 `/sse`。
`MCP_HOST`、`MCP_PORT`、`MCP_PATH`、`MCP_SSE_PATH`、`MCP_DEBUG` 可覆寫設定。
無效 transport 會退回 stdio，警告寫入 stderr，確保 stdout 僅包含協定訊息。

此入口提供配置於伺服器端的帳號能力，適合由可信任操作員透過 stdio 或本機 HTTP 使用。
HTTP 入口沒有另設使用者登入；若需遠端存取，須先配置有身分驗證的代理並限制連線來源。

## API 與授權設定

- `VIOLATION_RECORD_API_URL`：違規 API，預設 `http://127.0.0.1:8002`。
- `FCM_API_URL`：FCM API，預設 `http://127.0.0.1:8003`。
- `WORKER_OIDC_CLIENT_ID`、`WORKER_OIDC_CLIENT_SECRET` 與
  `WORKER_OIDC_TOKEN_ENDPOINT`（或 `OIDC_ISSUER_URL`）：服務帳號的
  client credentials。必須依現有部署流程配置帳號、租戶及工地權限；
  只有 Keycloak client 設定不足以取得業務 API 授權。
- `MCP_STATIC_BEARER`：可選，僅供違規查詢工具使用現有 OIDC access token。
  上傳與 FCM 仍使用服務帳號；靜態 token 到期後需自行更新。
- `MODEL_FETCH_API_URL` 與 `MODEL_FETCH_BEARER_TOKEN`：模型下載端點及 Bearer token，
  沿用 `src/model_fetcher.py` 的下載設定。
- `MCP_UPSTREAM_TIMEOUT_SECONDS`、`MCP_REMOTE_IMAGE_TIMEOUT_SECONDS`：預設 20 秒。
- `MCP_MAX_REMOTE_IMAGE_BYTES`：推論輸入及上傳圖片上限，預設 10 MiB。

不提供本機帳號密碼登入或免驗證查詢。查詢使用連線池；上傳與 FCM 沿用既有 client。
服務關閉時會釋放模型與 HTTP 連線，即使 transport 的 task group 已取消。

## 工具

| 功能 | 工具 |
| --- | --- |
| 本機推論 | `inference_detect_frame` |
| 危害分析 | `hazard_detect_violations` |
| 違規查詢 | `violations_search`、`violations_get`、`violations_get_image`、`violations_my_sites` |
| 通知 | `notify_fcm_send`、`notify_line_push`、`notify_messenger_send`、`notify_telegram_send`、`notify_wechat_send`、`notify_broadcast_send` |
| 違規上傳 | `record_send_violation`、`record_batch_send_violations` |
| 串流截圖 | `streaming_capture_frame` |
| 模型 | `model_sync`、`model_list_available`、`model_get_local` |
| 幾何及格式驗證 | `utils_calculate_polygon_area`、`utils_point_in_polygon`、`utils_bbox_intersection`、`utils_validate_detections` |

`inference_detect_frame` 接收 `image_base64`，可設定 `model_key`、`use_ultralytics`、
`movement_thr`。同一設定重用模型；設定改變時先關閉舊模型再初始化，
並序列化共用 tracker 的操作。

`hazard_detect_violations` 接收至少六欄的偵測列 `[x1,y1,x2,y2,confidence,class_id]`，
或包含 `bbox`／`box`、`confidence`／`conf`、`class_`／`cls` 的物件。
`detection_items` 可逐次指定檢查開關；無效數值會回報錯誤。

上傳須提供 `image_base64`、`detections`、`warnings`、`site`、`stream_name`、
`model_id`、`model_version`；`timestamp` 為可選 ISO 時間。
`detections` 沿用 API 的至少七欄追蹤列：
`[x1,y1,x2,y2,confidence,class_id,track_id]`；`warnings` 為危害分析回傳的結構化物件。
`cone_polygon`、`pole_polygon` 可提供區域座標。
模型 ID／版本必須存在於目前帳號有權使用的 registry。
批次工具接收 1–100 個相同欄位的物件，依序處理並保留個別失敗結果。
不再使用示範成功回應、預設工地／攝影機或假的上傳統計。

`streaming_capture_frame` 的 `frame_format` 支援 `base64`（JPEG）及 `array`，
開啟／讀取 timeout 各 10 秒，I/O 於工作執行緒完成並確保釋放 capture。
常駐偵測由現有 `main.py` worker 流程管理；移除原先只回傳「未支援」的串流控制工具。
`violations_get_image` 的網址仍需 API Bearer 授權；可用 `as_base64=true` 直接讀取圖片。

多平台通知安裝 `uv sync --extra mcp --extra social-notifications`。
平台 SDK 與影像依賴於呼叫相關工具時才載入；設定請參閱 `.env.example`。
FCM 使用 OIDC 服務帳號，其他平台使用各自 token／secret。
未配置的平台會在呼叫时回報錯誤，不影響 MCP 啟動。
訓練、評估及資料增強工具保留於各自的 `examples/YOLO_*` 目錄。
新增／修改工具後可執行 `uv run --no-sync pytest tests/examples/mcp_server` 驗證。
