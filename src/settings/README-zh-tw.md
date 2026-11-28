🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# 設定載入與所有權

環境變數在設定載入層解析為 `EnvironmentSettings`，業務程式使用設定欄位。
各領域保留獨立模型，未使用的通知 provider 不會要求憑證，也不會載入 ML 套件。

| 領域 | 設定入口 | 載入時機 |
| --- | --- | --- |
| 資料庫與啟動流程 | `examples/auth/settings_fields.py`、`connections.py` | engine 建立／應用啟動 |
| Redis 與外送 API | `connections.py` | client 建立 |
| BFF proxy、cookie | `examples/bff/settings.py` | 程序載入 |
| Session 期限與加密 | `sessions.py` | 程序載入；cipher 初次使用 |
| Worker 認證 | `worker_auth.py` | token provider 第一次認證，刷新沿用同一份設定 |
| 擷取與推論 | `capture.py`、`inference.py` | 串流／detector 建立 |
| Rust／Python GPU worker | `inference.py` 的 `backend`、`native_library_dirs` | worker 建立；模型驗證後選取後端 |
| GPU 跨程序配額 | `gpu_resources.py`、`inference.py` | 共用目錄／實體 GPU namespace；NVENC 預設共用，推論與記憶體需明確啟用 |
| Rust 後處理選擇 | `postprocess.py` | 每個攝影機程序啟動；變更後重啟 |
| OpenCV／Rust 擷取 | `capture.py` 的 `backend`、`decoder`、工具路徑 | 攝影機啟動；Rust FFmpeg 後端需明確選用 |
| 媒體、繪字與 NVENC | `media.py` | gateway／renderer 建立；NVENC 保留額度期間固定上限 |
| 通知 provider | `notifications.py` | provider 建立 |
| MCP | `examples/mcp_server/settings.py` | transport 啟動／tool 建立或第一次使用 |
| 管理政策與診斷 | `policy_settings.py`、`debug_settings.py` | 請求邊界／middleware 建立 |

元件持有自己的設定，不把設定拆成多個參數沿呼叫鏈轉傳。純函式保留工作資料
參數；已有的明確建構參數仍可使用，不必改成額外傳入整包設定。

Python `main.py` 以 `STREAM_CAPTURE_INFERENCE_HISTORY_BY_CAMERA` 設定各路
推論可保留的影格數。鍵是 PostgreSQL 的 `stream_configs.id`，值為 1 到 4，
含最新影格。例如在 `.env` 加入 `STREAM_CAPTURE_INFERENCE_HISTORY_BY_CAMERA='{"64":4}'`，
64 號就能依序辨識集中抵達的影格。100 毫秒期限從擷取迴圈將影格交給 Python
推論緩衝時起算。歷史影格逾時後取最新一張，重連時會清空緩衝。
預設 `{}` 時每路只取最新一張。即時預覽持續取用最新畫面。

GPU 配額使用 Rust file-lock lease，沒有安裝 Rust wheel 時沿用相同協定的 Python
實作。所有參與者必須共用 `GPU_RESOURCE_DIRECTORY`、`GPU_RESOURCE_DEVICE_ID`
及上限；Compose 的攝影機程序／replica 共用 `gpu-resource-locks` volume。
不同 Compose project 或主機程序須另外掛載同一目錄，使用相同 UID。
不得在工作仍執行時刪除 lock 檔或重建目錄。配額變更若與使用中程序不一致會拒絕。

`MEDIA_NVENC_MAX_SESSIONS` 現在是上述共用範圍的 session 上限；FFmpeg 子程序
繼承 lease descriptor，因此控制程序退出後，仍存活的 encoder 繼續佔用名額。
`GPU_RESOURCE_INFERENCE_SLOTS` 與 `GPU_RESOURCE_MEMORY_LIMIT_MB` 預設為 0。
啟用時要求 `YOLO_WORKER_BACKEND=rust`，記憶體另需設定每個模型的
`YOLO_WORKER_GPU_MEMORY_RESERVATION_MB`；每份保留向上取整至 64 MiB，
總上限向下取整。這是宣告預算，沒有追蹤外部 CUDA 程序、NVDEC 或 GPU glyph
快取，不保證不會 OOM。請保留足夠餘裕，API／Python backend 不在其管理範圍。

`YOLO_WORKER_NATIVE_ASYNC=true` 才使用 pinned memory、stream/event 及兩個
buffer。`YOLO_WORKER_NATIVE_MICRO_BATCH_SIZE=0` 沿用 engine 容量；較小值切分
現有 worker batch，最多保留兩份未收取結果，shape 變更前先排空。
單一 worker batch 若沒有切分，該次請求不會獲得兩個 batch 的重疊效益。
設定及量測見 [Rust GPU worker](../../native/inference/README-zh-tw.md)。

## 優先順序與驗證

明確參數 > 環境變數 > 預設值。建構入口使用 `from_overrides()`，統一處理
欄位別名與 `AliasChoices`；`None` 表示未指定，空字串與 `0` 會送入驗證，
不會因真假值判斷而意外退回環境設定。例如 Redis 的明確空密碼可以覆寫環境值。

既有環境變數名稱與有效預設值保持相容。新收斂的 timeout、port、容量及布林值
會在載入時驗證；錯字、非有限數值、超出範圍與無效 MCP transport 都會報錯，
不再靜默改用其他設定。來源、發布及預覽 FPS 的選擇規則維持不變。

服務 URL 驗證 HTTP(S)、主機與連接埠，拒絕內嵌帳密。Worker token endpoint
維持 HTTPS 或 loopback HTTP 限制。`__Host-` cookie 必須啟用 secure。
API 與 Worker 的 overlay readiness TTL 共用同一個設定欄位及既有的最小值 5 秒。

新的 provider 與 transport 憑證使用 `SecretStr`，不出現在模型 repr 或一般
序列化輸出。密碼保留原始空白；BFF 加密金鑰與 MCP static bearer 則維持既有
的首尾空白正規化。Redis URL 會編碼密碼並處理 IPv6。

## 設定變更與測試

程序／元件持有的設定不因修改 `os.environ` 自動變更。變更部署環境後，重建
對應元件或重新啟動服務。Token provider 的刷新沿用原憑證；session cipher
沿用初次使用的金鑰。BFF 金鑰更換需另行安排既有 session 撤銷或金鑰遷移。
測試應在建立元件之前設定環境，避免 mock `os.getenv` 這類實作細節。

工地／租戶政策檔與 gateway 狀態檔仍依既有規則讀取內容。Overlay 語言政策
保留以設定內容為鍵的有界快取；呼叫端在自己的生命週期持有不可變政策。
Social client ID 的舊環境欄位組合與語言快取的讀取仍留在設定層。

`tests/src/configuration_contract_test.py` 驗證設定優先順序、資源建立前拒絕錯誤、
生命週期快照、秘密值與共用 TTL，並禁止業務模組新增直接 `os.getenv()` 讀取。
