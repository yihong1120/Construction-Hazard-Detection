🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# 獨立 Rust 攝影機流程

`construction-pipeline` 可執行本機影片、RTSP 與 HTTP(S) 來源的擷取、TensorRT 推論、追蹤、危險區域分析、標註及輸出。異常事件經由 PostgreSQL outbox 保存證據，並透過既有 HTTP 介面上傳與派送。Rust 程序使用 FFmpeg、TensorRT、CUDA 與 OpenCV 的原生介面，執行時無須安裝 Python、PyTorch 或 Ultralytics。管理 API 和模型訓練可獨立運作。

```mermaid
flowchart LR
    C[JSON / PostgreSQL 設定] --> S[Rust 監督與 generation 管理]
    S --> V[程序內 libav 解碼<br/>每來源單一 owner]
    V --> B[有硬上限的影格池<br/>保留 YUV 或重用 BGR 緩衝區]
    B --> L[Latest mailbox]
    L --> I[每模型共享 worker<br/>取得 GPU 配額後才選最新影格]
    I --> P[共用前處理池<br/>YUV 融合或 OpenCV INTER_LINEAR]
    P --> G[TensorRT / CUDA<br/>同步或兩槽非同步]
    G --> T[有界 CPU executor<br/>追蹤與危險分析]
    T --> M[純 metadata snapshot<br/>含 generation / 時效]
    L --> R[最新影格<br/>Rust 多語標註]
    M --> R
    R --> E[程序內 libav 編碼 / mux<br/>NVENC 或 libx264]
    L -->|"未標註輸出可直接縮放 YUV"| E
    E --> O[RTSP / MP4]
    T --> D[Redis 警報與模型狀態]
    T --> J[Rust JPEG 原圖證據]
    J --> Q[PostgreSQL 持久化 outbox]
    Q --> U[證據上傳工作者及併發上限]
    U -->|儲存上傳回執| Q
    Q -->|已有回執的事件| N[獨立通知工作者與重試]
```

## 從原始碼建置

在 repository 根目錄執行下列指令。建議環境為 Linux、Rust 1.93.1、C 編譯器與 FFmpeg 開發套件。此範例使用 Ubuntu 套件及 rustup。

```bash
sudo apt-get install build-essential pkg-config ffmpeg \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
  fonts-noto-core fonts-noto-cjk
rustup toolchain install 1.93.1 --profile minimal --component rustfmt
cargo +1.93.1 build --release --locked --manifest-path native/pipeline/Cargo.toml
```

FFmpeg 的開發與執行時函式庫須使用相同 major version。非系統 SDK 可用 `FFMPEG_ROOT` 指定安裝根目錄。Noto 字型供多語標註使用。

### 先以 CPU 驗證規則

下例使用 JSONL 輸入執行危險分析，產生含追蹤資料、警報和區域多邊形的 JSONL。此指令只讀寫本機檔案。

```bash
work_dir="$(mktemp -d)"
printf '%s\n' '{"tracked":[[0,0,10,20,0.9,2,0,0]],"reset":true,"rules":{"detect_no_safety_vest_or_helmet":true}}' \
  > "$work_dir/input.jsonl"
native/pipeline/target/release/construction-pipeline analyze \
  --input "$work_dir/input.jsonl" --output "$work_dir/result.jsonl"
cat "$work_dir/result.jsonl"
```

CPU 建置也可重播預錄偵測資料，為獨立串流流程提供測試輸入。重播以 `experimental_replay` 選用，若啟用派送僅接受 loopback 端點，Redis 則須指定隔離前綴。

### 啟用 TensorRT 推論

真實推論使用 `tensorrt` feature。準備相容的 TensorRT、CUDA、OpenCV core／imgproc SDK 與正確的 runtime 函式庫。下列 SDK 路徑請依安裝位置設定。

```bash
export TRT_INCLUDE=/opt/TensorRT/include
export TRT_LIB=/opt/TensorRT/lib
export CUDA_INCLUDE=/usr/local/cuda/include
export CUDA_LIB=/usr/local/cuda/lib64
export OPENCV_ROOT=/
export LD_LIBRARY_PATH="$TRT_LIB:$CUDA_LIB:${LD_LIBRARY_PATH:-}"
cargo +1.93.1 build --release --locked --features tensorrt \
  --manifest-path native/pipeline/Cargo.toml
```

準備能由選定 TensorRT runtime 與 GPU 載入的 `.engine`，將路徑設為 `YOLO_TRT_ENGINE`。`init` 以輸入影片偵測來源尺寸、計算 engine SHA-256，並建立一份有限格數、只產生本機檔案的設定。`run` 與 `outbox` 需要可連線的 PostgreSQL。設定 `DATABASE_URL` 前請參閱 [資料庫連線與遷移](POSTGRES_OUTBOX.md)。

```bash
run_dir="$(mktemp -d)"
pipeline=native/pipeline/target/release/construction-pipeline
"$pipeline" init --config "$run_dir/config.json" --engine "$YOLO_TRT_ENGINE" \
  --source tests/videos/test.mp4 --state-dir "$run_dir/state" \
  --version local-experiment --max-frames 120
"$pipeline" validate --config "$run_dir/config.json"
"$pipeline" run --config "$run_dir/config.json"
"$pipeline" outbox --config "$run_dir/config.json"
```

範例產生 clean、英文及繁中 MP4 和 `summary.json`。派送預設關閉，工作時段內的事件仍可寫入本機 outbox。`validate` 檢查 JSON 結構，模型載入時再檢查 engine 的 SHA-256。`init` 會保護既有設定檔，須使用新檔名。

## CPU 工作者設定

數字 `cpu_workers` 預設採固定模式，省略時依可用 CPU 並行量設定，最多 8 個。加入 `cpu_workers_auto` 後，`cpu_workers` 成為自動模式的上限。以下範例將每兩路配置為一個共享工作額度，上限為 16。

```json
{
  "cpu_workers": 16,
  "cpu_workers_auto": {
    "cameras_per_worker": 2,
    "min_workers": 1,
    "reserve_cpus": 4
  }
}
```

Rust 以已載入且啟用、未過期的鏡頭數除以 `cameras_per_worker`，無條件進位，再套用上下限。實際上限是 `cpu_workers` 與「可用 CPU 並行量減去 `reserve_cpus`」中較小者，至少保留一個工作額度。RTSP 暫時重連期間仍計入已啟用鏡頭。省略 `cpu_workers_auto` 即使用固定模式。

模式和上限在啟動時讀取，工作池內的空閒執行緒會休眠。鏡頭啟停、過期或 PostgreSQL catalog 刷新後，原工作池會調整可同時執行和排入的工作數。縮減時，已執行與已排隊的工作會完成。`summary.json` 的 `cpu_workers` 記錄初始額度、目前額度、上限與變更次數。啟動與額度變更都會寫入日誌。

共享工作池負責追蹤、區域分析、標註及證據 JPEG，排隊與執行中的工作合計受並行額度兩倍限制。串流解碼、編碼及模型前處理各有自己的資源設定。量測時可並列 `render_skipped`、`encoded_unique` 和 `encoded_repeated`，同時記錄輸入解析度、幀率、觀看需求及 CPU 用量。

## 排程與可靠性

- 每模型一個 Rust 推論執行緒與 TensorRT context，由所有攝影機共享。批次等待上限 2 ms、預設每模型請求佇列 32；超過 engine profile 的 batch 不會強行執行。
- 即時標註預覽有框線或警報時直接在輸出尺寸合成。同尺寸的多語輸出共用一張縮放後的底圖；空白標註直接交給編碼器。直播編碼器重用同一影格已轉換的像素，維持輸出幀率與新的時間戳。
- 標註、警報摘要及串流狀態支援 `zh-TW`、`zh-CN`、`en`、`ja`、`vi`、`id`、`fr` 和 `th`。已知類別依播放語系翻譯，自訂類別可用 `Model.labels` 指定各語系名稱。按需播放會將 `zh-Hans`、`fr-CA`、`th_TH` 等別名轉成對應的串流語系，並只啟動被請求的版本。CJK 及泰文可使用系統 Noto 字型，也可在 `fonts` 設定指定字型。
- 即時 HTTP(S) 來源遇到讀取逾時或 EOF 時會獨立重連。HTTP(S) 檔案仍依媒體時間戳播放，逐格離線處理仍在 EOF 結束。
- `lossless: false` 為即時模式，採最新影格，主動跳格以限制延遲；`lossless: true` 使用有界佇列，適合離線逐格驗證。`fps` 設定輸出時脈，目前未另外設定每鏡頭的推論 FPS 預算。`summary.json` 分別紀錄 captured、processed、skipped、stale、encoded 與 failures。
- 影格由 `Arc` 持有；所有讀取者釋放後，buffer 才回到影格池。擷取重連會更新 generation，拒絕舊推論結果並重設追蹤。
- RTSP 使用 TCP，讀取逾時／停滯後退避重連。前進中的 PTS 允許正常靜態畫面；缺少 PTS 時以低成本影像差異檢查凍結。即時輸出超過 5 秒沒有新影格會顯示斷線畫面。
- GPU 推論、模型記憶體與 NVENC 的 flock 預算可跨程序協調；只有共用 lease directory、namespace 與 capacity 的程序會互相協調。記憶體額度是設定的保留量，不是即時 VRAM 計量。
- 解碼與編碼各有獨立 Rust owner 執行緒；長期 I/O 不占用 Tokio blocking pool。沒有 FFmpeg 子程序或 rawvideo pipe。NVENC lease 持有至原生 encoder 釋放；RTSP 失敗可重連並回退 CPU，檔案輸出失敗直接回報，避免重開時覆蓋已錄製內容。來源、輸出與憑證不寫入 libav 原始錯誤日誌。
- JSON 設定可連接 PostgreSQL catalog，驗證已發布權重／engine manifest 的版本與 SHA 關係；LISTEN 加輪詢更新設定，先停止舊工作再切換模型。更新失敗保留上一份已驗證設定。
- 工作時段採用每路的 `work_start_hour` 和 `work_end_hour`，依影格當地時間判定。時段包含開始小時，於結束小時截止，`0` 至 `24` 代表全天。JSON 的 `streams[].timezone` 預設為 `Asia/Taipei`，PostgreSQL catalog 的鏡頭也使用台北時間。工作時段外只有人員進入交通錐或電線桿管制區會產生證據和通知。即時辨識與標註全天運作。
- Redis 警報使用既有 warning stream 格式；另提供 Rust latest metadata。站點 generation 與攝影機 owner token 阻擋過期寫入，模型狀態可回應 publication challenge。
- 原始 JPEG 與事件在 PostgreSQL 同一交易提交，再由 HTTP dispatcher 派送。上傳 receipt 先持久化才通知；通知重試／程序重新啟動不需再次上傳已成功的證據。
- Outbox 預設限制尚保留的 JPEG 與 metadata 總量為 512 MiB，每個派送階段最多嘗試 20 次。永久錯誤進入失敗佇列。這是 payload 額度，PostgreSQL 的表、索引與 WAL 仍需額外空間。服務端需支援 `Idempotency-Key`，避免遠端操作成功而本機尚未記錄時造成重複操作。
- `record_detections` 預設關閉；啟用後每鏡頭診斷 JSONL 每 10 MiB 輪替，保留目前與上一份。SIGINT／SIGTERM 會停止擷取、join 原生 media owner、完成 encoder trailer、關閉模型並寫出統計。

`delivery` 使用現有 `/upload` multipart、通知 JSON 與 OAuth client-credentials 協定，憑證僅由設定指定的環境變數讀取。

`delivery.evidence_workers` 與 `delivery.notification_workers` 各預設同時處理兩個請求。推播延遲時，證據仍可持續上傳。通知重試只取用已儲存的回執及 metadata，無須從 PostgreSQL 讀取 JPEG。端點提供重試間隔時，派送程式至少等待該時間，最長以一天計。

預錄偵測結果必須明確啟用 `experimental_replay`；重播派送限 loopback，Redis 必須使用隔離 prefix，避免測試資料流入正式服務。
重排失敗事件可使用 `outbox --config ... --requeue-failed`。

## 擷取與處理時脈

- `capture.buffer_frames` 是每來源 BGR buffer 的硬上限（即時預設 6），不是快取數；滿載時繼續解碼／排空來源但不做 BGR 轉換。離線 lossless 模式等待空間，保留全部影格。buffer 尾端保留 codec SIMD 需要的 64 bytes padding，最後一個 `Arc` 釋放才回收。
- 推論請求只排入攝影機 mailbox，不把排隊當下的舊影格固定在佇列中。取得 GPU admission 後，worker 才取最新影格，並將實際影格與結果一起交還追蹤；每鏡頭最多一個待完成請求。
- 即時標註會用最新影格配上最新分析。保留 YUV 的影格到達時準備標註，編碼器依各輸出的 FPS 取用最新畫面。中繼資料只持有追蹤和幾何資訊。分析結果超過 `overlay_max_age_ms`（預設 1000）或 generation 不符時顯示 `Analysis pending`。Lossless 模式的影格與分析逐格對應。
- `capture.max_lag_ms` 預設 2000：相對第一個解碼 PTS 的落後量持續增加時重連，避免解碼追不上卻一直累積舊資料；不是攝影機到瀏覽器延遲量測。PTS 明顯倒退也會切換 generation。`read_timeout_ms` 限制可中斷的原生 I/O。
- `summary.json` 增加 `converted`、`capture_dropped`、`capture_reconnects`、`capture_pool_peak`、`rendered`、`render_skipped`、`render_pending`。trace 增加實際 `source_pts`、`source_lag_ms` 與 `batch_size`，能區分推論跳格、轉換 admission 及標註降載。

`native/media` 只有窄版 C ABI adapter，Rust 負責生命週期、取消、排程與背壓。預設 CPU 解碼／BGR 前處理仍會搬移影像；可選 NVDEC／GPU NV12 路徑見下方。帶有旋轉／鏡射 display matrix 的檔案來源目前明確拒絕，避免默默輸出不同方向的偵測；一般未旋轉 RTSP／影片可使用。
SDK 採用 FFmpeg 的 [send／receive 解碼介面](https://ffmpeg.org/doxygen/8.0/demux_decode_8c-example.html) 與 [編碼／封裝介面](https://ffmpeg.org/doxygen/8.0/mux_8c-example.html)。

## 供幀與前處理增量優化

獨立 Rust 的前處理 pool 由所有模型共用，不會為每個模型另開一組執行緒。
`preprocess_workers` 預設 1，控制模型前處理的執行緒數。
它與 `cpu_workers`（追蹤／幾何／標註／JPEG）分開計算，並非整個程序 CPU 使用率上限。
只有多張 batch 能使用跨影像並行；單張 batch 也能使用下述精確 CPU 取樣。

```json
{
  "preprocess_workers": 8,
  "exact_cpu_resize": true,
  "gpu": {
    "asynchronous": false,
    "pipeline_depth": 1
  }
}
```

將此片段合併至既有設定。batch 和輸入尺寸須符合 engine profile。較大的前處理工作池可用於多張批次，建議同時觀察整體 CPU 成本。

`exact_cpu_resize` 預設 `true`。等比例、奇數整數縮小及原尺寸影像可直接以 Rust 取樣，
letterbox padding、BGR 通道與 OpenCV INTER_LINEAR 的輸出逐 byte 相同。
其他比例、放大或混合 batch 中不符合條件的影像個別使用原 OpenCV 路徑；
設為 `false` 可回到完整 OpenCV 參考路徑。此設定只影響獨立 Rust pipeline，Python 綁定維持原前處理。

`gpu.exact_resize` 預設 `false`，是需要 `gpu.asynchronous: true` 的實驗選項。
它將完整 CPU BGR 影像上傳後，在 CUDA 合併縮放與 RGB CHW 正規化；
整批必須符合上述精確比例，否則回到 CPU 前處理。
此路徑仍傳輸完整影像，並使用 pinned buffer 和 GPU buffer。

`gpu.asynchronous: true` 可使用 pinned buffer／CUDA stream；搭配 `pipeline_depth: 2` 才會讓下一批 CPU 前處理與目前 GPU 工作交錯。
單一 TensorRT context 的 shape 改變前，會先排空在途 batch；空佇列不等待湊第二批，取消／關閉也會排空。
兩槽模式目前不能同時啟用全域 `inference_slots`；保留 depth 1 可沿用全域 admission。

啟用 `record_detections` 後，每筆 trace 新增 `inference_timing`：
`batch_id`、`queue_ms`、`admission_ms`、`prepare_ms`、`submit_ms`、`gpu_wait_decode_ms`、`batch_span_ms`、`in_flight`，以及 `analysis_queue_ms`／`analysis_cpu_ms`。
另記 `gpu_preprocess` 與 `cpu_exact_images`；後者為整批採用 Rust 精確取樣的影像數，不是該筆影格的個別計數。
同一 batch 的批次時間會出現在多個影格紀錄中，統計時需依模型與 batch ID 去重；模型 worker 重建後 batch ID 會重設。
這些是 host wall time，不是 GPU kernel event 計時。同步 `submit_ms` 包含執行與等待，非同步則量到提交；`gpu_wait_decode_ms` 包含 collect 等待與結果解碼。

`summary.json` 新增 `capture_read_ms`（含來源等待）、`capture_convert_ms`、分析 queue／工作累計、`encode_ms`。
輸出另記 `encoded_unique`、`encoded_repeated`、`encoded_offline`；unique 依來源 generation／sequence 計數，不表示像素或畫面內容一定變化。
這些累計包含暖機與啟動，不能直接當成中間量測窗口的速率。

`capture.max_convert_fps` 預設 `0`（不限速）。若用途只需每路 15 FPS，可明確設為 `15`：
解碼與 PTS／連線存活檢查仍繼續，但超出預算的影格在 BGR 轉換前略過，避免轉換後才被 latest mailbox 覆蓋。
此為每路的轉換上限，不是總 GPU 配額；也限制 clean／annotated 能取得的新影格速率。
輸出設定 `fps` 不會因此改變，高於轉換上限時可能重複影格。需要逐格推論／更高更新率時保持 `0` 或提高上限。
`lossless: true` 忽略此設定，保留每格；`summary.json` 的 `capture_rate_dropped` 另計主動略過的解碼格，
與 `capture_dropped`（buffer 耗盡）分開。限速採 monotonic 固定相位與早到容差，斷線後不補送舊時槽。

## CPU YUV 與按需輸出

`capture.host_yuv: true` 讓 CPU 解碼保留引用計數的 YUV 影格。即時串流、縮圖與錄影直接縮放到編碼器的 limited-range YUV420P。
後端將框線、遮罩及字形交給編碼器，依輸出色彩矩陣與 4:2:0 色度覆蓋率繪製。告警 JPEG 由保留影格編碼，凍結檢查直接取樣解碼平面。
`summary.json` 的 `encoder_host_frames` 記錄 YUV 影格提交次數，`encoder_annotated_host_frames` 記錄其中有 YUV 標註的影格，重複提交由其他計數器記錄。

精確前處理支援與設定解析度相符、寬高皆為偶數的 8-bit planar 4:2:0、4:2:2 和 4:4:4。
校準查表可供原生尺寸和奇數整倍縮小直接取樣。其他受支援尺寸重用掃描列緩衝區，產生與 OpenCV 一致的 letterbox 像素，包含 720p 縮至 640 × 360 的兩倍縮小。不支援的格式仍走全幅 BGR 參考路徑。

後端播放可用兩個預設關閉的 `output` 選項。

- `clean_passthrough`：由同一輸入連線轉送壓縮封包，不增加攝影機讀取連線。限 live RTSP／libav ingress，來源大小與 FPS 不符時自動回編碼。
- `on_demand`：使用既有 Redis detail／語系觀看租約啟停 live 輸出。同一 pipeline 程序每 250 ms 集中查詢已註冊鏡頭的租約，所有標註輸出都無訂閱時停止標註排程。推論與事件處理持續執行。

瀏覽器基準測試的獨立 JSON 鏡頭可設定 `output.client_overlay`，用以測試中繼資料繪製。

這條 CPU 路徑採逐張回退：混合批次中，不支援融合的影格才建立 BGR。
色彩查表逐一驗證全部 256³ 組合後，可由 48 MiB 縮成每個 profile 16.125 MiB；
如果校準結果不能分離，仍保留完整查表。建立時暫時需要完整表，不代表啟動峰值也減少同樣大小。
融合取樣只填 letterbox 邊界，不再先填整張再覆寫內容。

兩個額外選項預設關閉，可供獨立比對：

```json
{
  "metadata_async": true,
  "capture": { "host_yuv": true },
  "gpu": { "asynchronous": true, "direct_pinned": true, "pipeline_depth": 1 }
}
```

`gpu.direct_pinned` 讓融合與逐張回退直接填入已保留的 pinned slot，省去模型大小 buffer 的一次 CPU 複製，
仍保留 H2D 上傳。需要非同步模式；錯誤與 panic 會釋放未提交的 reservation，collect 前不重用已提交 slot。
`metadata_async` 使用每鏡頭一格的 latest-only mailbox；只持有追蹤／幾何資料，不持有影像。
Redis 慢速時合併視覺狀態，警報 JPEG／事件仍經原有持久化 outbox，停止時完成最後一格發送與 owner 清理。
`summary.json` 提供 `metadata_offered`／`metadata_sent`／`metadata_superseded`／`metadata_errors`。

交通錐中心與電線桿警戒圓的幾何輸入相同時，分析會重用多邊形。標註、metadata 與事件證據共用座標。`summary.json` 的 `geometry_cache_hits` 與 `geometry_cache_misses` 分別記錄啟用的幾何檢查命中與未命中的次數。

live 輸出等到觀看需求與首格資料齊備才建立 native owner；租約失效時結束執行緒與 muxer，續租後等 keyframe 重建。
字型系統延後到首次後端繪字才初始化，`fonts_initialized` 回報程序共用的字型狀態。

`host_yuv` 與 `nvdec_device` 為互斥擷取模式。

PostgreSQL 目錄管理的鏡頭由後端依語系產生標註預覽與原尺寸串流。CPU 解碼時設定 `capture.host_yuv: true`，即可直接在編碼器的 YUV 平面繪圖。設定預覽解析度並搭配 Redis 啟用 `database.media.on_demand`，讓後端只為有觀看需求的串流與語系編碼。

Flutter 透過播放 API 選擇 `overlay` 與觀看語言。Rust 播放牆可包含最多 32 台鏡頭，Flutter 直接播放回傳的 HLS 並續租。

### 20 FPS 後端標註預覽

將下列片段合併到已設定 Redis 觀看租約的 PostgreSQL 鏡頭設定。這組設定使用 CPU YUV 擷取，將 640 × 360 預覽編碼設為 21 FPS，為每秒 20 張新影格的目標留一格餘裕。

```json
{
  "cpu_workers": 16,
  "metadata_async": true,
  "capture": { "host_yuv": true, "max_convert_fps": 0 },
  "database": {
    "media": {
      "on_demand": true,
      "preview": { "width": 640, "height": 360, "fps": 21 }
    }
  }
}
```

目標是每秒 20 張新影格時，可先用能穩定提供 24、25 或 30 FPS 的來源，留一些餘裕應付來源抖動。獨立鏡頭設定可使用 `streams[].output.preview.fps`。只開預覽時，用 `encoded_unique` 除以該鏡頭運作秒數計算有效 FPS，搭配 `encoded_repeated`、`capture_rate_dropped` 與 `render_skipped` 分辨來源、擷取限速或標註排程的影響。

## Rust 原生 RTSP 設定

```json
{
  "capture": { "rust_rtsp": true, "max_convert_fps": 15 }
}
```

`capture.rust_rtsp` 預設 `false`。開啟後，`rtsp://` 的 TCP RTSP／RTP 接收、H.264／H.265 重組與時間戳由 Rust `retina` 處理；CPU 解碼仍使用程序內 libavcodec，不需要 NVDEC、OpenCV 擷取或 FFmpeg 子程序。
這個開關只更換接收器，不會自動消除解碼、BGR 轉換及模型前處理。`rtsps://` 與本機檔案沿用 libavformat；Rust 接收器遇到不支援的 codec 會回報並由既有重連流程處理。

每來源最多排隊兩個完整壓縮影格，滿載時背壓，不任意丟棄參考影格。初次連線與 RTP 遺失後等待隨機存取格；遺失後重建 decoder 並更新 generation，避免沿用舊追蹤結果。參數改變時重連。握手與影片停滯有期限，停止會取消接收任務並關閉 TCP；不在背景留下 TEARDOWN 重試。

`summary.json` 增加 `rust_rtsp_frames`、`rust_rtsp_bytes`、`rust_rtsp_lost_packets`、`rust_rtsp_keyframe_dropped`、`rust_rtsp_handoff_ms`、`rust_rtsp_decoder_resets`。
frames／bytes 計完整 access unit，不含網路協定標頭；handoff 是整次執行的 queue 交接等待累計，不能當作單格或端到端延遲。
此選項建議先以實際型號和預期連線條件驗證長時間擷取與重連。

## 可選 NVDEC／NV12 擷取

下列片段合併至完整設定；`tensorrt` build 現在也會啟用 `construction-media/cuda`，
需要 CUDA headers／driver library，以及具 NVDEC 功能的 FFmpeg。CPU-only build 維持原依賴。

```json
{
  "capture": { "nvdec_device": 0, "max_convert_fps": 15 },
  "gpu": { "device": 0, "asynchronous": true, "pipeline_depth": 1 }
}
```

預設不啟用 NVDEC；須與推論 device 相同，不接受 replay 模型。
支援的 8-bit NV12、原生尺寸影格直接留在 GPU；CUDA 融合取樣、letterbox 與 RGB CHW 正規化，
使用啟動時從原 FFmpeg YUV420P 轉換器校準的共用色彩表，每個 device／profile 為 48 MiB。
符合寬度 32 倍數、奇數整數等比例縮小／原尺寸、BT.601／709 條件才走直接路徑；其他 batch 下載並使用 CPU 前處理。
此模式的 `max_convert_fps` 限制保留供推論／觀看的候選影格，NVDEC 仍持續解碼；設 `0` 可測完整來源供應率。
標註、事件 JPEG 與現有編碼需要像素時才下載並快取 BGR；輸出仍是原 CPU 處理，並非全流程零拷貝。
無 PTS 時保留每秒最多一次的凍結取樣，有 PTS 時直接檢查其進度。
summary 另記 `gpu_admitted`、`gpu_materialized`、`gpu_download_payload_bytes`，trace 另記 `nv12_preprocess`。

## 驗證

在 repository 根目錄執行。單元測試可直接執行，完整檔案重播及 outbox 測試另需 PostgreSQL。

```bash
cargo +1.93.1 fmt --manifest-path native/pipeline/Cargo.toml -- --check
cargo +1.93.1 test --locked --manifest-path native/pipeline/Cargo.toml --lib
node --test native/pipeline/web/overlay.test.mjs
```

隔離 PostgreSQL 範例使用 Docker，採用與 CI 一致的資料庫名稱。`CHD_RUST_TEST_DATABASE_URL` 只接受 loopback 的 `chd_rust_pipeline_test`。

```bash
(
  set -eu
  test_db="$(docker run --detach --rm \
    -p 127.0.0.1:25433:5432 \
    -e POSTGRES_USER=fixture -e POSTGRES_PASSWORD=fixture \
    -e POSTGRES_DB=chd_rust_pipeline_test postgres:17.11-bookworm)"
  trap 'docker stop "$test_db" >/dev/null' EXIT
  for attempt in $(seq 1 30); do
    if docker exec "$test_db" pg_isready -h 127.0.0.1 -U fixture -d chd_rust_pipeline_test >/dev/null; then
      break
    fi
    sleep 1
  done
  docker exec "$test_db" pg_isready -h 127.0.0.1 -U fixture -d chd_rust_pipeline_test
  export CHD_RUST_TEST_DATABASE_URL=postgresql://fixture:fixture@127.0.0.1:25433/chd_rust_pipeline_test
  cargo +1.93.1 test --locked --manifest-path native/pipeline/Cargo.toml
)
```

[CI 工作流程](../../.github/workflows/rust-pipeline.yml) 另外啟動隔離 Redis 與 MediaMTX，測試 32 個 RTSP session、語系觀看租約、catalog 更新及過期狀態。32 路樣本共用本機影片來源並重播偵測資料，測得的是隔離環境的流程正確性。若要評估可支援的實際攝影機數，須另以目標模型、來源及觀看需求量測。

如需與 Python 規則或實際 CUDA 推論比對，可在已安裝本專案 Python 相依套件的環境下，使用 [規則比對腳本](../../scripts/validate_rust_pipeline.py) 與 [CUDA 驗證腳本](../../scripts/validate_standalone_cuda.py)。以 `--help` 查看 binary、engine 及輸出目錄參數。
