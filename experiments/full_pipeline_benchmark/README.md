# 攝影機完整流程比較

在同一台主機依序執行三種攝影機處理流程。三組共用 YOLO26n INT8 engine、640 輸入尺寸、batch 上限 8、2 ms batch 等待、RTSP 來源和警報規則。

| 方案 | 流程 |
| --- | --- |
| `python` | `main.py`，Python TensorRT worker 與 Python 後處理 |
| `hybrid` | `main.py`，Rust TensorRT worker 與 Rust 後處理 |
| `rust` | 獨立 `construction-pipeline`，單一程序管理擷取、推論、分析、輸出和 outbox |

Python 兩組由 `legacy.py` 加入危險判定 trace，並將 OpenCV 和解碼執行緒各限制為 1。Rust 使用 `record_detections` 記錄相同處理邊界。測試輸出採 libx264、ultrafast、CRF 23、單執行緒及兩秒 GOP，預設 15 FPS。Python 的 clean 和 annotated 輸出共用擷取影格，FFmpeg 子程序由測試 wrapper 套用輸出參數。獨立 Rust 在程序內呼叫 libav。

## 建置與依賴

以下指令都從 repository 根目錄執行。主機需要 NVIDIA GPU、能載入實驗 engine 的 TensorRT／CUDA 環境、Git、Docker、FFmpeg 和專案 Python 3.15。FFmpeg wrapper 使用 `/usr/bin/ffmpeg`。先安裝專案與量測工具，再於同一 Python 環境準備 TensorRT bindings。

```bash
uv sync --locked --extra streaming --extra yolo --extra yolo-gpu
source .venv/bin/activate
uv pip install psutil nvidia-ml-py matplotlib
```

- Python TensorRT bindings 的建置與 SDK 要求見 [專案安裝說明](../../README.md#1-prepare-python)。bindings 和 Rust binary 各自使用的 TensorRT 版本均須能載入同一個 engine。
- `hybrid` 需在上述 Python 環境建置並安裝 [Rust 推論 wheel](../../native/inference/README.md#tensorrt-版本與建置) 與 [Rust 後處理 wheel](../../bindings/python/postprocess/README.md#建置與切換)。推論指南也說明 `LD_LIBRARY_PATH` 和 `YOLO_WORKER_NATIVE_LIBRARY_DIRS` 的設定。
- 獨立 binary 依 [TensorRT 建置方式](../../native/pipeline/README-zh-tw.md#啟用-tensorrt-推論) 啟用 `tensorrt` feature。
- 將相符的 YOLO26n checkpoint 放在 `models/pt/best_yolo26n.pt`，另備已匯出的 INT8 `.engine`。測試會核對 checkpoint 和 engine 的 SHA-256。engine 預設從 `models/int8_engine/best_yolo26n.engine` 讀取，也可用 `--engine` 指定路徑。這兩個模型檔需另外準備。

## 啟動隔離服務

PostgreSQL、Redis、MediaMTX 和 HTTP mock 均綁定 loopback。帳密、token 和上傳資料是測試用 fixture。使用這組獨立容器及資料庫執行測試。

```bash
docker run -d --rm --name chd-e2e-pg \
  -e POSTGRES_USER=fixture -e POSTGRES_PASSWORD=fixture \
  -e POSTGRES_DB=chd_e2e_benchmark -p 127.0.0.1:25433:5432 postgres:17.11-bookworm
docker run -d --rm --name chd-e2e-redis \
  -p 127.0.0.1:26380:6379 redis:7.4.10-alpine3.21
docker run -d --rm --name chd-e2e-rtsp \
  -v "$PWD/experiments/full_pipeline_benchmark/mediamtx.yml:/mediamtx.yml:ro" \
  -p 127.0.0.1:28554:8554 -p 127.0.0.1:29997:9997 \
  bluenviron/mediamtx@sha256:f8628851106cc053f9175b248050bb5f362a6e65abd72297c167a1cb5a9a3be2
```

確認 PostgreSQL 可經 TCP 連線、Redis 回傳 `PONG`、MediaMTX API 回傳路徑清單，再讓 mock 保持執行。

```bash
docker exec chd-e2e-pg pg_isready -h 127.0.0.1 -U fixture -d chd_e2e_benchmark
docker exec chd-e2e-redis redis-cli ping
curl --fail http://127.0.0.1:29997/v3/paths/list

mkdir -p outputs/full_pipeline_fixture
python -m experiments.full_pipeline_benchmark.mock \
  --journal outputs/full_pipeline_fixture/delivery.jsonl
```

## 準備與發布 RTSP 來源

在另一個終端將 `tests/videos/test.mp4` 製成固定 30 秒循環影片。原片為 640×352，放大後量測 1920×1080 的解碼、影格搬移和編碼工作量。來源解析度或幀率變更時，請同時更新下方影片及實驗設定。

```bash
ffmpeg -nostdin -n -filter_threads 1 -i tests/videos/test.mp4 \
  -vf scale=1920:1080:flags=bilinear,fps=15 -t 30 -an \
  -c:v libx264 -preset ultrafast -crf 18 -threads 1 \
  -g 15 -keyint_min 15 -sc_threshold 0 -bf 0 -pix_fmt yuv420p \
  outputs/full_pipeline_fixture/source.mp4
ffmpeg -nostdin -re -stream_loop -1 \
  -i outputs/full_pipeline_fixture/source.mp4 \
  -an -c:v copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:28554/source
```

## 量測及生成報告

在第三個終端啟用先前的 Python 環境。每次使用新的 `--output` 目錄，並讓量測依序執行。以下設定比較 32 路輸入與 4 路觀看需求。每路觀看啟用 clean 和 annotated 共兩份輸出。

```bash
export PYTHON_DOTENV_DISABLED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m experiments.full_pipeline_benchmark.benchmark \
  --backends python hybrid rust --cameras 32 --viewers 4 \
  --width 1920 --height 1080 --fps 15 --language en \
  --rounds 3 --warmup 30 --seconds 30 \
  --output outputs/full_pipeline_32

python -m experiments.full_pipeline_benchmark.report \
  --primary outputs/full_pipeline_32
```

每輪會先等待各路完成第一筆偵測 trace，並檢查指定輸出的 ready 狀態和尺寸。三組輪流先執行，再依指定的時間暖機與取樣。可透過 `--cameras 1 --viewers 1` 進行單路比較。

`report` 每個輸入目錄接受一種攝影機數量，以及每輪三種方案的完整 `results.json`。可用 `--single` 或 `--low-resolution` 附加其他完整實驗目錄。`summary.json`、`comparison.png` 和 `comparison.svg` 預設寫入主實驗目錄的 `report/`，可用 `--output` 選擇其他位置。

- 各輪原始資料包含 `settings.json` 所記錄的 engine 與 binary 雜湊，以及各輪的 `result.json`、`samples.json`、程式日誌和影格 trace。報告列出各方案中位數及實際最小、最大值。
- CPU 百分比以一個邏輯核心為 100%，PSS 以比例分攤共享記憶體。程序樹統計包含主程序及其 worker、FFmpeg 子程序，並另行記錄可取得的服務 CPU 資料。全機 GPU 使用率可能含其他工作，適合當成量測背景。
- `frame_age_p95_ms` 從解碼取回影格計至危險判定完成。比較時請同時觀察完成判定數、來源積壓、掉格及輸出 ready 狀態。觀看端到端延遲須另行量測。

執行完畢後在各終端停止來源和 mock，再停止測試容器。

```bash
docker stop chd-e2e-pg chd-e2e-redis chd-e2e-rtsp
```
