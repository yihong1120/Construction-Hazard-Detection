# Python／Rust 對照實驗

本目錄是可重現的效能實驗，基準程式使用本 repo 的 `YoloDetector` 預設 centroid
追蹤器與 `DangerDetector`。Rust 版本保留同樣的匹配順序、門檻、track ID、駕駛排除、
空間索引與警示輸出。實驗會產生逐格紀錄、每輪摘要及輸入雜湊。

## 比較範圍

- 真實輸入：`tests/videos/test.mp4` 中每隔 18 格取樣，共 48 格；另加
  `tests/cv_dataset/images` 的 10 張圖片。YOLO26n、640×640、FP32、batch=1、
  confidence=0.25。16 個預處理張量另用於推論對照。
- 規則：未戴安全帽、未穿背心、人員接近機具／車輛，包含駕駛排除。
- 不包含交通錐／電線桿多邊形、Hungarian tracker、攝影機連線、影像解碼與前處理、
  標註繪圖、編碼、MediaMTX、Redis／PostgreSQL、outbox 與通知。
- 16／64／256 個物件的資料是有固定亂數種子的合成壓力測試，不代表實際攝影機負載。
- 模型推論兩邊使用相同的 ONNX、相同版本 ONNX Runtime 及同一個共享函式庫檔案。
  CPU 與 CUDA 可使用不同的 Python 環境，搭配各自載入的 ORT runtime。

## 計時方法

Rust 使用 `cargo build --release --locked`。每種設定執行 7 輪，Python／Rust
輪流先跑，皆固定在同一個 Intel P-core、單一 ORT 計算執行緒。輸入讀檔、模型載入、
暖機、JSON 讀寫與 tracker 重建不計入處理時間；逐格結果配置、警示建構與回收則計入。

`results.json` 中每一輪的毫秒值為該輪處理總時間／影格數；`median_paired_speedup`
為同輪 Python／Rust 比值的中位數，`paired_speedup_range` 為 7 輪的最小／最大值，
**不是統計信賴區間**。P95 是每輪逐格耗時的 95 百分位，再取各輪中位數。

GPU 原有工作負載保留；每次 CUDA 量測前後都保存 `nvidia-smi` 狀態。這不是獨占 GPU
實驗，幾個百分點的差異應視為可能的負載／時脈波動，不應直接宣稱 Rust 加速。
`peak_rss_kib` 包含初始化、輸入載入及依賴，不是完整伺服器的常駐記憶體比較。

## 重現

使用 Linux、Rust ≥1.92、ONNX Runtime 1.31.0 的 CPU 與 CUDA runtime，以及具有
NumPy、OpenCV、scikit-learn、Shapely 與 ORT Python bindings 的環境。`--core` 指定測試程序可用的邏輯 CPU。
CPU 使用目前的 Python。若 CUDA bindings 安裝於另一個環境，設定 `CHD_BENCH_GPU_PYTHON`；以隔離套件目錄提供 bindings 時，可設定 `CHD_BENCH_GPU_SITE`。

```bash
cargo build --release --locked --manifest-path experiments/rust_benchmark/Cargo.toml

export CHD_BENCH_CPU_ORT=/path/to/cpu/libonnxruntime.so.1.31.0
export CHD_BENCH_GPU_ORT=/path/to/cuda/libonnxruntime.so.1.31.0
export CHD_BENCH_GPU_PYTHON=/path/to/gpu-python/bin/python
export LD_LIBRARY_PATH="/path/to/cuda/lib:/path/to/cudnn/lib:${LD_LIBRARY_PATH:-}"

# 使用具有 NumPy、OpenCV、scikit-learn、Shapely、ORT 的 Python 3.15。
PYTHON_DOTENV_DISABLED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python experiments/rust_benchmark/benchmark.py prepare

PYTHON_DOTENV_DISABLED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python experiments/rust_benchmark/run_experiment.py --core 4 --rounds 7
```

`prepare` 使用本機測試影片與圖片，不會讀取 `.env` 的攝影機或連線正式資料庫。
模型 `models/pt/best_yolo26n.onnx` 不在 Git 中，須先準備；本次模型與原始碼的 SHA256
保存在 manifest。預設會先比較每一格完整的追蹤／警示內容，再比較 CPU 與 CUDA 的
完整原始推論輸出；任一項不一致便停止量測。

原始逐格時間、輸出、環境資訊與 GPU 狀態保存在 `data/`，摘要寫入 `results.json`。可用
`--verify-only` 只做正確性檢查。基準程式用原始 Python class definition 執行追蹤器，
以避開 3.14 無法解析模組頂層 PEP 810 lazy import 的限制；追蹤器本身沒有重寫。
