# 同一 INT8 TensorRT engine 的 Python／Rust 比較

這個效能實驗以本機影片、測試圖片與 TensorRT engine 比較兩種推論路徑，執行後產生 `results.json`。

## 比較的實作

- Python：repo `uv.lock` 鎖定的 Ultralytics 8.4.183；此輪使用 `rect=True`、
  `quantize=8`。正式 worker 的 INT8 設定為 `rect=False`；本輪只重現矩形
  對照實驗，正式整合請見 [native/inference](../../native/inference/README.md)。
- Rust：Rust 負責批次、letterbox 尺寸、警示前的偵測結果篩選及座標還原；
  透過小型 C++ ABI 呼叫 OpenCV 與 TensorRT。BGR→RGB、HWC→CHW、正規化
  合併成一個 CUDA kernel，使用 NVRTC 於初始化時編譯。
- 兩邊使用**同一個現有 INT8 `.engine` 及相同 TensorRT 11.1 共享函式庫**，
  都呼叫同步 `executeV2`。Rust 不使用 Python 或 PyTorch 執行推論。
- 此實驗沒有使用 `ultralytics-inference` crate；該套件主要從 ONNX 載入，
  與直接沿用現有 engine 是不同實作路徑。

Rust 的 fused kernel、簡化結果建構及 Rust／C++ 的呼叫方式都會影響結果，
不能將差異全部歸因於程式語言。使用 PyTorch CUDA 建立的 256 項正規化 LUT
確保每個 uint8 值與參考實作得到相同浮點結果。

## 輸入、正確性及計時

`tests/videos/test.mp4` 每 18 格取樣，共 48 格，另加本機測試圖片 10 張。
輸入先解碼為 BGR bytes，讀檔及解碼不計時。保留 Ultralytics 的 `rect=True`、
stride=32、`imgsz=640`，所以實際 tensor 可以是 352×640 等矩形；不同形狀
混合成 batch 時使用完整 640×640。batch 8 的最後一批有 2 張。

測試 YOLO26n／YOLO26s、batch 1／8。每一個設定先比對 58 格：

1. 正規化後的輸入 tensor 必須逐位元相等，否則停止。
2. 比較所有原始網路輸出並記錄最大誤差。
3. 偵測數量、類別、順序必須一致；confidence 誤差不得超過 1e-5，原圖
   座標誤差不得超過 0.002 pixel，否則停止。

兩個量測範圍分開計時：

| 名稱 | 範圍 |
| --- | --- |
| `infer` | 同一批預先放在 GPU 的輸入，呼叫 TensorRT backend，輸出留在 GPU；包含主機呼叫與同步成本 |
| `detect` | 記憶體中的 BGR 影格→letterbox→傳輸／正規化→推論→信心值篩選→原圖座標與 CPU 偵測列 |

暖機後，固定在同一個 Intel P-core 執行 7 輪，Python／Rust 輪流先跑；
OpenCV／PyTorch 的 CPU 計算執行緒設為 1，driver／TensorRT 仍有輔助執行緒。
JSON 命令、結果讀寫、模型載入及初始化均在計時外；Rust 逐格結果配置計入。
`mean_ms_per_image` 使用該輪總耗時／實際影格數，沒有把最後 2 張當成 8 張。
`p95_ms_per_batch` 是批次延遲，不是逐張延遲。加速比為每輪比值的中位數，
最小／最大範圍不是信賴區間。這不是只用單一 CUDA event 得到的純 kernel 時間。

`detect` 不包含攝影機、解碼、shared-memory／queue IPC、batch 等待、追蹤、
危險規則、繪圖、編碼、儲存與通知，因此不是完整伺服器的可承載鏡頭數。
GPU 既有工作負載保留，每輪前後記錄 `nvidia-smi`；幾個百分點的差異需要
視為可能的外部負載／時脈波動。

## 重現

需要 Python 3.15 與 repo 的推論相依套件、Rust 1.93.1、C++17 compiler、
TensorRT 11.1 的 matching headers／runtime／Python bindings、OpenCV core
與 imgproc、CUDA runtime／driver／NVRTC。OpenCV 的測試版本為 C++ 4.10.0、
Python 5.0.0，前處理逐位元相等是必要驗證。CUDA kernel 此次以 `sm_89`
編譯，只驗證 RTX 4090；使用其他 GPU 前需調整 `bridge.cpp` 的架構選項。

在隔離環境準備可載入同一 engine 的 TensorRT 版本及共享函式庫。
`PYTHONPATH` 與 `LD_LIBRARY_PATH` 應指向彼此匹配的 Python bindings 與 runtime。

依環境設定路徑，`CUDA_INCLUDE` 必須是完整的 toolkit headers，`CUDA_LIB`
需提供可連結的 `libcudart.so`／`libnvrtc.so`：

```bash
export TRT_INCLUDE=/path/to/TensorRT-11.1/include
export TRT_LIB=/path/to/TensorRT-11.1/lib
export CUDA_INCLUDE=/path/to/cuda/include
export CUDA_LIB=/path/to/cuda/lib
export OPENCV_ROOT=/path/to/opencv-prefix  # 系統安裝可填 /
export PYTHONPATH=/path/to/isolated-python-packages
export LD_LIBRARY_PATH="$TRT_LIB:$CUDA_LIB:$OPENCV_ROOT/usr/lib/x86_64-linux-gnu"
export PYTHON_DOTENV_DISABLED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

cargo build --release --locked \
  --manifest-path experiments/rust_trt_benchmark/Cargo.toml
python experiments/rust_trt_benchmark/benchmark.py --prepare
python experiments/rust_trt_benchmark/benchmark.py \
  --models yolo26n yolo26s --batches 1 8 --rounds 7 --core 4
```

`data/` 忽略於 Git，保存輸入 manifest、BGR 影格、正規化 LUT、原始逐批時間、
worker log 及最近一次驗證的 tensor／網路輸出。每種設定的所有計時輪次摘要
保存在 `results.json`，包含 engine、runtime、原始碼與輸入的 SHA256。
`VmHWM` 是整個測試程序的 CPU 峰值 RSS，包含 framework、輸入及 buffers；
兩個程序都先載入相同的 58 格，不代表整套伺服器的每鏡頭記憶體。

參考：[NVIDIA TensorRT runtime API](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/c-api-docs.html)、
[Ultralytics 官方 Rust 推論](https://docs.ultralytics.com/inference/)。
