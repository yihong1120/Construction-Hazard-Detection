🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# YOLO 模型訓練

訓練與匯出工地危害辨識用的 YOLO 模型。即時辨識的工作程序會從模型目錄載入已發布的模型檔。

## 安裝相依套件

從專案根目錄執行 `uv sync --locked --extra training`。使用 CPython 3.15 匯出 TensorRT
引擎時，請依[原生套件安裝說明](../../README-zh-tw.md#1-準備-python)建置 wheel，之後使用
`uv run --no-sync python` 執行指令，以保留這些套件。

## 主要腳本

- `train.py`：訓練、驗證、預測與匯出 Ultralytics 模型。
- `export_int8_engine.py`：將 `models/pt` 內指定 `.pt` checkpoint 匯出成 INT8 TensorRT `.engine`。
- `test.py`：選用的本機預測工具，未納入版本庫。

## 訓練模型

```bash
uv run --no-sync python examples/YOLO_train/train.py \
  --model_name yolo11n.pt \
  --epochs 100 \
  --data_config dataset/data.yaml
```

訓練腳本會依主機狀態選擇 CUDA、MPS 或 CPU。batch size 請依 GPU 記憶體調整。

## 驗證或預測

```bash
uv run --no-sync python examples/YOLO_train/train.py \
  --model_name runs/detect/train/weights/best.pt \
  --predict_image dataset/valid/images/example.jpg
```

## 匯出

```bash
uv run --no-sync python examples/YOLO_train/train.py \
  --model_name runs/detect/train/weights/best.pt \
  --export_format onnx
```

## 匯出 INT8 TensorRT 引擎

預設會使用 `examples/YOLO_train/cv_dataset/data.yaml` 作為 INT8 calibration
資料，輸出到 `models/int8_engine`：

```bash
uv run --no-sync python examples/YOLO_train/export_int8_engine.py yolo26n
```

如果 `cv_dataset` 是 `images/`、`labels/` 的扁平結構，腳本會自動產生暫時
calibration YAML，將全部 `images/` 和對應 `labels/` 作為 `val` 使用，讓
TensorRT 不需要額外的 `val/images` 目錄。

此腳本是 Ultralytics `YOLO.export(format="engine", quantize=8)` 的薄包裝。
預設 `--fraction 1.0`，依官方參數設定使用完整 calibration dataset；大量圖片會
明顯增加匯出時間。

也可以直接指定檔名或路徑：

```bash
uv run --no-sync python examples/YOLO_train/export_int8_engine.py \
  models/pt/best_yolo26n.pt
```

單一模型可指定輸出檔名；只給檔名時會輸出到 `models/int8_engine`：

```bash
uv run --no-sync python examples/YOLO_train/export_int8_engine.py yolo26n \
  -o yolo26n_int8.engine
```

也可以指定完整輸出路徑：

```bash
uv run --no-sync python examples/YOLO_train/export_int8_engine.py yolo26n \
  -o models/int8_engine/custom_yolo26n.engine
```

一次匯出多個模型：

```bash
uv run --no-sync python examples/YOLO_train/export_int8_engine.py yolo26n yolo26s yolo26m
```

常用參數：

```bash
uv run --no-sync python examples/YOLO_train/export_int8_engine.py yolo26n \
  --data examples/YOLO_train/cv_dataset \
  --device 0 \
  --imgsz 640 \
  --batch 1 \
  --workspace 4 \
  --fraction 1.0
```

## 部署模型權重

在模型目錄中發布選好的權重檔，登錄路徑、版本及 SHA-256，再將串流的 `model_key`
設為該模型 ID。使用 INT8 推論時，引擎清單須對應同一份來源權重與版本。
