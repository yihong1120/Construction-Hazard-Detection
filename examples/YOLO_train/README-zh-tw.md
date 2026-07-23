🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLO Training

工地危害 YOLO 模型的訓練與匯出工具。此目錄和直播 runtime 分離；`main.py` 只會透過
YOLO worker 使用已匯出的 `.pt` 模型。

## 主要腳本

- `train.py`：訓練、驗證、預測與匯出 Ultralytics 模型。
- `export_int8_engine.py`：將 `models/pt` 內指定 `.pt` checkpoint 匯出成 INT8 TensorRT `.engine`。
- `test.py`：小型預測/測試 helper。

TensorRT engine 重建工具位於 `scripts/rebuild_tensorrt_engines.py` 與
`scripts/rebuild_single_engine.py`。

## 訓練模型

```bash
python examples/YOLO_train/train.py \
  --model_name yolo11n.pt \
  --epochs 100 \
  --data_config dataset/data.yaml
```

訓練腳本會依主機狀態選擇 CUDA、MPS 或 CPU。batch size 請依 GPU 記憶體調整。

## 驗證或預測

```bash
python examples/YOLO_train/train.py \
  --model_name runs/detect/train/weights/best.pt \
  --predict_image dataset/valid/images/example.jpg
```

## 匯出

```bash
python examples/YOLO_train/train.py \
  --model_name runs/detect/train/weights/best.pt \
  --export_format onnx
```

## 匯出 INT8 TensorRT Engine

預設會使用 `examples/YOLO_train/cv_dataset/data.yaml` 作為 INT8 calibration
資料，輸出到 `models/int8_engine`：

```bash
python examples/YOLO_train/export_int8_engine.py yolo26n
```

如果 `cv_dataset` 是 `images/`、`labels/` 的扁平結構，腳本會自動產生暫時
calibration YAML，將全部 `images/` 和對應 `labels/` 作為 `val` 使用，讓
TensorRT 不需要額外的 `val/images` 目錄。

此腳本是 Ultralytics `YOLO.export(format="engine", quantize=8)` 的薄包裝。
預設 `--fraction 1.0`，依官方參數設定使用完整 calibration dataset；大量圖片會
明顯增加匯出時間。

也可以直接指定檔名或路徑：

```bash
python examples/YOLO_train/export_int8_engine.py \
  models/pt/best_yolo26n.pt
```

單一模型可指定輸出檔名；只給檔名時會輸出到 `models/int8_engine`：

```bash
python examples/YOLO_train/export_int8_engine.py yolo26n \
  -o yolo26n_int8.engine
```

也可以指定完整輸出路徑：

```bash
python examples/YOLO_train/export_int8_engine.py yolo26n \
  -o models/int8_engine/custom_yolo26n.engine
```

一次匯出多個模型：

```bash
python examples/YOLO_train/export_int8_engine.py yolo26n yolo26s yolo26m
```

常用參數：

```bash
python examples/YOLO_train/export_int8_engine.py yolo26n \
  --data examples/YOLO_train/cv_dataset \
  --device 0 \
  --imgsz 640 \
  --batch 1 \
  --workspace 4 \
  --fraction 1.0
```

## 部署 Checkpoint

將選定的 checkpoint 重新命名為 worker 使用的規則：

```text
models/pt/best_<model_key>.pt
```

例如 `models/pt/best_yolo26n.pt` 會被 `model_key` 為 `yolo26n` 的 stream 設定選用。
