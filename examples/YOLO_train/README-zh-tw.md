🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLO Training

工地危害 YOLO 模型的訓練與匯出工具。此目錄和直播 runtime 分離；`main.py` 只會透過
YOLO worker 使用已匯出的 `.pt` 模型。

## 主要腳本

- `train.py`：訓練、驗證、預測與匯出 Ultralytics 模型。
- `export_int8_trt.py`：在有 calibration data 時匯出 TensorRT INT8 engine。
- `int_gen.py`：準備 INT8 匯出的 calibration images。
- `QAT.py`：quantisation-aware training 實驗。
- `test.py`：小型預測/測試 helper。

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

## 部署 Checkpoint

將選定的 checkpoint 重新命名為 worker 使用的規則：

```text
models/pt/best_<model_key>.pt
```

例如 `models/pt/best_yolo26n.pt` 會被 `model_key` 為 `yolo26n` 的 stream 設定選用。
