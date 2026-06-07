🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLO Training

Training and export utilities for construction-hazard YOLO models. This folder
is separate from the live runtime: `main.py` only consumes exported `.pt` model
files through the YOLO worker path.

## Main Scripts

- `train.py`: trains, validates, predicts, and exports Ultralytics models.
- `export_int8_trt.py`: exports TensorRT INT8 engines when calibration data is
  available.
- `int_gen.py`: prepares calibration images for INT8 export.
- `QAT.py`: quantisation-aware training experiment.
- `test.py`: small prediction/testing helper.

## Train A Model

```bash
python examples/YOLO_train/train.py \
  --model_name yolo11n.pt \
  --epochs 100 \
  --data_config dataset/data.yaml
```

The training script chooses CUDA, MPS, or CPU according to the host. Use a batch
size that fits GPU memory.

## Validate Or Predict

```bash
python examples/YOLO_train/train.py \
  --model_name runs/detect/train/weights/best.pt \
  --predict_image dataset/valid/images/example.jpg
```

## Export

```bash
python examples/YOLO_train/train.py \
  --model_name runs/detect/train/weights/best.pt \
  --export_format onnx
```

## Deploy A Checkpoint

Rename the selected checkpoint to the worker naming convention:

```text
models/pt/best_<model_key>.pt
```

For example, `models/pt/best_yolo26n.pt` is selected by stream configurations
whose `model_key` is `yolo26n`.
