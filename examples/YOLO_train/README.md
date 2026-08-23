🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLO Training

Training and export utilities for construction-hazard YOLO models. This folder
is separate from the live runtime: `main.py` only consumes exported `.pt` model
files through the YOLO worker path.

## Main Scripts

- `train.py`: trains, validates, predicts, and exports Ultralytics models.
- `export_int8_engine.py`: exports selected `models/pt` checkpoints to INT8
  TensorRT `.engine` files.
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

## Export INT8 TensorRT Engines

By default this uses `examples/YOLO_train/cv_dataset/data.yaml` for INT8
calibration and writes outputs to `models/int8_engine`:

```bash
python examples/YOLO_train/export_int8_engine.py yolo26n
```

If `cv_dataset` uses the flat `images/` and `labels/` layout, the script
generates a temporary calibration YAML that uses all `images/` and matching
`labels/` as `val`, so TensorRT does not require a separate `val/images`
directory.

This script is a thin wrapper around Ultralytics
`YOLO.export(format="engine", quantize=8)`. It defaults to `--fraction 1.0`,
which follows the official export argument for using the full calibration
dataset. Large datasets will substantially increase export time.

You can also pass a filename or path:

```bash
python examples/YOLO_train/export_int8_engine.py \
  models/pt/best_yolo26n.pt
```

For a single model, you can choose the output filename. A bare filename is
written under `models/int8_engine`:

```bash
python examples/YOLO_train/export_int8_engine.py yolo26n \
  -o yolo26n_int8.engine
```

You can also pass a full output path:

```bash
python examples/YOLO_train/export_int8_engine.py yolo26n \
  -o models/int8_engine/custom_yolo26n.engine
```

Export multiple models in one run:

```bash
python examples/YOLO_train/export_int8_engine.py yolo26n yolo26s yolo26m
```

Common options:

```bash
python examples/YOLO_train/export_int8_engine.py yolo26n \
  --data examples/YOLO_train/cv_dataset \
  --device 0 \
  --imgsz 640 \
  --batch 1 \
  --workspace 4 \
  --fraction 1.0
```

## Deploy A Checkpoint

Rename the selected checkpoint to the worker naming convention:

```text
models/pt/best_<model_key>.pt
```

For example, `models/pt/best_yolo26n.pt` is selected by stream configurations
whose `model_key` is `yolo26n`.
