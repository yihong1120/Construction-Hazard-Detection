🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLO Evaluation

Evaluation helpers for trained YOLO construction-hazard models. Use these
scripts after training to compare model checkpoints and check whether a new
model is good enough to copy into `models/pt/`.

## Scripts

- `convert_yolo_to_coco.py`: converts YOLO label files into COCO JSON.
- `evaluate_yolo.py`: runs Ultralytics validation with a `data.yaml`.
- `evaluate_sahi_yolo.py`: runs sliced SAHI inference and COCO metrics.

## Convert YOLO Labels To COCO

```bash
python examples/YOLO_evaluation/convert_yolo_to_coco.py \
  --labels_dir dataset/valid/labels \
  --images_dir dataset/valid/images \
  --output dataset/coco_annotations.json
```

## Evaluate With Ultralytics

```bash
python examples/YOLO_evaluation/evaluate_yolo.py \
  --model_path models/pt/best_yolo26n.pt \
  --data_path dataset/data.yaml
```

## Evaluate With SAHI

```bash
python examples/YOLO_evaluation/evaluate_sahi_yolo.py \
  --model_path models/pt/best_yolo26n.pt \
  --coco_json dataset/coco_annotations.json \
  --image_dir dataset/valid/images
```

SAHI is useful when small objects are missed at normal image scale. It is
slower than direct Ultralytics validation, so use it for model assessment rather
than the live stream path.

## Runtime Placement

Production workers load models named `best_<model_key>.pt` from `models/pt/`.
After evaluation, copy only the selected checkpoint into that directory and set
the matching stream `model_key` in the database management API.
