🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLO Data Augmentation

Utilities for expanding YOLO-format construction-site datasets. The scripts
expect the usual dataset layout:

```text
dataset/
  train/
    images/
    labels/
```

## Scripts

- `data_augmentation_albumentations.py`: creates augmented images and matching
  YOLO label files with Albumentations and OpenCV.
- `visualise_bounding_boxes.py`: draws YOLO labels onto one image for manual
  inspection.

## Run Augmentation

Run from the repository root:

```bash
python examples/YOLO_data_augmentation/data_augmentation_albumentations.py \
  --train_path dataset/train \
  --num_augmentations 30
```

The augmentation script writes additional image and label files next to the
input training set. Review the generated samples before mixing them into a
final training dataset.

## Check Labels Visually

```bash
python examples/YOLO_data_augmentation/visualise_bounding_boxes.py \
  --image dataset/train/images/example.jpg \
  --label dataset/train/labels/example.txt \
  --save \
  --output visualised_image.jpg
```

## Notes

- Keep class ordering aligned with the `names` section in the training
  `data.yaml`.
- Very small and very large images are resized before augmentation so bounding
  boxes remain usable.
- Use this folder for dataset preparation only; runtime inference is handled by
  `src/yolo_worker.py` and `src/yolo_detector.py`.
