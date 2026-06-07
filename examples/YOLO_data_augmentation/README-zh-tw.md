🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLO Data Augmentation

用於擴增 YOLO 格式工地資料集的工具。腳本預期資料集為常見結構：

```text
dataset/
  train/
    images/
    labels/
```

## 腳本

- `data_augmentation_albumentations.py`：使用 Albumentations 與 OpenCV 產生擴增圖片
  與對應 YOLO label。
- `visualise_bounding_boxes.py`：將 YOLO label 畫到單張圖片上，用於人工檢查。

## 執行資料增強

從 repo 根目錄執行：

```bash
python examples/YOLO_data_augmentation/data_augmentation_albumentations.py \
  --train_path dataset/train \
  --num_augmentations 30
```

腳本會在輸入 training set 旁寫入額外 image 與 label 檔。正式混入訓練資料前，請先抽樣
檢查生成結果。

## 視覺化檢查 Label

```bash
python examples/YOLO_data_augmentation/visualise_bounding_boxes.py \
  --image dataset/train/images/example.jpg \
  --label dataset/train/labels/example.txt \
  --save \
  --output visualised_image.jpg
```

## 注意事項

- class 順序必須和 training `data.yaml` 的 `names` 一致。
- 過小與過大的圖片會先 resize，避免 bounding box 變得不可用。
- 此目錄只用於資料準備；正式推論由 `src/yolo_worker.py` 與 `src/yolo_detector.py`
  負責。
