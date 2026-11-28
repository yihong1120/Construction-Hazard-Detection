🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# YOLO 資料增強

用於擴增 YOLO 格式工地資料集的工具。腳本預期資料集為常見結構：

```text
dataset/
  train/
    images/
    labels/
```

## 安裝相依套件

從專案根目錄執行 `uv sync --locked --extra training`。使用 CPython 3.15 匯出 TensorRT
引擎時，請依[原生套件安裝說明](../../README-zh-tw.md#1-準備-python)建置 wheel，之後使用
`uv run --no-sync python` 執行指令，以保留這些套件。

## 腳本

- `data_augmentation_albumentations.py`：使用 Albumentations 與 OpenCV 產生擴增圖片
  與對應 YOLO 標註。
- `visualise_bounding_boxes.py`：將 YOLO 標註 畫到單張圖片上，用於人工檢查。

## 執行資料增強

從專案根目錄執行下列指令。

```bash
uv run --no-sync python examples/YOLO_data_augmentation/data_augmentation_albumentations.py \
  --train_path dataset/train \
  --num_augmentations 30
```

工具會把新增的影像和標註檔寫到原訓練資料旁。納入訓練集之前，請先抽樣檢查結果。

## 目視檢查標註框

```bash
uv run --no-sync python examples/YOLO_data_augmentation/visualise_bounding_boxes.py \
  --image dataset/train/images/example.jpg \
  --label dataset/train/labels/example.txt \
  --save \
  --output visualised_image.jpg
```

## 注意事項

- 類別順序須和訓練設定檔 `data.yaml` 的 `names` 一致。
- 程式會先調整過小或過大的圖片尺寸，確保標註框仍然有效。
- 此目錄只用於資料準備；正式推論由 `src/yolo_worker.py` 與 `src/yolo_detector.py`
  負責。
