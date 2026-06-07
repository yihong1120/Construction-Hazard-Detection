🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLO Evaluation

用於評估訓練後 YOLO 工地危害模型的工具。建議在將新模型複製到 `models/pt/` 前，先用
這些腳本比較 checkpoint 表現。

## 腳本

- `convert_yolo_to_coco.py`：將 YOLO label 轉成 COCO JSON。
- `evaluate_yolo.py`：使用 Ultralytics 與 `data.yaml` 驗證模型。
- `evaluate_sahi_yolo.py`：使用 SAHI sliced inference 與 COCO metrics 評估。

## YOLO Label 轉 COCO

```bash
python examples/YOLO_evaluation/convert_yolo_to_coco.py \
  --labels_dir dataset/valid/labels \
  --images_dir dataset/valid/images \
  --output dataset/coco_annotations.json
```

## 使用 Ultralytics 評估

```bash
python examples/YOLO_evaluation/evaluate_yolo.py \
  --model_path models/pt/best_yolo26n.pt \
  --data_path dataset/data.yaml
```

## 使用 SAHI 評估

```bash
python examples/YOLO_evaluation/evaluate_sahi_yolo.py \
  --model_path models/pt/best_yolo26n.pt \
  --coco_json dataset/coco_annotations.json \
  --image_dir dataset/valid/images
```

SAHI 對小物件漏檢分析很有幫助，但比直接 Ultralytics validation 慢，因此建議用於模型
評估，而不是直播主路徑。

## 部署位置

正式 worker 會從 `models/pt/` 載入 `best_<model_key>.pt`。評估完成後，只將選定的
checkpoint 放入該目錄，並在 database management API 設定對應 stream 的 `model_key`。
