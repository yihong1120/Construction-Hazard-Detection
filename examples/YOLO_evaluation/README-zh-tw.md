🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# YOLO 模型評估

用於評估訓練後 YOLO 工地危害模型的工具。建議在將新模型複製到 `models/pt/` 前，先用
這些腳本比較 checkpoint 表現。

## 安裝相依套件

從專案根目錄執行 `uv sync --locked --extra training`。使用 CPython 3.15 匯出 TensorRT
引擎時，請依[原生套件安裝說明](../../README-zh-tw.md#1-準備-python)建置 wheel，之後使用
`uv run --no-sync python` 執行指令，以保留這些套件。

## 腳本

- `convert_yolo_to_coco.py`：將 YOLO label 轉成 COCO JSON。
- `evaluate_yolo.py`：使用 Ultralytics 與 `data.yaml` 驗證模型。
- `evaluate_sahi_yolo.py`：使用 SAHI sliced inference 與 COCO metrics 評估。

## 將 YOLO 標註轉成 COCO 格式

```bash
uv run --no-sync python examples/YOLO_evaluation/convert_yolo_to_coco.py \
  --labels_dir dataset/valid/labels \
  --images_dir dataset/valid/images \
  --output dataset/coco_annotations.json
```

## 使用 Ultralytics 評估

```bash
uv run --no-sync python examples/YOLO_evaluation/evaluate_yolo.py \
  --model_path models/pt/best_yolo26n.pt \
  --data_path dataset/data.yaml
```

## 使用 SAHI 評估

```bash
uv run --no-sync python examples/YOLO_evaluation/evaluate_sahi_yolo.py \
  --model_path models/pt/best_yolo26n.pt \
  --coco_json dataset/coco_annotations.json \
  --image_dir dataset/valid/images
```

SAHI 對小物件漏檢分析很有幫助，但比直接 Ultralytics validation 慢，因此建議用於模型
評估，而不是直播主路徑。

## 部署位置

完成評估後，在模型目錄發布選好的權重檔，再透過資料管理 API 設定串流的 `model_key`。
