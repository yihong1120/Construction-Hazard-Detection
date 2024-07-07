🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLOv8 評估範例

此儲存庫提供了用於評估 YOLOv8 物件偵測模型的範例和腳本，特別著重於評估特定數據集上模型的準確度和性能。

## 使用方法

### 將 YOLO 格式轉換為 COCO 格式

要將標註從 YOLO 格式轉換為 COCO 格式，請使用 `convert_yolo_to_coco.py` 腳本。指定 YOLO 標籤和圖片的目錄，以及 COCO 格式 JSON 的輸出路徑：

```bash
python convert_yolo_to_coco.py --labels_dir dataset/valid/labels --images_dir dataset/valid/images --output dataset/coco_annotations.json
```

### 使用 SAHI 評估模型

要使用 SAHI 庫來評估 YOLOv8 模型，請運行 `evaluate_sahi_yolov8.py` 腳本。提供模型、COCO JSON 和圖片目錄的路徑：

```bash
python evaluate_sahi_yolov8.py --model_path "../../models/best_yolov8n.pt" --coco_json "dataset/coco_annotations.json" --image_dir "dataset/valid/images"
```

此腳本將輸出各種 IoU 閾值下的評估指標，如平均精度和召回率。

### 使用 Ultralytics YOLO 評估模型

要使用 Ultralytics 框架進行評估，請執行 `evaluate_yolov8.py` 腳本。同樣地，指定模型和數據配置文件的路徑：

```bash
python evaluate_yolov8.py --model_path "../../models/best_yolov8n.pt" --data_path "dataset/data.yaml"
```

### 功能

- **YOLO 至 COCO 轉換**：將您的數據集標註從 YOLO 格式轉換為 COCO 格式，以便與各種評估工具兼容。
- **模型評估**：使用 SAHI 庫或 Ultralytics 框架來評估您的 YOLOv8 模型。
- **性能指標**：透過像是 mAP、精度和召回率等指標，獲得模型性能的深入見解。
- **可自訂的評估**：腳本允許針對您特定的數據集和模型進行靈活的評估設置。

## 配置

根據您特定數據集的需求，調整評估參數，如信心閾值和 IoU 指標。根據您的項目目錄結構，需要調整模型、數據集和標註文件的路徑。
