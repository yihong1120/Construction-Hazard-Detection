🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLOv8 微調範例

此儲存庫包含用於微調 YOLOv8 物件偵測模型的範例和腳本。特別旨在提升模型針對特定資料集或應用的性能。

## 使用方式

### 模型訓練

要訓練 YOLO 模型，請使用 `train.py` 腳本。指定模型名稱、訓練週期數及資料配置：

```bash
python train.py --model_name 'yolov8n.pt' --epochs 100 --data_config 'dataset/data.yaml'
```

### 使用腳本訓練 YOLO 模型

您可以使用 `train_yolo_models.sh` 腳本自動化不同 YOLO 模型的訓練過程：

```bash
bash train_yolo_models.sh
```

此腳本將會下載基礎模型（如果它們不存在），並開始每個模型的訓練過程。

### 模型導出

訓練後，您可以使用 `train.py` 腳本將您的 YOLO 模型導出到不同的格式，例如 ONNX：

```bash
python train.py --model_name 'yolov8n.pt' --export_format 'onnx' --onnx_path 'yolov8n.onnx'
```

### 模型預測

要使用 YOLO 模型進行預測，請指定預測的圖片路徑：

```bash
python train.py --model_name 'yolov8n.pt' --predict_image 'path/to/image.jpg'
```

## 特點

- **微調**：改善您的模型在特定資料集上的性能。
- **模型訓練**：從頭開始或使用預訓練的權重訓練 YOLO 模型。
- **模型導出**：將您訓練的模型導出到不同的格式以便部署。
- **模型預測**：輕鬆使用您訓練的模型對新圖片進行預測。
- **批次訓練**：使用提供的 shell 腳本訓練多個 YOLO 模型。

## 配置

根據您的資料集要求修改訓練參數，如週期數和資料配置。您也可以在 YOLO 配置文件中調整增強和超參數。