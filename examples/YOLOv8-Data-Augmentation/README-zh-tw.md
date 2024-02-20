🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLOv8 資料增強範例

本倉庫包含用於對影像資料集進行資料增強的範例和腳本，特別是旨在提高 YOLOv8 物件偵測模型的表現。對於需要豐富資料集以改善模型訓練的機器學習項目尤其有用。

## 使用方式

### 資料增強

要對您的資料集進行資料增強，請使用 `data_augmentation.py` 腳本。指定您的訓練資料路徑和每張圖片的增強數量：

```bash
python data_augmentation.py --train_path '../path/to/your/data' --num_augmentations 30
```

### 移動和重命名增強後的檔案

增強後，您可以使用 `run_augmentation_and_move.sh` 腳本將增強後的圖像和標籤檔案移動並重命名到主資料集目錄：

```bash
bash run_augmentation_and_move.sh
```

此腳本確保增強檔案正確排序並存儲於現有資料集檔案中。

### 視覺化邊界框

要在圖像上視覺化邊界框，請使用 `visualise_bounding_boxes.py` 腳本。提供圖像和相應標籤檔案的路徑：

```bash
python src/visualise_bounding_boxes.py --image 'path/to/image.jpg' --label 'path/to/label.txt'
```

您可以通過使用 `--save` 標誌並指定 `--output` 路徑來選擇保存視覺化圖像。

## 特點

- **資料增強**：通過應用一系列隨機變換來增強您的圖像資料集。
- **並行處理**：利用多線程進行更快的增強處理。
- **邊界框保留**：確保增強圖像保持準確的邊界框註釋。
- **混洗**：隨機排列資料集的順序以防止模型過度擬合。
- **視覺化**：輕鬆在您的圖像上視覺化邊界框以進行驗證。

## 配置

可以通過修改 `data_augmentation.py` 腳本中的增強序列來自定義資料增強過程。您可以根據您的資料集要求添加或移除變換。