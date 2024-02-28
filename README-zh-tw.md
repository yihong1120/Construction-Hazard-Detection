以下是上述內容的繁體中文版本：

🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# 建築工地危險偵測

"建築工地危險偵測" 是一個旨在提高建築工地安全的人工智慧工具。利用 YOLOv8 模型進行物件偵測，此系統能夠識別潛在危險，如懸掛的重物和鋼管。對訓練好的模型進行後處理以提高準確率。該系統設計用於即時環境部署，為偵測到的任何危險提供立即分析和警告。

## 資料集資訊
訓練此模型的主要資料集是 [Roboflow 的建築工地安全圖像資料集](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow/data)。我們已對此資料集進行了額外的註解增強，並在 Roboflow 上公開訪問。增強後的資料集可以在此處找到：[Roboflow 上的建築危險偵測](https://universe.roboflow.com/side-projects/construction-hazard-detection)。此資料集包括以下標籤：

- `0: '安全帽'`
- `1: '口罩'`
- `2: '無安全帽'`
- `3: '無口罩'`
- `4: '無安全背心'`
- `5: '人員'`
- `6: '安全錐'`
- `7: '安全背心'`
- `8: '機械設備'`
- `9: '車輛'`

我們全面的資料集確保模型能夠識別常見於建築環境中的各種潛在危險。

## 安裝指南
要設置此項目，請遵循以下步驟：
1. 克隆存儲庫：
   ```
   git clone https://github.com/yihong1120/Construction-Hazard-Detection.git
   ```
2. 導航到項目目錄：
   ```
   cd Construction-Hazard-Detection
   ```
3. 安裝所需的依賴項：
   ```
   pip install -r requirements.txt
   ```

## 使用方法

此系統設計用於即時偵測和警告建築工地的危險。按照以下詳細步驟有效利用該系統：

### 準備環境
1. **設置硬體**：確保您擁有一台處理能力足夠的計算機和一台高品質攝像機，以在建築工地捕捉實時畫面。

2. **安裝攝像機**：策略性地放置攝像機，覆蓋處理重物和鋼管的高風險區域。

### 訓練模型

#### 收集數據
收集建築工地的圖片或視頻，重點關注各種危險，如重物、鋼管和人員存在。有關數據增強技術的示例，請訪問 [YOLOv8 數據增強範例](examples/YOLOv8-Data-Augmentation)。

#### 數據標記
準確識別並標註危險和人物圖像以進行數據標記。資料集文件中提供了詳細的註解指南。

#### 訓練 YOLOv8
使用註解後的資料集訓練 YOLOv8 模型。執行以下命令，根據您的資料集和硬件能力調整參數：
```
python train.py --model_name 'yolov8n.pt' --epochs 100 --data_config 'dataset/data.yaml'
```
有訓練指南和進階訓練選項，請參考 [YOLOv8 訓練範例](examples/YOLOv8-Train)。

### 後處理和部署

#### 應用後處理
訓練後，使用後處理技術提高模型的準確性。有關後處理方法和評估策略的詳細指南，請查看我們的 [YOLOv8 評估範例](examples/YOLOv8-Evaluation)。

#### 評估 YOLOv8
評估您的 YOLOv8 模型的性能對於確保其在現實場景中的有效性至關重要。我們提供兩種主要評估策略：直接評估 YOLOv8 模型和結合 SAHI+YOLOv8 的評估，以改善複雜場景中的檢測。要了解如何應用這些方法並解釋結果，請訪問 [YOLOv8 評估範例](examples/YOLOv8-Evaluation)。

#### 模型整合與系統運行
將訓練好的模型與能夠處理攝像機實時畫面的軟件進行整合。使用以下命令啟動系統，該命令將使用您的攝像頭畫面啟動檢測過程：
```
python src/demo.py
```

### 實時監控與警報
1. **監控**：系統將持續分析來自建築工地的實時畫面，偵測任何潛在危險。

2. **警報**：當系統偵測到人處於危險狀態時，將觸發警報。確保有一套機制（如連接的警報或通知系統）立即通知現場人員。

## 部署指南

要使用 Docker 部署 "建築工地危險偵測" 系統，請遵循以下步驟：

### 構建 Docker 鏡像
1. 確保 Docker Desktop 已安裝並在您的機器上運行。
2. 打開終端並導航到克隆存儲庫的根目錄。
3. 使用以下命令構建 Docker 鏡像：
   ```
   docker build -t construction-hazard-detection .
   ```

### 運行 Docker 容器
1. 鏡像構建完成後，可以使用以下命令運行容器：
   ```
   docker run -p 8080:8080 -e LINE_TOKEN=your_actual_token construction-hazard-detection
   ```
   將 `your_actual_token` 替換為您實際的 LINE Notify 令牌。

   此命令將啟動容器並

暴露應用程序的 8080 端口，使您能夠從主機機器在 `http://localhost:8080` 訪問它。

### 注意事項
- 確保 `Dockerfile` 存在於項目的根目錄中，並根據您的應用需求進行適當配置。
- `-e LINE_TOKEN=your_actual_token` 標誌在容器內設置 `LINE_TOKEN` 環境變量，這對於應用程序發送通知是必需的。如果您有其他環境變量，可以以類似方式添加它們。
- `-p 8080:8080` 標誌將容器的 8080 端口映射到您主機機器的 8080 端口，允許您通過主機的 IP 地址和端口號訪問應用程序。

有關 Docker 使用和命令的更多信息，請參考 [Docker 文檔](https://docs.docker.com/)。

## 貢獻
我們歡迎對此項目的貢獻。請按照以下步驟操作：
1. 分叉存儲庫。
2. 進行您的更改。
3. 提交一個清晰描述您改進的拉取請求。

## 開發路線圖
- [x] 數據收集和預處理。
- [x] 使用建築工地數據訓練 YOLOv8 模型。
- [x] 開發後處理技術以提高準確性。
- [x] 實施實時分析和警報系統。
- [x] 在模擬環境中進行測試和驗證。
- [ ] 在實際建築工地進行現場測試部署。
- [ ] 根據用戶反饋進行持續維護和更新。

## 授權
本項目採用 [AGPL-3.0 授權](LICENSE.md)。