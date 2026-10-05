# 圖片偵測 overlay v2

相對於 detection base URL：`POST /v2/detect`。
原生 Bearer access token；Web 沿用既有 BFF 的轉送與認證入口。
與 /detect 共用權限、限流、上傳大小限制、模型版本驗證及推論程序。
舊 /detect 繼續回傳像素 xyxy 陣列，/models 與 WebSocket 格式沒有改變。

multipart 欄位：

| 欄位 | 說明 |
| --- | --- |
| image | 圖片檔案，必填 |
| model | /models 提供的模型 ID，必填 |
| model_version | /models 提供的版本，前端應傳入以固定類別意義 |
| compute_regions | boolean，預設 true，執行安全錐／電桿區域計算；false 才跳過 |
| view | regions（預設）或 objects；regions 必須同時 compute_regions=true，否則 422 |

預設執行分群且 view=regions，與違規詳情的一般顯示政策一致。
需要同時顯示全部物件框時傳 view=objects；只有明確不需要區域時才同時傳
compute_regions=false、view=objects。一般物件模式顯示安全錐，區域模式隱藏安全錐
個別框及名稱，回饋模式開關維持 true（僅顯示用途，不授予回饋權限）。

## 回應

回應直接為 JSON 物件，無 data 外層，與違規詳情共用 overlay schema 1：

```json
{
  "overlay_schema_version": 1,
  "model_id": "model-key",
  "model_version": "immutable-version",
  "view": "regions",
  "image_size": {"width": 640, "height": 480},
  "overlay_coordinates": {
    "space": "normalized", "bbox_format": "xywh", "polygon_format": "xy",
    "origin": "top_left", "reference": "original_image"
  },
  "class_metadata": [
    {"id": 42, "code": "safety_cone", "display_name": "交通錐", "color": "#FF5722"}
  ],
  "overlay_objects": [{
    "object_id": "det_0", "code": "safety_cone", "label": "safety_cone",
    "display_name": "交通錐", "color": "#FF5722", "confidence": 0.9,
    "bbox": {"x": 0.2, "y": 0.3, "w": 0.1, "h": 0.2},
    "show_box": false, "show_label": false,
    "feedback_show_box": true, "feedback_show_label": true,
    "is_flagged": false, "flag_reason": null, "flag_note": null
  }],
  "overlay_regions": [],
  "region_status": {"cone": "empty", "pole": "empty"}
}
```

範例省略完整 overlay_style 配色字典，實際會提供，格式與違規詳情一致。
分群完成有區域時回傳 available 與多筆 overlay_regions；每筆包含
id、kind、display_name、points、color、fill_opacity、closed。
執行後無區域回傳 empty，未執行為 not_computed。Flutter parser 須接受
not_computed。每個區域独立畫 Path，座標乘 BoxFit 後的 imageRect 並加偏移；
不要再次除原圖尺寸。區域先畫，再畫物件；依後端顯示開關控制框與名稱。

## 計算與限制

依模型 classes.code 將 safety_cone／utility_pole 映射至既有 geometry 的
語義 ID，再使用同一套 HDBSCAN 與區域幾何函式；不將所有模型的 6/9
直接視為安全錐／電桿。每個請求建立自己的 clusterer，在 worker thread
執行，單一 API worker 最多同時處理 2 個 overlay 建構任務。

保留既有 HDBSCAN 參數（min_samples=3、min_cluster_size=2），不強制將
噪聲點圍成管制區域。物件很多不保證有有效分群。電桿區域沿用既有緩衝
與合併規則，因此區域 ID 不代表可跨影格追蹤的 HDBSCAN cluster ID。
此端點不建立違規紀錄、不執行人員侵入警告、不宣告圈內為安全區。
歷史違規 API 仍讀取保存的區域，不在查詢時重新分群。

## 上線

先重啟 detection API，確認 /openapi.json 出現 /v2/detect；Flutter 再將
圖片偵測切換至新端點並共用 overlay painter。新前端不可用舊 /detect
回應當作 normalized overlay。資料庫不需 migration；違規 API 因共用
schema 模組也需隨同部署檔案。實機 iOS／Web 顯示仍需聯調。


兩個服務的 overlay 欄位現由 `examples/shared/overlay_schemas.py` 的
`OverlayPayload` 統一定義與繼承，欄位型別、預設值與座標契約一致。
偵測回應不包含尚未建立的違規 ID、工地或審核資料；這些業務欄位仍由
違規紀錄 API 提供。/v2/detect 預設計算並回傳所有有效區域；沒有有效
群集時仍是 overlay_regions=[] 和 empty，不能偽造多邊形。
