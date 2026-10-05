# Flutter 模型清單 API（後端實作）

本文件記錄本 repository 的實作。前端提供的外部交接文件尚未取得，
**錯誤 JSON、404／503 與空清單的 `null` 語意仍須與 Flutter 確認**。

## 路由與認證

| 用途 | 模型清單 | 偵測 |
| --- | --- | --- |
| YOLO 服務 | `GET /models` | `POST /detect` |
| Web BFF | `GET /bff/detection/models` | `POST /bff/detection/detect` |
| native（附帶 Nginx 範例） | `GET /hazard/api/detect/models` | `POST /hazard/api/detect/detect` |

部署若另有外層前綴，依現有 detectionBaseUrl 加上 `/models`。
Web 延用 BFF session cookie，native 延用 Bearer JWT。
BFF 的 detection 服務轉送及 Nginx 的 `/hazard/api/detect/` 規則已涵蓋此端點。
目前偵測服務沒有逐模型權限配置：所有通過相同 JWT 驗證的使用者
（含 guest）可使用相同清單。管理模型檔案的角色限制不變。
清單回應使用 `Cache-Control: no-store`，不消耗偵測速率配額。

## 模型清單

```json
{
  "default_model_id": "yolo26n",
  "models": [
    {"id": "yolo26n", "display_name": "YOLO26n"}
  ]
}
```

- 候選來自 `MODEL_VARIANTS`，依設定順序去重，且對應模型檔案必須存在、可讀。
- 一般及 SAHI 模式使用 `models/pt/best_<id>.pt`；TensorRT 使用
  `models/int8_engine/best_<id>.engine`。
- 延遲載入及 LRU 淘汰的模型仍可列出；查詢清單不載入所有模型至 GPU。
  因此檔案存在不代表已完成推論驗證，偵測時仍可能回報載入失敗。
- 已知載入失敗的檔案版本不列入清單；更換檔案（mtime 或大小改變）、
  成功手動重載或重啟服務後可重新嘗試。
- `DEFAULT_MODEL_ID` 預設為 `yolo26n`。若不在可用清單，選取清單最後一項
  （與現有預載最小模型的排序慣例一致）。
- 無可用模型時回傳 HTTP 200：`{"default_model_id": null, "models": []}`。
- ID 大小寫有別，偵測不會自動替換模型或將不合法 ID 正規化成另一模型。

## 偵測及模型錯誤

`POST /detect` 延用 multipart 欄位 `image`、`model`；成功回應維持
`[[x1, y1, x2, y2, confidence, class_id], ...]`。
每次請求都重新驗證 ID、模型檔案與載入狀態，即使模型已快取也會檢查檔案。

未註冊、已從設定移除或不合法的 ID 回傳 **404**：

```json
{"detail": {"code": "MODEL_NOT_FOUND", "message": "Model not found"}}
```

已註冊但檔案移除、不可讀或載入失敗，回傳 **503**：

```json
{"detail": {"code": "MODEL_UNAVAILABLE", "message": "Model is unavailable"}}
```

驗證錯誤、認證失敗、速率限制及一般推論錯誤沿用既有行為。
此變更未修改偵測類別或 class ID；部署新權重前須確認與既有對應相容。
修改 `MODEL_VARIANTS`／`DEFAULT_MODEL_ID` 環境設定後須重啟服務。

## 上線順序與聯調

1. 先部署後端，使用實際 native JWT 及 Web BFF session 分別查詢模型清單。
2. 以清單的 default_model_id 上傳圖片，確認既有偵測結果及 class ID。
3. 確認 Flutter 可解析上述錯誤格式與空清單，並測試模型下架、重試及認證失敗。
4. 完成聯調後再發布新版 Flutter；舊版後端沒有 `/models`，新版前端會停用偵測。

Repository 測試使用隔離的認證／推論替身；不代表已完成實際 GPU、Nginx 或 Flutter 聯調。
