🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Violation Records Backend

用於儲存、查詢與提供工地危害違規紀錄的 FastAPI 服務。當 stream processor 確認警示事件後，
`src/violation_sender.py` 會將違規圖片與 metadata 上傳到這裡。

## 職責

- 將違規 metadata 儲存在 PostgreSQL。
- 將上傳的違規圖片存到受控的 `static/` 目錄。
- 強制 JWT 與 site-based access control。
- 提供 filtered、paginated violation search。
- 透過安全的 relative path 提供違規圖片。
- 使用 synonym expansion 支援多語 keyword search。

## 檔案

- `app.py`：FastAPI application。
- `routers.py`：record list、detail、upload、site 與 image routes。
- `schemas.py`：Pydantic request 與 response models。
- `violation_manager.py`：database 與 image-storage logic。
- `path_utils.py`：安全 static-path validation。
- `search_utils.py`：多語詞典搜尋，不需下載 NLP 模型。
- `violation_analytics.py`：聚合 SQL 與圖表回應組裝。
- `settings.py`：static-directory settings。

## 執行

```bash
uvicorn examples.violation_records.app:app \
  --host 127.0.0.1 \
  --port 8002 \
  --workers 2
```

module `main()` 直接執行時仍預設 port `8081`，但專案 runtime 使用 port `8002`。

## 必要設定

升級 PostgreSQL 不會自動補上專案新增的索引。既有資料庫應由管理員使用本機維護的
索引遷移腳本增補索引；遷移腳本與量測、調校紀錄不隨 repo 發布。
勿對既有資料庫重跑會重建資料表的 `init.postgres.sql`。

```dotenv
VIOLATION_RECORD_API_URL=http://127.0.0.1:8002
DATABASE_URL=postgresql+asyncpg://username:password@127.0.0.1/construction_hazard_detection
OIDC_ENABLED=true
OIDC_ISSUER_URL=https://sso.example.com/realms/visionnaire
OIDC_JWKS_URL=https://sso.example.com/realms/visionnaire/protocol/openid-connect/certs
OIDC_AUDIENCE=visionnaire-api
OIDC_ACCOUNT_URL=https://sso.example.com/realms/visionnaire/account
```

## Endpoints

- `GET /my_sites`：登入使用者可見的 sites。
- `GET /violations/filter-options?site_id=...`：回傳登入者在所選工地可用的
  鏡頭與固定違規項目；省略 `site_id` 時，回傳所有授權工地的鏡頭。
  `stream_id` 是鏡頭設定的數字 ID，
  不是鏡頭名稱。
- `GET /violations`：filtered 與 paginated violation list。支援選用的
  `site_id`、`stream_id`、`violation_type`、時間範圍；審核者可加 `flagged=true`
  與選用的 `review_status=pending|resolved|dismissed`。
- `GET /violations/analytics`：違規趨勢、排行與時段統計；僅 `admin` 與
  `super_admin` 可存取，支援與紀錄頁相同的 `site_id`、`stream_id`、
  `violation_type`、`start`、`end`、`bucket` 條件，且所有聚合都會套用；
  其他角色一律回傳 `403 violation_analytics_forbidden`。
- `GET /violations/{violation_id}`：單筆違規細節。
- `POST /violations/{violation_id}/feedback`：送出誤判、漏判、類別錯誤或 bbox
  錯誤的結構化回饋，預設進入待審核狀態。
- `PATCH /violations/{violation_id}/review`：admin / super_admin 更新 flagged record
  的審核狀態。
- `GET /get_violation_image?image_path=...`：從 `static/` 回傳圖片。
- `POST /upload`：stream processor 上傳圖片與 metadata。

## 保留策略

若是沒有 archive storage 的本機硬碟部署，圖片檔與 DB 紀錄應使用相同保留週期。目前建議
保留 18 個月，除非法律或合約要求不同期限。

刪除舊資料時，請同時移除 DB rows 與對應檔案。不要留下 `static/` 孤兒圖片，也不要保留
圖片路徑已不存在的紀錄。

## 儲存注意事項

- `static/` 目錄是營運證據儲存，不是 live stream buffer。
- HLS/WebRTC live segments 屬於 MediaMTX，應使用短很多的 retention window。
- 如果磁碟空間接近上限，請降低保留期限或先匯出紀錄再刪除。


## 回饋與審核選項

選項由本服務依已驗證的帳號、工地權限及指定紀錄產生：

- `GET /violations/{violation_id}/feedback-options?locale=zh-TW`
- `GET /violations/{violation_id}/review-options?locale=zh-TW`

原生端使用 `Authorization: Bearer <access_token>`。目前 Nginx 的
violation base URL 是 `https://changdar-server.mooo.com/hazard/api/violations/`，
因此回饋選項完整 URL 為
`https://changdar-server.mooo.com/hazard/api/violations/violations/{violation_id}/feedback-options?locale=zh-TW`。
前一個 `violations` 是反向代理前綴，後一個是本服務既有資源路徑。

回應維持 `schema_version: 1`、`revision`、`options: [{id, label}]`、
`default_id: null`、`policy: {}`，並設定 `Cache-Control: private, no-store`。
回饋類別只取原始快照；缺快照時維持空 options，不查最新模型。審核選項取決於操作者角色及目前審核狀態。
提交仍使用既有 feedback/review API 並重新驗證。

management 的 `violation_feedback_labels`／`violation_review_actions` scope
已移除並回傳 404；前端須直接使用上述違規端點。
部署後需重啟違規服務；程式修改不會自動更新運行中的路由。


### 人工漏判回報

使用既有 `POST /violations/{id}/feedback`，例如：

```json
{
  "type": "false_negative",
  "corrected_label": "person",
  "corrected_bbox": [10, 20, 100, 200],
  "note": "此處漏判一名人員"
}
```

`corrected_label` 必須使用 feedback-options 提供的穩定 code；
`corrected_bbox` 延續既有 `[x1, y1, x2, y2]` 座標契約。
人工回饋不要求、不寫入，也不回傳模型 ID／版本。
舊客戶端多傳的 model_version 不會參與驗證或保存。
original_label 僅保留作為客戶端回報的原始標籤文字，不代表模型識別。

前端按鈕根據選項載入狀態及可用類別判斷；紀錄快照決定可用回饋類別。
選項與提交都檢查紀錄權限；提交時重驗類別與座標。
模型推論的版本校驗及舊紀錄既有 metadata 不受此人工標註流程影響。
不需 migration 或回填歷史資料；既有 feedback 的舊版本欄位保留，
新回饋不填寫該欄位。

### 類別名稱與顏色

`/models` 的 classes、違規詳情與列表的 `class_metadata`，以及
feedback-options 的 options 共用類別 code 與 `#RRGGBB` 顏色。
建立紀錄時將目錄類別保存為 model_classes；對外 class_metadata 為
該快照的名稱、數字 ID、code 與顏色。有效顏色標準化為大寫，
缺失或無效色由後端以穩定 code 補色，不影響名稱或回饋驗證。

新模型調色不修改歷史快照；禁止以最新目錄回填未知來源的舊紀錄。
回饋 POST 不需 color 或模型識別，伺服器在鎖定紀錄後重驗快照 code。
可保留原始影像供 App 疊圖；串流伺服器繪圖也使用推論模型的 metadata。
本次沒有新增 API 路徑或資料表欄位，不恢復 /client-options。

### 全部框色由後端決定

詳情與列表增加 `overlay_style`（schema_version 1）：

```json
{
  "schema_version": 1,
  "class_colors": {"person": "#FF9800", "vehicle": "#FFEB3B"},
  "unknown_class_color": "#9E9E9E",
  "cone_polygon_color": "#FF4081",
  "pole_polygon_color": "#448AFF",
  "warning_color": "#F44336"
}
```

class_colors 實際包含後端完整配色，且該筆有效快照色會覆蓋同 code。
詳情 `overlay_objects[].color` 為已解析的框色，可直接繪製；標記狀態
不改变它。自行繪製 detection_items 時，先用 class_metadata 的 ID/code
對應與 color，再用後端 class_colors；不能確認數字 ID 語義時使用
unknown_class_color，不可查最新模型猜測。區域線使用相應 polygon_color。

Flutter 必須移除本機固定色與 hash 備援；後端顏色資料缺失或無效時，
顯示配色載入失敗／重試，保留原圖，暫停相應疊圖。API 增加欄位不會
自動停用舊 App 的 fallback。此主機沒有 Flutter 專案，需 App 端同步修改。
回饋選項和提交共用同一份已解析類別表，仍須通過紀錄權限檢查。
已燒錄進 JPEG 的舊框色無法透過 metadata 改變。

### 後端分群與歷史區域繪圖（overlay schema 1）

串流 DangerDetector 執行 HDBSCAN，分群區域同時用於危險判定與即時繪圖。
同一次 detect_danger 回傳的 warnings、cone/pole polygons 隨證據影像一起
寫入違規紀錄。區域偵測仍受串流功能開關控制，不因前端開啟疊圖而啟用。
上傳 polygons 為原圖像素的 `[[[x,y],...],...]` JSON 字串；後端驗證點數、
數值有限性與非零面積，不接受 NaN 或退化邊界。null 表示未提供，[] 表示
保存的結果為空，不代表一定已啟用／執行分群。

`GET /violations/{id}` 新增以下欄位，不改動原有資料 envelope：

```json
{
  "overlay_schema_version": 1,
  "image_size": {"width": 640, "height": 360},
  "overlay_coordinates": {
    "space": "normalized",
    "bbox_format": "xywh",
    "polygon_format": "xy",
    "origin": "top_left",
    "reference": "original_image"
  },
  "overlay_regions": [{
    "id": "cone_0",
    "kind": "cone",
    "display_name": "安全錐管制區域",
    "points": [[0.1, 0.2], [0.5, 0.2], [0.5, 0.8], [0.1, 0.2]],
    "color": "#FF4081",
    "fill_opacity": 0.4,
    "closed": true
  }],
  "region_status": {"cone": "available", "pole": "empty"}
}
```

範例 points 僅示意。讀取時只將已保存邊界除以原圖寬高，絕不呼叫分群、
讀目前攝影機配置或替歷史資料製造區域。region ID 是該紀錄內的邊界識別，
不是可跨影格追蹤的 cluster ID。區域座標可超出 0～1（例如緩衝區超出圖片），
前端應裁切至圖片範圍，不能將每個頂點 clamp 後改變形狀。

region_status 每種區域各自回傳：
- available：有可繪製的已保存邊界。
- empty：已保存空陣列；無法只憑此狀態區分未啟用或未形成有效群聚。
- not_recorded：沒有保存資料，不表示當時沒有群聚。
- invalid：舊資料無法解析或邊界不合法；其他證據仍可讀。
- image_unavailable：有邊界但無法取得原圖尺寸，不能安全換算。

物件 overlay_objects 新增 code 與 display_name；有快照時直接顯示
`display_name` 並使用 `color`，label 保留既有值相容。無法解析的 ID 仍為 class-N 與灰色。
不使用目前模型猜測歷史數字 ID。類別解析與區域
幾何的讀取獨立。

Flutter 整合：列表是精簡摘要，進詳情須 GET 詳情；先畫區域填色／邊界，
再畫物件框及名稱。使用 BoxFit 後的 imageRect：
`px = imageRect.left + x * imageRect.width`、
`py = imageRect.top + y * imageRect.height`。
框的 w/h 分別乘顯示寬高，不能把 w/h 當 x2/y2，也不能再除原圖尺寸一次。
若缺原圖尺寸，image_size 為 null，不提供無法換算的像素區域。

保留 cone_polygons/pole_polygons 原始字串供舊客戶端；新客戶端只畫
解析完成的 overlay_regions，避免兩條資料流重複畫同一區域。
本次沿用資料庫已有的區域欄位，不需 migration，不回填未知歷史類別。

### 列印前端實際收到的 JSON

設定 `VIOLATION_DEBUG_RESPONSE_IDS=*` 可列印所有紀錄；也可指定
`155157,155158` 限定紀錄。修改後重啟服務。
之後 GET 符合設定的詳情、feedback-options、review-options 時，Uvicorn 日誌會
以 `[violation-response]` 印出路徑、HTTP status 與完成序列化的 JSON body。
不列印請求 headers、Cookie、token 或圖片二進位資料，不改變回應內容。
每筆日誌最多 2 MiB，超出時標記 truncated=true。排查完成將設定留空
並重啟即可關閉；其餘紀錄不會列印。

### 紀錄類別快照

詳情、列表、回饋選項與提交驗證只使用紀錄保存的 `model_classes` 快照。
缺少快照時回傳空類別選項，不套用固定類別表或查詢最新模型。
新上傳必須提供實際執行偵測的 `model_id` 與 `model_version`。
直播 renderer 依 producer 提供的 class metadata 解析類別，未知 ID 保持未知。

### 物件顯示政策：一般畫面與回饋模式

每個 overlay_objects 物件新增四個布林欄位。安全錐的回應範例：

```json
{
  "object_id": "det_0",
  "code": "safety_cone",
  "display_name": "交通錐",
  "color": "#FF5722",
  "show_box": false,
  "show_label": false,
  "feedback_show_box": true,
  "feedback_show_label": true
}
```

此範例僅列出相關欄位，bbox、confidence 等仍保留。
一般詳情／全螢幕依 show_box、show_label 決定是否畫框及名稱；進入
回饋或除錯模式時，改依 feedback_show_box、feedback_show_label。
目前其他物件均為 true；未知類別不因猜測 ID 而隱藏。
新欄位缺失時可視為 true，保留舊後端顯示行為。

Flutter 不應硬編碼 safety_cone 的顯示例外，也不能將隱藏物件從本機
原始資料刪除。這四個欄位只有呈現用途，不代表回饋權限；能否回饋仍須
依既有權限／選項及 POST 驗證。已標記或人工補框的安全錐也使用相同政策。

後端即時串流使用同一顯示政策，只使用 canonical `safety_cone` code。
原始安全錐偵測仍參與分群、警告判斷、保存與回饋，overlay_regions
不受個別物件顯示開關影響。若分群結果空，一般画面只會隱藏安全錐框，
不會自動產生區域；本次沒有修改 HDBSCAN 參數或歷史分群結果。
