🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# 違規紀錄後端

用於儲存、查詢與提供工地危害違規紀錄的 FastAPI 服務。當串流處理程序確認警示事件後，
`src/violation_sender.py` 會將違規圖片與中繼資料上傳到這裡。

## 職責

- 將違規中繼資料儲存在 PostgreSQL。
- 將上傳的違規圖片存到受控的 `static/` 目錄。
- 驗證 JWT，並依工地權限控管存取。
- 提供可篩選、分頁的違規紀錄查詢。
- 透過安全的相對路徑提供違規圖片。
- 以同義詞擴充查詢，支援多語關鍵字搜尋。

## 檔案

- `app.py`：FastAPI 應用程式。
- `routers.py`：紀錄清單、詳情、上傳、工地及影像路由。
- `schemas.py`：Pydantic 請求與回應模型。
- `violation_manager.py`：資料庫與影像儲存邏輯。
- `path_utils.py`：驗證靜態檔案路徑的安全性。
- `search_utils.py`：多語詞典搜尋，不需下載 NLP 模型。
- `violation_analytics.py`：聚合 SQL 與圖表回應組裝。
- `settings.py`：靜態檔案目錄設定。

## 執行

```bash
uvicorn examples.violation_records.app:app \
  --host 127.0.0.1 \
  --port 8002 \
  --workers 2
```

直接執行模組的 `main()` 時，預設連接埠為 `8081`。專案執行環境使用 `8002`。

## 必要設定

升級 PostgreSQL 不會自動補上專案新增的索引。既有資料庫應由管理員使用本機維護的
索引遷移腳本增補索引；遷移腳本與量測、調校紀錄由管理員另行維護。
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

## API 端點

- `GET /my_sites`：登入使用者可見的工地。
- `GET /violations/filter-options?site_id=...`：回傳登入者在所選工地可用的
  鏡頭與固定違規項目；省略 `site_id` 時，回傳所有授權工地的鏡頭。
  `stream_id` 使用鏡頭設定的數字 ID。
- `GET /violations`：可篩選、分頁的違規紀錄清單。支援選用的
  `site_id`、`stream_id`、`violation_type`、時間範圍；審核者可加 `flagged=true`
  與選用的 `review_status=pending|resolved|dismissed`。
- `GET /violations/analytics`：違規趨勢、排行與時段統計；僅 `admin` 與
  `super_admin` 可存取，支援與紀錄頁相同的 `site_id`、`stream_id`、
  `violation_type`、`start`、`end`、`bucket` 條件，且所有聚合都會套用；
  其他角色一律回傳 `403 violation_analytics_forbidden`。
- `GET /violations/{violation_id}`：單筆違規細節。
- `POST /violations/{violation_id}/feedback`：送出誤判、漏判、類別錯誤或 bbox
  錯誤的結構化回饋，預設進入待審核狀態。
- `PATCH /violations/{violation_id}/review`：管理員或全站管理員可更新已標記紀錄的審核狀態。
- `GET /get_violation_image?image_path=...`：從 `static/` 回傳圖片。
- `POST /upload`：串流處理程序上傳圖片與中繼資料。

## 保留策略

若是未另外封存的本機硬碟部署，影像檔與資料庫紀錄應使用相同保留週期。目前建議
保留 18 個月，除非法律或合約要求不同期限。

刪除舊資料時，請同時移除資料庫紀錄與對應檔案。避免 `static/` 留下無法追溯的影像，或紀錄指向已移除的檔案。

## 儲存注意事項

- `static/` 目錄存放營運證據。
- MediaMTX 管理 HLS 與 WebRTC 的即時影音片段，保留期限應比證據影像短。
- 如果磁碟空間接近上限，請降低保留期限或先匯出紀錄再刪除。

證據接收會在串流讀取時累計大小，並於發布前在有界工作池驗證影像格式、尺寸
及可解碼性。`VIOLATION_IMAGE_MAX_UPLOAD_BYTES` 預設 20 MiB，
`VIOLATION_IMAGE_MAX_PIXELS` 預設 67,108,864 像素；超限回傳 413，無效影像
回傳 400，無效中繼資料回傳 422。工作池最多兩個執行緒、八個已接納工作，
額度用盡回傳 503 `IMAGE_WORK_BUSY`。warnings、偵測列及多邊形統一驗證，
內部直接傳遞已驗證物件。

影像接受 PNG 與 JPEG，與受保護的原圖讀取入口一致。影像保留原始位元組，
新檔名依實際格式選擇副檔名；路徑改變時，先更新復原
紀錄再發布檔案。既有證據路徑仍可讀取，不需要 SQL 遷移。歷史偵測會保留
原始列索引，跳過無效列不再改變 `det_N`，讓既有回饋維持對應。

縮圖以暫存檔及原子替換發布，同一來源的並行請求共用工作。快取鍵保留來源
副檔名，例如 `image.png.jpg`，避免不同格式撞名；舊快取不再使用，可依既有
快取維護方式清理。證據解碼與縮圖共用有界影像工作池，應用關閉時會等待回收。

## 回饋與審核選項

選項由本服務依已驗證的帳號、工地權限及指定紀錄產生：

- `GET /violations/{violation_id}/feedback-options?locale=zh-TW`
- `GET /violations/{violation_id}/review-options?locale=zh-TW`

原生端使用 `Authorization: Bearer <access_token>`。Nginx 對外提供的
以此部署範例來說，違規服務的基底網址是 `https://app.example.org/hazard/api/violations/`，
因此回饋選項完整 URL 為
`https://app.example.org/hazard/api/violations/violations/{violation_id}/feedback-options?locale=zh-TW`。
前一個 `violations` 是反向代理前綴，後一個是本服務既有資源路徑。

回應維持 `schema_version: 1`、`revision`、`options: [{id, label}]`、
`default_id: null`、`policy: {}`，並設定 `Cache-Control: private, no-store`。
回饋類別取自原始快照。若無快照，`options` 為空陣列。審核選項取決於操作者角色及目前審核狀態。
提交仍使用既有回饋或審核端點，並重新驗證。

管理服務的 `violation_feedback_labels`／`violation_review_actions` 範圍
已移除並回傳 404；前端須直接使用上述違規端點。
部署後需重啟違規服務；程式修改不會自動更新執行中的路由。

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

`corrected_label` 必須使用 feedback-options 提供的穩定識別碼；
`corrected_bbox` 延續既有 `[x1, y1, x2, y2]` 座標契約。
人工回饋不要求、不寫入，也不回傳模型 ID／版本。
舊版用戶端額外傳送的 `model_version` 不會參與驗證或保存。
`original_label` 僅保留用戶端回報的原始標籤文字，不代表模型識別。

前端按鈕根據選項載入狀態及可用類別判斷；紀錄快照決定可用回饋類別。
選項與提交都檢查紀錄權限；提交時重驗類別與座標。
模型推論的版本檢查及舊紀錄的中繼資料不受此人工標註流程影響。
不需資料庫遷移或回填歷史資料；既有回饋的舊版本欄位保留，
新回饋不填寫該欄位。

### 類別名稱與顏色

`/models` 的 `classes`、違規詳情與列表的 `class_metadata`，以及
回饋端點的 `options` 共用類別識別碼與 `#RRGGBB` 顏色。
建立紀錄時將目錄類別保存為 `model_classes`。對外的 `class_metadata` 包含
該快照的名稱、數字 ID、識別碼與顏色。有效顏色標準化為大寫，
缺失或無效色由後端以穩定識別碼補色，不影響名稱或回饋驗證。

新模型調色不修改歷史快照；禁止以最新目錄回填未知來源的舊紀錄。
回饋 POST 不需 `color` 或模型識別，伺服器在鎖定紀錄後重新驗證快照中的類別識別碼。
可保留原始影像供應用程式繪製疊圖；串流伺服器繪圖也使用推論模型的中繼資料。
本次沒有新增 API 路徑或資料表欄位，既有的 `/client-options` 仍已停用。

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

`class_colors` 包含後端完整配色。若該筆快照有有效顏色，會優先使用相同類別的快照色。
詳情 `overlay_objects[].color` 為已解析的框線顏色，可直接繪製；標記狀態
不會改變框色。自行繪製 `detection_items` 時，先查 `class_metadata` 中數字 ID 對應的識別碼
與顏色，再查後端的 `class_colors`。無法辨識的 ID 使用 `unknown_class_color`。
區域邊界採用對應的多邊形顏色。

Flutter 使用後端調色盤。後端顏色資料缺失或無效時，
顯示配色載入失敗／重試，保留原圖，暫停相應疊圖。若舊版用戶端使用固定色或依雜湊值產生顏色，
須同步修改應用程式，使配色以後端回應為準。
回饋選項和提交共用同一份已解析類別表，仍須通過紀錄權限檢查。
已燒錄進 JPEG 的舊框色無法透過中繼資料改變。

### 後端分群與歷史區域繪圖（overlay schema 1）

串流的 `DangerDetector` 執行 HDBSCAN，分群區域同時用於危險判定與即時繪圖。
同一次 `detect_danger` 回傳的警示與交通錐、電桿區域多邊形隨證據影像一起
寫入違規紀錄。區域偵測仍受串流功能開關控制，不因前端開啟疊圖而啟用。
上傳的多邊形使用原圖像素的 `[[[x,y],...],...]` JSON 字串；後端驗證點數、
數值有限性與非零面積，不接受 NaN 或退化邊界。`null` 表示未提供，`[]` 表示
保存的結果為空，不代表一定已啟用／執行分群。

`GET /violations/{id}` 新增以下欄位，不改動原有回應結構：

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

範例中的座標僅示意。讀取時只將已保存邊界除以原圖寬高，絕不呼叫分群、
讀目前攝影機配置或替歷史資料製造區域。區域識別碼僅在該筆紀錄內有效。區域座標可超出 0～1（例如緩衝區超出圖片），
前端應裁切至圖片範圍，逐點限制座標數值會改變多邊形形狀。

`region_status` 分別回傳每種區域的狀態。

- available：有可繪製的已保存邊界。
- empty：已保存空陣列；無法只憑此狀態區分未啟用或未形成有效群聚。
- not_recorded：沒有保存資料，不表示當時沒有群聚。
- invalid：舊資料無法解析或邊界不合法；其他證據仍可讀。
- image_unavailable：有邊界但無法取得原圖尺寸，不能安全換算。

`overlay_objects` 中的物件提供 `code` 與 `display_name`；有快照時直接顯示
`display_name` 並使用 `color`。`label` 保留既有值，以相容舊版用戶端。無法解析的 ID 顯示為灰色的 `class-N`。
不使用目前模型猜測歷史數字 ID。類別解析與區域
幾何的讀取獨立。

Flutter 整合：列表是精簡摘要，進詳情須 GET 詳情；先畫區域填色／邊界，
再畫物件框及名稱。使用 `BoxFit` 後的 `imageRect`。
`px = imageRect.left + x * imageRect.width`、
`py = imageRect.top + y * imageRect.height`。
框的寬高（w/h）分別乘以顯示寬高，不能把 w/h 當 x2/y2，也不能再除原圖尺寸一次。
若缺原圖尺寸，`image_size` 為 `null`，不提供無法換算的像素區域。

保留 `cone_polygons` 與 `pole_polygons` 的原始字串供舊版用戶端讀取，新版用戶端僅繪製
解析完成的 `overlay_regions`，避免兩條資料流重複畫同一區域。
本次沿用資料庫已有的區域欄位，不需資料庫遷移，不回填未知歷史類別。

### 列印前端實際收到的 JSON

設定 `VIOLATION_DEBUG_RESPONSE_IDS=*` 可列印所有紀錄；也可指定
`155157,155158` 限定紀錄。修改後重啟服務。
之後 GET 符合設定的詳情、feedback-options、review-options 時，Uvicorn 日誌會
以 `[violation-response]` 印出路徑、HTTP 狀態碼與序列化的 JSON 本文。
略過請求標頭、Cookie、權杖或圖片二進位資料，不改變回應內容。
每筆日誌最多 2 MiB，超出時標記 `truncated=true`。排查完成將設定留空
並重啟即可關閉，其餘紀錄不會列印。

### 紀錄類別快照

詳情、列表、回饋選項與提交驗證只使用紀錄保存的 `model_classes` 快照。
缺少快照時回傳空類別選項，不套用固定類別表或查詢最新模型。
新上傳必須提供實際執行偵測的 `model_id` 與 `model_version`。
即時影像繪製程序依來源提供的類別中繼資料解析類別，未知 ID 保持未知。

### 一般畫面與回饋模式的物件顯示規則

`overlay_objects` 中的每個物件帶有四個布林欄位。安全錐的回應範例：

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

後端即時串流使用同一顯示政策，只依正式識別碼 `safety_cone` 判定類別。
原始安全錐偵測仍參與分群、警告判斷、儲存與回饋，`overlay_regions`
不受個別物件顯示開關影響。若分群結果空，一般畫面只會隱藏安全錐框，
不會自動產生區域；本次沒有修改 HDBSCAN 參數或歷史分群結果。
