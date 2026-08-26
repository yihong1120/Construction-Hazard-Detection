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
- `search_utils.py`：多語 search helpers。
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

```dotenv
VIOLATION_RECORD_API_URL=http://127.0.0.1:8002
DATABASE_URL=postgresql+asyncpg://username:password@127.0.0.1/construction_hazard_detection
JWT_SECRET_KEY=replace-with-a-long-random-secret
```

## Endpoints

- `GET /my_sites`：登入使用者可見的 sites。
- `GET /violations/filter-options?site_id=...`：回傳登入者在所選工地可用的
  鏡頭與固定違規項目；省略 `site_id` 時，回傳所有授權工地的鏡頭。已部署的 Web
  用戶端也可相容使用 `GET /filter-options`。`stream_id` 是鏡頭設定的數字 ID，
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
