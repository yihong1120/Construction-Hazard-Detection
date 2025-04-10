
🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# 違規紀錄管理範例

本範例示範如何使用 FastAPI、SQLAlchemy 及相關技術，建立一個完整的違規紀錄管理系統。它展示了：

- 針對違規紀錄的 CRUD 操作。
- 安全的影像處理與儲存。
- 基於 JWT 的使用者驗證與工地（site）存取控制。
- 多語言關鍵字搜尋與同義詞擴充。
- 使用 Pydantic 結構化的 API 回應與請求模式。

## 目錄

1. [概述](#概述)
2. [先決條件](#先決條件)
3. [目錄結構](#目錄結構)
4. [安裝](#安裝)
5. [設定](#設定)
6. [執行範例](#執行範例)
7. [API 端點](#api-端點)
8. [補充說明](#補充說明)

## 概述

此違規紀錄管理伺服器提供強大的 API，用於處理記錄於各工地的違規事項。它具備：

- 能夠擷取使用者可存取的工地清單。
- 提供分頁及依關鍵字、時間範圍、工地等條件進行違規紀錄的篩選。
- 檢視特定違規紀錄的詳細資訊，包括中繼資料與影像。
- 上傳新違規紀錄功能，可支援完整的中繼資料。

## 先決條件

1. **Python 3.9+** – 建議使用以支援 FastAPI、非同步程式及型別標註。
2. **Redis** – 視您的專案設計需求而定（若使用本範例的 Redis 套件則需要）。
3. **SQLAlchemy 與 非同步資料庫**（如 PostgreSQL 或 MySQL） – 用於儲存違規資料。
4. **FastAPI 與其相依套件** – 用於建立與處理 API。

## 目錄結構

以下為 `examples/violation_records` 資料夾的主要結構：

```
examples/violation_records/
├── app.py
├── routers.py
├── schemas.py
├── search_utils.py
├── violation_manager.py
├── static/ (執行時自動建立，用於存放影像)
└── README.md
```

- **`app.py`**：主要的 FastAPI 應用程式進入點。
- **`routers.py`**：定義違規紀錄相關的 FastAPI 路由。
- **`schemas.py`**：使用 Pydantic 定義的請求與回應模式。
- **`search_utils.py`**：提供關鍵字搜尋功能，包含同義詞擴充。
- **`violation_manager.py`**：管理違規紀錄與影像的儲存至本機及資料庫。
- **`static/`**：用於存放上傳的違規影像（自動建立）。

## 安裝

1. **取得本專案**：透過克隆此儲存庫或將 `examples/violation_records` 資料夾複製到您的專案中。
2. **建立並啟用** Python 虛擬環境（可選但強烈建議）：

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # 對於 Linux/macOS
   venv\Scripts\activate     # 對於 Windows
   ```

3. **安裝必要套件**：

   ```bash
   pip install fastapi sqlalchemy databases[asyncpg] fastapi_jwt pydantic aiofiles uvicorn
   ```

## 設定

- **資料庫 & JWT**：請在您的環境變數或設定檔中配置好資料庫連線，以及 JWT 設定（如密鑰、有效時長等），依您專案需求而定。
- **影像與 static 資料夾**：預設影像會存於 `static/` 中。若在正式環境，可以考慮使用雲端儲存或專屬 CDN 服務。

## 執行範例

若要使用 Uvicorn 在本機運行伺服器：

```bash
python examples/violation_records/app.py
```

預設會在 `0.0.0.0:8081` 提供 FastAPI 服務。可自行調整主機位址及埠號。

## API 端點

### 1. 取得可存取的工地

```
GET /my_sites
```
- 回傳使用者目前可存取的工地清單。

### 2. 取得違規紀錄

```
GET /violations
```
- 可依 `site_id`、`keyword`、`start_time`、`end_time` 進行篩選，並支援 `limit`/`offset` 分頁。

### 3. 取得單一違規紀錄

```
GET /violations/{violation_id}
```
- 取得特定違規紀錄的詳細資訊，包括中繼資料與影像路徑等。

### 4. 取得違規影像

```
GET /get_violation_image?image_path=...
```
- 傳回對應違規紀錄的影像。路徑需相對於 `static/` 資料夾。

### 5. 上傳新違規紀錄

```
POST /upload
```
- 建立一筆新的違規紀錄，包含中繼資料及影像檔，上傳至 `static/` 中。

## 補充說明

- **安全性**：請確保 JWT 權杖能被正確驗證，同時每個端點皆有存取控制（例如：使用者只能存取其所被授權的工地）。
- **效能**：可在資料庫中為常用的查詢欄位建立索引。若大量影像須儲存，建議改用雲端儲存服務或 CDN 。
- **多語言搜尋**：在 `search_utils.py` 可擴充更多語言或更複雜的同義詞組；請依您的專案需求調整。
- **進一步擴充**：
  - 在 `violation_manager.py` 中新增更多欄位或管理邏輯（例如：版本化影像）。
  - 改進分頁策略或自訂排序方式。
  - 當有新違規建立時，透過 webhook 或通知將訊息整合至外部系統。
