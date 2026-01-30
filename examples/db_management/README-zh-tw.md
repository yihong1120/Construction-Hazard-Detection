
🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# 資料庫與使用者管理後端（Database Management Backend）

此目錄包含使用FastAPI構建的資料庫與使用者管理後端服務，涵蓋用戶、群組、網站、串流設定與權限功能，並透過JWT進行身份驗證與授權。

以下提供`examples/db_management`目錄內檔案結構、使用說明，以及生產環境部署建議與最佳實踐。

## 檔案結構（File Structure）

```
examples/db_management/
├── __init__.py
├── app.py
├── deps.py
├── routers/
│   ├── auth.py
│   ├── users.py
│   ├── groups.py
│   ├── sites.py
│   ├── features.py
│   └── streams.py
├── schemas/
│   ├── auth.py
│   ├── user.py
│   ├── group.py
│   ├── site.py
│   ├── feature.py
│   └── stream_config.py
└── services/
    ├── auth_service.py
    ├── user_service.py
    ├── group_service.py
    ├── site_service.py
    ├── feature_service.py
    └── stream_config_service.py
```

---

## 核心元件說明

### `app.py`

* **用途**：FastAPI應用程式主要入口，負責建立路由、中介軟體、CORS設定，並整合JWT身份驗證。
* **特色**：

  * 設定與註冊各管理模組的API路由（使用者、群組、網站等）。
  * 管理應用程式啟動及關閉時的任務。

**啟動方式**：

```bash
uvicorn examples.db_management.app:app --host 127.0.0.1 --port 8000
```

---

### `deps.py`

* **用途**：定義跨API端點的共用身份驗證與權限檢查函數。
* **主要函數**：

  * `get_current_user`：驗證JWT並取得當前使用者資訊。
  * `require_admin`：檢查使用者是否具備管理員權限。
  * `require_super_admin`：檢查使用者是否為超級管理員。
  * `_site_permission`：確認使用者對指定網站或群組的權限。

---

### `routers/`

定義不同功能的API端點：

* **`auth.py`**：

  * 使用者登入、登出及Token刷新端點。

* **`users.py`**：

  * 使用者建立、刪除、密碼更新、狀態變更與角色管理。

* **`groups.py`**：

  * 群組建立、更新、刪除與群組功能權限管理。

* **`sites.py`**：

  * 網站建立、修改、刪除及網站與使用者的關聯管理。

* **`features.py`**：

  * 功能開關的建立與管理，並將功能指定給特定群組。

* **`streams.py`**：

  * 串流設定的建立與管理，包含群組串流數量限制。

---

### `schemas/`

使用Pydantic模型進行資料驗證：

* `auth.py`：JWT登入、登出、刷新Token的資料結構。
* `user.py`：使用者管理相關的資料模型。
* `group.py`：群組相關的資料模型。
* `site.py`：網站相關的資料模型。
* `feature.py`：功能開關管理相關的資料模型。
* `stream_config.py`：串流設定相關的資料模型。

---

### `services/`

包含與資料庫互動的邏輯，使用非同步的SQLAlchemy ORM：

* `auth_service.py`：JWT認證邏輯、登入/登出處理及Redis快取。
* `user_service.py`：使用者帳號的CRUD操作。
* `group_service.py`：群組管理的CRUD操作。
* `site_service.py`：網站管理與相關資料清理操作。
* `feature_service.py`：功能開關管理及與群組的功能綁定操作。
* `stream_config_service.py`：串流設定管理及串流數量限制操作。

---

## 使用說明

### 1. 安裝依賴套件

```bash
pip install fastapi uvicorn sqlalchemy aiomysql redis pydantic PyJWT python-multipart
```

### 2. 資料庫與Redis啟動

* 啟動MySQL/MariaDB：

  ```bash
  docker run -d --name mysql-db -e MYSQL_ROOT_PASSWORD=root -e MYSQL_DATABASE=db_management -p 3306:3306 mysql:8
  ```

* 啟動Redis：

  ```bash
  docker run -d --name redis -p 6379:6379 redis
  ```

### 3. 配置環境變數 (`.env`)

```
DATABASE_URL=mysql+aiomysql://user:password@localhost/db_management
REDIS_URL=redis://localhost
JWT_SECRET_KEY=your_jwt_secret_key_here
```

### 4. 啟動FastAPI服務

```bash
uvicorn examples.db_management.app:app --host 127.0.0.1 --port 8000
```

瀏覽 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 查看API文件。

---

## 測試

單元測試位於專案頂層的 `tests/` 資料夾中：

```bash
pytest --cov=examples.db_management --cov-report=term-missing
```

測試內容包含：

* 身份驗證流程
* 使用者、群組、網站與功能CRUD操作
* 權限與角色驗證

---

## 生產環境部署注意事項

1. **安全性**

   * 設定明確的CORS來源限制。
   * JWT密鑰應嚴格保密，定期輪換。

2. **性能與擴展**

   * 使用Gunicorn與Uvicorn提升併發效能。
   * 必要時考慮Redis集群部署。

3. **資料庫管理**

   * 定期資料庫備份。
   * 監控並優化資料庫效能。

4. **安全連線**

   * 使用Nginx反向代理並配置SSL/TLS。

5. **日誌與監控**

   * 實施完善的日誌紀錄。
   * 採用如Prometheus、Grafana進行監控與警示。

---

## 最佳實踐建議

* 嚴格以Pydantic驗證資料輸入。
* 定期審核並更新依賴套件，避免安全漏洞。
* 完善的日誌與異常處理，以利問題追蹤。

---

## 聯絡與更多資訊

若需更多身份驗證與使用者管理相關資訊、進階功能或部署諮詢，請參考專案其他資料夾，或直接與開發團隊聯絡。
