🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Authentication And Authorisation

`examples/` 內 FastAPI 服務共用的驗證與授權模組。這裡提供 JWT 驗證、Redis-backed
token/session cache、密碼雜湊、資料庫 session，以及 user/site models。

此目錄是共用 library，不是獨立 application。

## 主要檔案

- `config.py`：從環境變數讀取設定。
- `database.py`：async SQLAlchemy engine、session factory 與 base model。
- `models.py`：唯一的 SQLAlchemy ORM 定義模組，在 `global_lifespan`
  建立 metadata 前載入。
- `jwt_config.py`：JWT access 與 refresh dependencies。
- `cache.py`：Redis user cache、token cache 與輕量 rate helper。
- `redis_pool.py`：HTTP 與 WebSocket handlers 共用的 async Redis pool。
- `user_service.py`：user/site access helper 與 cache invalidation。
- `token_cleanup.py`：清理過期 token cache entries。
- `security.py` 與 `jwt_scheduler.py`：可選的 secret 產生工具。

## 必要設定

```dotenv
DATABASE_URL=postgresql+asyncpg://username:password@127.0.0.1/construction_hazard_detection
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=password
JWT_SECRET_KEY=replace-with-a-long-random-secret
```

同一套部署中的所有 API service 必須使用同一個穩定、高熵的 `JWT_SECRET_KEY`。更換 key
會立刻讓現有 access token 與 refresh token 失效，因此只能作為有計畫的安全操作。

## 密碼

專案使用 `pwdlib[argon2]` 雜湊與驗證使用者密碼。不要儲存明文密碼，也不要用可逆加密
取代 Argon2。

## Redis 用途

Redis 儲存小型驗證與授權狀態：

- user cache entries；
- refresh-token references；
- access-token `jti` lists；
- effective site access cache；
- 共用 cache helpers 使用的 Lua script 狀態。

Redis 不儲存直播影像 frame。

## 被哪些服務使用

- `examples/db_management/`
- `examples/local_notification_server/`
- `examples/streaming_web/`
- `examples/violation_records/`
- `examples/YOLO_server_api/`

請直接啟動上述服務；它們會透過 lifespan 與 dependency wiring 匯入此目錄。
