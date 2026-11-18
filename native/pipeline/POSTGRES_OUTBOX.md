# PostgreSQL 警報 outbox

Rust pipeline 的警報事件、JPEG、上傳收據與通知重試狀態，統一儲存在 PostgreSQL。
線上寫入、領取、重試和查詢使用 PostgreSQL。升級時可唯讀匯入舊 Rust SQLite 佇列。
違規紀錄透過既有上傳 API 建立，outbox 持久化待完成的派送工作。

## 連線與隔離

可將下列內容合併至既有 Rust JSON 設定：

```json
{
  "outbox": {
    "url_env": "RUST_OUTBOX_DATABASE_URL",
    "namespace": "construction-rust-local"
  }
}
```

`url_env` 是環境變數名稱，不是連線密碼。使用既有本機 PostgreSQL URL；支援 `postgresql://` 與既有的 `postgresql+asyncpg://`。
未設定 `outbox.url_env` 時，優先沿用 `database.url_env`，否則使用 `DATABASE_URL`。
未設定 namespace 時，使用 canonical `state_dir` 的 SHA-256。需要移動狀態目錄或由多個派送程序共享佇列時，明確指定並保持同一 namespace。

啟動時建立獨立的 `rust_pipeline` schema，不修改既有業務表：

| 表 | 用途 |
| --- | --- |
| `outbox_namespaces` | namespace 與待處理 payload 配額 |
| `outbox_events` | JSONB 事件、BYTEA JPEG、派送狀態、上傳收據與租約 |
| `outbox_imports` | 舊 SQLite 的匯入指紋與紀錄 |

DDL 見 [001_outbox.sql](migrations/001_outbox.sql)。首次初始化的連線角色需要建立 schema／表的權限。
每個 pipeline 使用最多 4 個 outbox 連線，共同服務所有攝影機；不是每鏡頭建立一組連線池。

## 寫入與派送

- JPEG 在有界 CPU pool 編碼；PostgreSQL 存取由 SQLx 非同步執行，交易提交後才回報持久化成功。
- 事件與 JPEG 同一交易寫入，主鍵為 `(namespace, event_id)`，重複加入不覆寫既有資料。
- 配額以 namespace 的計數列加鎖更新，避免並行寫入超額；完成派送後釋放相同 payload 額度。
- 使用 `FOR UPDATE SKIP LOCKED` 領取到期事件。每次領取產生新的 token，過期 worker 不能覆寫重新領取後的收據、重試或完成狀態。
- 上傳收據先提交，再發送通知。重啟或重試時，已有收據的事件略過再次上傳。
- 保留原指數退避、永久失敗佇列與每 namespace 最近 10,000 筆已完成紀錄。
- 預設 payload 配額仍為 `delivery.outbox_limit_bytes` 的 512 MiB，最大嘗試次數仍為 20。此額度不等於 PostgreSQL 表、索引與 WAL 的實際磁碟用量。

PostgreSQL 必須可連線才能啟動與保存新事件；沒有自動回退 SQLite。
執行中寫入失敗會回報持久化錯誤與 failure 計數，不視為寫入成功；尚未提交的事件不具持久化保證。
派送器遇到資料庫連線錯誤會等待並重試，已提交的事件與租約可在恢復連線／重啟後繼續處理。
HTTP 端仍需支援既有 `Idempotency-Key`，處理遠端已成功但本地尚未記錄結果的重送窗口。

## 舊 SQLite 遷移

停止舊 pipeline 後，使用新的 `run --config ...`：取得原 state directory 的程序鎖後，自動匯入該目錄的 `outbox.sqlite3`，完成後才啟動派送。
也可以先執行明確的匯入指令：

```bash
native/pipeline/target/release/construction-pipeline outbox \
  --config /path/to/rust.json --migrate-sqlite
native/pipeline/target/release/construction-pipeline outbox \
  --config /path/to/rust.json
native/pipeline/target/release/construction-pipeline outbox \
  --config /path/to/rust.json --requeue-failed
```

- 以 SQLite 唯讀交易取得一致快照，使用有界批次傳送；所有 PostgreSQL 匯入與指紋紀錄在同一交易提交。
- 保留 JPEG、事件、收據、嘗試次數、排程與失敗原因；舊 `inflight` 轉為可恢復的 `pending`。
- 原 SQLite 不刪除、不改寫。相同指紋重複匯入回報 0，不重新加入已完成的工作。
- 任一損毀資料、配額超限或與目標事件衝突都回滾整次匯入，保留來源供處理。
- 匯入後若舊 SQLite 又被改寫，會拒絕繼續自動匯入，要求先協調兩邊進度；不要讓舊版本繼續使用同一佇列。回滾舊程式前也需要協調已派送進度。

只有指定的舊 Rust `state_dir/outbox.sqlite3` 會被匯入，不會掃描或匯入歷史效能測試、Python outbox 或其他程序的資料檔。

## 啟動與驗證

在 repository 根目錄執行。以下設定適用於已有攝影機和模型設定的 Rust JSON。先將 `RUST_OUTBOX_DATABASE_URL` 交由 shell 環境或服務的環境設定提供，再指向已建立的設定檔。

```bash
pipeline=native/pipeline/target/release/construction-pipeline
"$pipeline" validate --config /path/to/rust.json
"$pipeline" outbox --config /path/to/rust.json
"$pipeline" run --config /path/to/rust.json
```

`outbox` 初始化資料表並列出目前各派送狀態的事件數量。`run` 在 `delivery.enabled` 為 `true` 時一併啟動 HTTP 派送。獨立處理既有佇列可用 `dispatch --config /path/to/rust.json`，並沿用同一資料庫及 namespace。上傳及通知服務可使用請求的 `Idempotency-Key` 對重送請求去重。

[獨立流程的測試設定](README.md#驗證) 提供可重跑的本機 PostgreSQL 容器。設定 `CHD_RUST_TEST_DATABASE_URL` 後，可單獨執行儲存與遷移測試。

```bash
cargo test --locked --manifest-path native/pipeline/Cargo.toml --test postgres_outbox
```

測試要求 loopback 位址及 `chd_rust_pipeline_test` 資料庫，涵蓋併行領取、配額競爭、逾期租約、跨重啟收據、namespace 隔離，以及匯入衝突與回滾。實際攝影機負載的效能請以相同來源、模型和派送設定量測。
