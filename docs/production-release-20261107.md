# 正式後端部署紀錄：2026-11-07

已於台北時間 **21:23** 部署至 `changdar-server.mooo.com` 的正式機 B。切換控制程序耗時約 8 秒；各 API 服務依序恢復。

## 版本與範圍

- Release：`20261107-714745553e63`
- 來源：當時工作目錄的 423 個部署檔案快照，逐檔驗證 SHA256，並非新的 Git commit。
- 來源識別：`714745553e634f807e548c6e4e2ec8ae0975ac0b3fb39c161aa8ed2664188383`
- 正式版本目錄：`/home/changdar02/backend-releases/20261107-714745553e63`
- 獨立 Python 3.15.0 runtime，套件由鎖檔及 hash 安裝；152 個 distributions 的相依檢查通過。
- 保留正式 TensorRT 11.1.0.106 與既有 INT8 engines，使用 NVIDIA `v11.1` 原始碼編譯 CPython 3.15 binding。

更新以下六個 systemd 服務的 `ExecStart`，保留原有服務使用者、其他 drop-ins、設定檔、連接埠與資料目錄：

| 服務 | 角色 | Loopback port |
| --- | --- | --- |
| `phase5-bff-production` | BFF／管理 API | 8006 |
| `phase6-notifications` | 通知 API | 8003 |
| `phase6-streaming` | 播放 API | 8800 |
| `phase6-violations` | 違規紀錄 API | 8002 |
| `phase7-yolo-api` | 推論 API | 8000 |
| `phase7-cameras` | 攝影機處理與推論 workers | — |

原 Python 3.14 runtimes 與應用目錄保留。DocFormify、MediaMTX、Keycloak、PostgreSQL、Redis、nginx、AI 與 wiki 未進行版本切換；DocFormify 與 MediaMTX 的主程序 PID 未改變。

## 資料庫

目標為正式資料庫 `construction_hazard_detection`。既有 migration checksum 無漂移；本次補上：

1. `20261101_01_model_publications`
2. `20261101_02_notification_delivery`
3. `20261102_01_model_versions`
4. `20261102_02_account_evidence_lifecycle`
5. `20261103_01_account_state_evidence_recovery`
6. `20261103_02_sync_progress_retention`

完成後 manifest 全部 10 項均通過 postcondition，30 個 ORM 資料表無缺少欄位。

正式庫先以同一 exported snapshot 產生完整 custom-format 備份。首次隔離還原因大型向量索引超過工具的 15 分鐘預算而中止；之後使用原備份，在獨立 PostgreSQL cluster 完成完整還原，確認 `pg_restore` 成功結束、66 個資料表筆數符合備份 COPY 紀錄、所有索引有效。隨後在同一隔離資料庫演練全部待執行遷移，確認重跑冪等性及既有資料表筆數不變。隔離資料庫已刪除，驗證 cluster 已停止。

- 備份大小：`6,866,780,514` bytes。
- 備份 SHA256：`06e962b3a1d5c476897ac78ef70bb33665d8600f4e41de61d4f212014037738e`。
- 正式機私有備份：版本目錄下 `operations/database-backup/database.dump`。
- 還原、演練與正式遷移結果保存在同一 `operations/` 目錄；版本來源及 runtime 紀錄位於 `release-manifest.json`、`runtime-provenance.json`。

## 前端相容性與設定

初次後端部署時未更換正式 Flutter artifact，當時 `main.dart.js` SHA256 前後一致：
`6c6a1d0d1d975c86d80eedadf6b2664e6ec50599c9349e0fa4feddbb529f08b8`。

初次部署為舊前端暫留的 `GET /list_stream_configs?site_id=...` 相容層，已於同日新版前端上線後移除。正式 `release_apps.py` 已刪除臨時處理器、路由註冊及專用 imports；現在僅使用 `GET /sites/{site_id}/stream-config`。請依 [Flutter 後端交接文件](flutter-backend-handoff.md) 使用 canonical routes。

移除前確認正式前端已引用 `/sites/{site_id}/stream-config`，且不再引用 `/list_stream_configs`。新版 `main.dart.js` SHA256：
`ab2e5ac338487366ac90d123ef93b08e753d9c15a3c4252f9104f216042f53bc`。

變更先經隔離 API 驗證，再僅重啟 `phase5-bff-production`；未執行資料庫遷移。正式站的舊路由回傳 404，正式路由與 BFF session 在未登入時維持 401。部署 manifest 的相容別名清單已清空，控制程式 hash 與變更紀錄同步更新，操作紀錄位於 `operations/compat-removal-20261107/`。

保留所有原有 FPS 設定：來源、clean 與發布為既有的 30 FPS 設定，預覽 15 FPS；未額外降低來源 FPS。登入與加密設定沿用正式設定；現有 session 為 `online_sso_v1`，閒置 TTL 為 604800 秒。

## 驗證結果與觀察

- 先前來源測試：2283 passed、146 subtests passed，coverage 96.72%；本次快照逐檔 hash 已驗證。
- 五個 API 的隔離 OpenAPI 與未登入授權檢查通過。
- 五款既有 TensorRT engines 均在新 runtime 完成單幀推論。
- 正式模型列表與有限影像推論使用既有 worker client 認證，均回傳 200；無效影像回傳 400。
- 初次部署時正式首頁、AI 與 wiki 健康檢查回傳 200；未登入的 BFF session、管理及相容路由回傳 401。後續移除相容層後，舊路由改為 404。
- API 回應可由 `X-Construction-Backend-Release: 20261107-714745553e63` 辨識版本。
- 六個服務啟動後均 active/running、未發生自動重啟；初步觀察無 runtime traceback，持續產生 worker throughput 紀錄。
- 觀察到攝影機來源讀取逾時，部署前已有同類紀錄；不能據此宣稱所有外部攝影機連線健康。
- 未以真實終端使用者執行帳號刪除、變更權限或發送通知；未完成完整 Flutter 人工操作驗收。

## 程式回切入口

正式機保留原 systemd 定義與 drop-ins，可使用以下入口回切六個服務的程式：

```sh
/home/changdar02/backend-releases/20261107-714745553e63/venv/bin/python \
  /home/changdar02/backend-releases/20261107-714745553e63/operations/activate_release.py \
  --rollback
```

需以正式機具 systemd 管理權限的帳號執行。程式回切會保留新增的資料庫結構及正式資料，不會還原備份覆蓋線上資料。日後回切前應檢查 `account_operations`、`evidence_write_operations` 等新版持久化工作是否待處理，安排其完成或保留處理能力。
