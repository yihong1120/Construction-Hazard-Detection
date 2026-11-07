# Visionnaire 主網站開發環境

`https://dev.changdar-server.mooo.com` → B 的獨立 Nginx／HTTPS 站點 →
Tailscale → A 的 dev gateway `18090` → 獨立的前端、BFF 與 CPU API。
AI 開發環境繼續使用 `https://ai-dev.changdar-server.mooo.com`。

## 公司 AI 電腦設定 API

`/bff/ai_access/api/company-ai` 已接到 A 的獨立 company-ai dev 後端 `127.0.0.1:18083`。
沿用本網站的 BFF session，寫入請求需要 `X-CSRF-Token` 與 `X-Company-AI-Action: 1`。
後端驗證 Keycloak issuer、dev audience `visionnaire-dev-api` 及當前 AI client 角色；
登入 dev 本身不代表已取得 AI 權限。BFF 移除瀏覽器傳入的 company-ai 內部 bridge／身分標頭。

兩邊私有設定由 `ai-gateway/deploy/dev/connect_bff.py` 配置；主網站 setup 保留固定的
`BFF_AI_ACCESS_API_URL=http://127.0.0.1:18083`，啟動檢查拒絕使用正式 upstream。
驗證可執行 `python -m deploy.dev.smoke_company_ai`，只建立並撤銷測試 enrollment，
不領取模型 key、不呼叫模型、不變更預算或角色，完成後移除自己的暫時 SSO session。
Windows 設定工具與模型入口仍由 ai-dev 提供，尚未切換到公司 dev 網域。

## 服務與資料

| 項目 | A 位址／路徑 |
| --- | --- |
| 主網站 dev gateway | `127.0.0.1:18090`；Tailscale 同埠僅允許 B |
| BFF／使用者、工地、群組管理 | `127.0.0.1:19005` |
| DocFormify 文件與簽核 API | `127.0.0.1:19004` |
| 違規紀錄 API | `127.0.0.1:19002` |
| 通知收件匣 API | `127.0.0.1:19003` |
| 串流目錄 API | `127.0.0.1:19800` |
| PostgreSQL | 獨立 Docker project `construction-dev`，`127.0.0.1:15433`，DB／role `construction_dev`；pgvector 0.8.1，image 固定 digest |
| Redis | 獨立 container／volume／密碼，`127.0.0.1:16380` |
| 私有設定 | `~/.config/construction-dev`，權限 `0700`，檔案 `0600` |
| 文件／簽名／static | `~/.local/share/construction-dev`，不使用正式 NFS 或 `/mnt/data` |
| 網站靜態成品 | 此 repo 的 `frontend/web`；dev Nginx 直接提供此目錄 |
| Python 後端原始碼 | 此 repo；DocFormify 使用 `~/Documents/DocFormify` |

dev 使用正式資料庫的 schema-only 基準，包含 `public` 與 `construction_hazard_detection`
兩個 schema、63 張表、預設值、identity、約束、索引、函式、trigger、enum、extension，
以及 26 筆 migration 版本／checksum。歷史備份表只有結構，沒有複製正式業務資料。
dev 保留自己的 ChangDar、測試群組、tenant、deployment、文件與測試資料。
功能目錄與 ChangDar 群組授權對齊正式的 `yolo_api`、`file_manage`、`doc_chat`；
`max_allowed_streams` 基準為 99。授權基準儲存在私有設定
`~/.config/construction-dev/authorization-baseline.json`，以功能名稱對應 dev 自己的 ID。
登入共用既有 Keycloak realm，但使用 `visionnaire-web-dev` client、精確的 dev callback、
PKCE S256、獨立 token audience `visionnaire-dev-api` 與 `__Host-vn_dev_session` cookie。
只有已連結到 dev 資料庫的帳號可以登入。dev 登出只清除 dev session，不登出正式 SSO。
既有 `frontend/web` 前端透過同來源 `/bff/auth/oidc/login` 登入，
由 Keycloak 處理登入後回到 dev；重新整理後由 BFF cookie 還原登入狀態。
登出沿用前端要求的相對 `/bff/auth/oidc/logout` bridge，
dev 只消耗自己的單次 state 並回到 dev 登入頁，不執行共用 Keycloak 的全域登出。
dev 不載入 Keycloak 管理憑證；帳號建立、刪除、密碼與狀態等共用身分操作停用。

## 目前功能範圍

可測試登入、工地／群組管理、空白文件清單與簽核、違規資料 API、通知收件匣及串流目錄。
畫面、導覽與預設頁面使用 `frontend/web` 既有成品的行為，沒有另建 dev UI。
可直接開啟 `/files`；網站回應以 `X-Construction-Environment: dev` 標示環境。

物件偵測已接上 A 的獨立 YOLO API（`127.0.0.1:19000`，單一 worker、推論併發 1）。
提供 YOLO26N/S/M/L/X，沿用正式版相同版本、類別與 SHA256 權重；
權重複本保存在 dev 私有目錄，模型清單只授權 dev tenant 的圖片辨識，沒有正式攝影機或工地授權。
模型延遲載入，記憶體最多保留一個模型。服務已設定開機啟動及失敗重啟。
新增環境的模型準備或重新驗證可使用 `python3 -m deploy.dev.prepare_detection`；
此指令只讀取正式模型目錄資料，保留既有 dev 模型修改，不複製業務資料。

**此階段尚未連接主網站聊天服務、攝影機 workers、即時媒體或文件背景轉檔 worker。**
AI 聊天可使用既有 `ai-dev`。主網站上述未連接的 API 明確回傳 503，不轉送正式服務。
FCM 外部推播停用；dev 沒有正式攝影機與文件資料。文件轉檔工作只會排入 dev Redis，
需要後續安裝獨立 worker；不會由正式 Windows／LibreOffice worker 消費。

## A 上啟動與更新

設定與資料均留在 Git 以外。首次準備使用：

```bash
cd "$HOME/Documents/Construction-Hazard-Detection"
"$HOME/.conda/envs/construction_env/bin/python" deploy/dev/setup.py
python3 deploy/dev/build_frontend.py
```

`setup.py` 使用既有私有 Keycloak bootstrap 設定建立它自己管理的 dev client，
並保留既有 dev 密碼與資料。沒有提供刪除或重建 volume 的步驟。
首次建立空資料庫時，需要已驗證的私有 schema snapshot 與授權基準。
`seed.py` 只對完全空白的 dev 資料庫還原 schema baseline；已有部分資料庫結構時拒絕覆蓋，
不再使用 ORM `create_all` 代替正式 schema／migration。
`build_frontend.py` 現在只驗證並登記既有 `frontend/web` 的版本與校驗碼，
不重新編譯、不覆寫此目錄。manifest 儲存在
`~/.local/share/construction-dev/gateway/frontend-manifest.json`，由 `/dev-build.json` 提供。

Python 修改後重啟對應 dev 服務，例如：

```bash
systemctl --user restart construction-dev-management
systemctl --user restart construction-dev-files
```

初始化與更新依序執行「還原空白資料庫基準 → 套用 migration manifest →
驗證 → seed → 啟動服務」。`setup.py` 會先暫停已啟動的 dev API；既有資料庫
只補上待執行遷移，沒有待執行項目時不重複備份或套用。每次遷移前先取得完整備份，
在同一獨立 dev PostgreSQL 的隨機新資料庫實際還原並核對資料筆數及索引，完成後
刪除驗證資料庫。備份與結果留在 `~/.local/share/construction-dev/migrations/setup-*`。
此流程不連線正式資料庫，也不會覆寫 `construction_dev` 來驗證備份。

schema 或 seed 失敗時，setup 不啟動 dev API；修正錯誤後重跑 setup 接續處理。
各服務啟動時另外以唯讀連線檢查 migration checksum、結構條件及待執行項目，
檢查未通過便停止啟動，不在一般服務 lifespan 中執行 DDL。

前端仍在獨立 Flutter source repo 開發；將產出的版本化 artifact 部署到此 repo
的 `frontend/web` 後，執行 `python3 deploy/dev/build_frontend.py` 更新 manifest。
不手動修改編譯後的 JS。dev 不再提供 `visionnaire-flutter-dev/build/web`。
全部 dev units 已啟用；changdar 的 linger 使其在 A 開機後自動啟動。

此次資料庫對齊先在 `15434` 候選容器還原 schema、帶入現有 dev 測試資料，
最後才建立並驗證所有索引與外鍵。完整比對通過後，`activate_database_parity.py`
停止 dev API、核對來源資料指紋、備份並切換資料卷，再重啟 dev API。
驗證失敗會還原原 dev 資料卷。原 `construction-dev_postgres-data` 保留，
目前使用 `construction-dev_postgres-parity-data`；私有備份與狀態在
`~/.local/share/construction-dev/parity`。沒有使用含 DROP TABLE 的 `init.postgres.sql` 更新資料庫。

後續定期或部署前可執行以下只讀比對；正式端連線設定 `default_transaction_read_only=on`：

```bash
python3 -m deploy.dev.database_parity
```

此工具同時檢查兩個 schema、欄位／預設值、約束、索引、trigger、函式、enum、sequence
設定、extension、RLS policy、migration checksum、功能目錄與 ChangDar 群組授權。
結構或基準授權改動會回傳失敗；不自動修改正式端。

## B 上安裝 HTTPS

已將 `gateway.py` 複製為 `~/construction-dev-gateway/deploy.py`。
先驗證 A 的私有連線與 artifact，然後才新增它管理的 dev 站點、申請獨立憑證、
驗證 Nginx 語法、reload 並等待新 workers。失敗時還原原先 dev 站點。
原有三個正式站點與 ai-dev 設定不會被覆寫。

```bash
sudo python3 "$HOME/construction-dev-gateway/deploy.py"
python3 "$HOME/construction-dev-gateway/deploy.py" --verify-only
```

憑證使用 B 現有 Certbot 帳號與排程；新增專用續期 reload hook。
不需要修改 DNS 或路由器的 TCP 80／443 轉發。

## 驗證

```bash
cd "$HOME/Documents/Construction-Hazard-Detection"
python3 -m pytest -q --override-ini='addopts=' deploy/dev/test_deployment.py deploy/dev/test_database_parity.py
python3 -m deploy.dev.smoke
systemctl --user list-units 'construction-dev*'
```

`smoke.py` 需要私有 Keycloak 管理憑證。它以管理 API 建立一次暫時的 ChangDar SSO session，
走實際 OIDC／PKCE callback，測試 dev 查詢、新增與刪除測試工地、CSRF 與正式 API 拒絕 dev token。
結束後只移除這次測試的 session；不保存密碼、token 或瀏覽器 cookie。

`browser_smoke.py` 額外使用 Playwright／Chromium，從公開 HTTPS 文件頁進入登入、
驗證 Keycloak client、實際瀏覽器 callback、文件頁、重新整理與 dev 登出。
同樣只建立並清除一次暫時的 operator SSO session：

```bash
python3 -m deploy.dev.browser_smoke --chrome /path/to/chromium
```

檢查公開 `index.html`、`main.dart.js` 與 `version.json` 的 SHA256，
應與此 repo 的 `frontend/web` 檔案一致。
