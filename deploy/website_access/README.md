# 網站 IP 存取管理

此功能與工地 `Site` 分開管理。全站管理員可管理 `main`、`ai`、`wiki`
的 IPv4／IPv6 地址與 CIDR，普通使用者與群組管理員不可變更全站網路邊界。
`GET /admin/capabilities` 的 `can_manage_website_access` 控制前端管理選單。

所有 Web 呼叫經既有 BFF session 與 CSRF 保護：
`/bff/db_management/admin/website-access`。原生 API 使用既有 OIDC Bearer
驗證。不可把 API bearer token 存進瀏覽器。

Flutter 原生 App 在 B 使用
`https://changdar-server.mooo.com/hazard/api/db_management/admin/website-access`
作為 API 基底路徑，並在每個請求帶 `Authorization: Bearer <access_token>`。
`GET` 基底路徑的 JSON 根節點是**陣列**，恰有 `main`、`ai`、`wiki` 三筆；
每筆含 `site_key`、`hostname`、`restricted`、`version`、`updated_at`、
`rules`。不要把它解析成 `{ "policies": [...] }`。`GET /context` 回傳
`{ "client_ip": "...", "can_manage_website_access": true }`；
`GET /gateway` 回傳 `online`、`checked_at`、`versions`。若 API 回 200 而
畫面仍說沒有網站政策，先記錄狀態碼與 JSON 形狀，再檢查 App 的解析模型。

| 方法 | API 路徑（相對於 `/admin/website-access`） | 用途 |
| --- | --- | --- |
| GET | 空路徑 | 列出三個網站、規則及版本 |
| GET | `/context` | 顯示服務實際看到的來源 IP |
| GET | `/gateway` | 檢查 Nginx 同步程序及已套用版本 |
| PUT | `/{site_key}` | 開啟或關閉限制 |
| POST | `/{site_key}/rules` | 新增 IP／網段與名稱 |
| PUT | `/{site_key}/rules/{uuid}` | 編輯 IP／名稱／啟用狀態 |
| DELETE | `/{site_key}/rules/{uuid}?expected_version=N` | 刪除規則 |
| GET | `/audit?site_key=ai&limit=50` | 查看變更紀錄（最多 200 筆） |

新增／編輯使用完整內容，例如：

```json
{"network":"60.249.248.158","label":"公司出口／VPN","enabled":true,"expected_version":0}
```

規則回應會將單一 IP 正規化為 `/32` 或 `/128`。每次操作回傳完整網站政策與
新的 `version`；下次變更帶入 `expected_version`，版本衝突回 409，須重讀。
開啟限制使用 `{"restricted":true,"expected_version":1}`。
新政策預設 unrestricted，新增 IP 不會自動開啟限制。最多 256 筆／網站。

啟用、修改、停用或刪除規則後，若目前管理員的來源 IP 不在有效白名單中，
交易回滾並回 `website_access_would_lock_you_out`。禁止空白名單與 `/0`；
完全開放請關閉 restricted。所有變更與 before/after 稽核紀錄在同一交易提交。
來源 IP 取自 ASGI client，不接受表單或任意 forwarded header 指定來源 IP。

防止自己被鎖出的檢查只在儲存設定時執行，不會替管理員增加白名單或豁免。
限制啟用且入口已套用後，所有角色都遵守相同規則；已登入的全站管理員若改用
白名單外的網路，其新 HTTP 請求同樣回 403，既有登入 session 不會繞過限制。

## 入口執行機制

`snapshot.py` 以應用程式帳號在 PostgreSQL READ ONLY／REPEATABLE READ 交易中
匯出完整政策。獨立 root 同步程序只接受三個已知網站、布林值及 numeric CIDR，
不將名稱或其他資料庫文字寫入 Nginx。每 5 秒更新 `geo` 白名單；通過 `nginx -t`
後才 reload，失敗保留上次成功設定。API 儲存成功與 Nginx 套用成功分開呈現，
前端比較 `/gateway` 的 `versions[site_key]` 與政策 `version`，且 `online=true`
才顯示「已套用」。資料庫或同步程序中斷時，Nginx 繼續執行上次成功的規則。

HTTPS server 層使用 `if ($vn_website_<site>_allowed = 0) { return 403; }`，
因此靜態檔案、API、WebSocket 握手及已有媒體 `auth_request` 的路由都受限制。
HTTP 僅保留 ACME 驗證及 HTTPS 轉址。規則變更會影響新 HTTP 請求／握手，
不會主動斷開已建立的 WebSocket 或串流連線。

Nginx 使用真實 `$remote_addr`，應用程式只信任本機入口代理。若日後增加 CDN
或其他反向代理，必須先配置限定可信來源的 real-IP 規則，不可直接信任客戶端
提供的 `X-Forwarded-For`。沒有 VPN 出口轉送時，入口看到的是使用者的公網 IP；
經公司 Exit Node 上網時才會看到公司公網 IP。

三個網站仍須通過原本 OIDC 登入與業務權限。AI、Wiki 登入依賴主網站上的
Keycloak，因此啟用 main 限制時須確認所有需要登入的人員及後端服務可從允許
的出口存取主網站。IPv4／IPv6 分開比對；應測試兩種連線。

## 部署

1. 在 dev 資料庫先執行 `deploy/website_access/migration.sql`。
   不在 API startup 自動執行 DDL。新資料庫也必須執行此 migration 來建立 catalog。
2. 更新管理 API 程式並重啟。正式與 dev 使用各自資料庫。
3. 在 repo root 執行 `python -m deploy.website_access.package <輸出目錄>`
   產生不含憑證的部署包。傳到 B 的 `~/website-access-management/bundle/` 後，
   先執行 `python3 ~/website-access-management/bundle/deploy.py --check`，
   再執行 `sudo python3 ~/website-access-management/bundle/deploy.py`。
   部署以 B 的既有 phase5 production 設定為基礎；安裝前備份
   runtime code、Nginx 設定及新功能設定。所有既有政策須為 unrestricted，
   入口啟用後再由管理員透過 API 加入白名單、開啟限制。
4. 本機開發可用 `WEBSITE_ACCESS_HOSTS_JSON` 明確指定 dev 的三個 hostname；
   同步程序 hosts 必須與管理 API 完全相同，禁止 dev 政策套用到正式入口。

同步程序狀態檔預設 `/run/website-access/status.json`，管理 API 只讀取狀態。
需要緊急解除限制時，在 B 的終端機先停止 `website-access-agent.service`，
再將 `/etc/nginx/conf.d/website-access.conf` 的三個 geo 改為 `default 1;` 且移除
其 CIDR 行，執行 `nginx -t` 與 reload。接著修正資料庫政策後再啟動同步程序。
