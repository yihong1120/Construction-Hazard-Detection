🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# 個人身分的工安唯讀 MCP

本套件維護工地與違規工具的名稱、描述、輸入 schema、資料投影、分頁及公開查詢證據。
Agent／LLM 執行、對話與用量由 `ai-gateway` 維護；gateway 僅使用通用 MCP connector。
工安查詢規則、OIDC 驗證、tenant 與工地權限保留在本 repo。

## 啟動

沿用本 repo 的資料庫、Redis、OIDC 與 deployment 設定，安裝既有 mcp extra：

```bash
uv sync --locked --extra mcp
uv run --no-sync uvicorn examples.mcp_server.agent_reads.app:create_app \
  --factory --host 127.0.0.1 --port 8093
```

本機入口為 `/mcp`。它是獨立、可選的唯讀程序，不要求一般違規 HTTP API 安裝 MCP。
`install(app, path='/mcp')` 也可將入口加入已有 DB／Redis／OIDC lifespan 的 ASGI app。
不在一般服務啟動時自動安裝，也不自動新增公開 Nginx 路由。

正式串接須配置已註冊的公開 HTTPS origin；工安 deployment 依 request 的可信 scheme
與 Host 選擇。反向代理應保留原有可信轉送規則，loopback URL 不能代替公開部署身分。
每個 MCP HTTP 請求都先經既有 `jwt_access`，包含探索與列工具；不接受瀏覽器 session
cookie 代替 Bearer，也不使用 worker token 或 `MCP_STATIC_BEARER` 備援。

## 工具

| 伺服器工具名稱 | 本 repo 的資料來源 |
| --- | --- |
| `construction_my_sites` | `violation_query_service.get_my_sites` |
| `construction_search_violations` | `violation_query_service.get_violations` |
| `construction_get_violation` | `violation_review_service.get_single_violation` |

工具直接呼叫既有領域服務，不經對自己的 HTTP 請求、不搬動 ORM 或複製 ACL。
每次 request 的已驗證身分與 DB session 維持到 SDK response 完成，並行請求不共用身分。
所有工具明確宣告唯讀且非破壞性；只允許公開 schema 的參數，不提供通知、審核、
上傳、圖片下載或即時推論。

搜尋接受真實 `site_id`、關鍵詞、成對且帶時區的起訖時間、1–100 的 limit 及 cursor。
回應保留 `items`、`next_cursor`、`has_more`；單頁筆數不是總數，沒有虛構的 total。
結果僅投影必要的紀錄資訊，移除圖片位址、內部路徑與無關欄位。
`provenance` 由本服務產生，包含來源、查詢時間、篩選、單頁範圍、最多 100 個識別碼
與限制；超過識別碼上限有截斷標記。Gateway 只轉送，不解讀這些領域欄位。

## Gateway 設定

在 gateway 的既有 Company Agent JSON 設定加入以下片段，使用實際註冊的 MCP URL：

```json
{
  "mcp_servers": [
    {
      "name": "safety",
      "url": "https://construction.example/agent-mcp",
      "allowed_tools": [
        "construction_my_sites",
        "construction_search_violations",
        "construction_get_violation"
      ]
    }
  ]
}
```

gateway 以來源描述及 schema 動態建立 `external_safety_*` 工具，不寫死工安欄位。
它會每次取得本人最新有效 OIDC token；該 token 必須符合本服務 issuer、JWKS 與
audience，`(provider, sub)` 必須對應本 tenant 的 active 帳號及當前工地權限。
登入 WebUI 本身不授予工安權限，不以姓名、email 或 LiteLLM key 推測身分。
不修改登入／session 契約或遷移身分資料表。

原本 `examples.mcp_server.main` 的 stdio／8092 worker 入口保留原用途；它使用伺服器
配置的身分，與本套件的個人 OIDC 入口分開，不能作為多人查詢的備援。

## 驗證

```bash
python -m pytest tests/examples/mcp_server/agent_reads_test.py \
  tests/examples/violation_records/agent_read_contract_test.py -q
```

測試使用替身 DB／已驗證身份，實際經 MCP SDK 傳輸驗證探索、schema、工具呼叫、
匿名拒絕、並行身分與權限錯誤；不呼叫真實模型或正式工地資料。
