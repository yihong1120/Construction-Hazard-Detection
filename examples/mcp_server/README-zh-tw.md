🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# MCP Server

將工地危害偵測能力提供給 MCP clients 與 agent workflows 的 FastMCP server。它包裝部分
專案模組與 HTTP APIs；不屬於主攝影機處理 loop。

## Tool Groups

- `inference_*`：對圖片執行物件偵測。
- `hazard_*`：由 detections 產生工地安全警示。
- `violations_*`：搜尋紀錄、取得紀錄細節與讀取圖片。
- `notify_*`：發送 LINE、Telegram 或 broadcast 訊息。
- `record_*`：上傳或排隊違規紀錄。
- `streaming_*`：檢查或控制 stream-oriented helpers。
- `model_*`：下載與檢查模型檔。
- `utils_*`：幾何與 detection validation helpers。

## 執行

從 repo 根目錄：

```bash
python -m examples.mcp_server.main
```

預設 transport 是 `streamable-http`。

```dotenv
MCP_TRANSPORT=streamable-http
MCP_HOST=0.0.0.0
MCP_PORT=8000
```

支援 transports：`stdio`、`sse`、`streamable-http`。

## 服務設定

只需設定你會使用的 integration：

```dotenv
VIOLATION_RECORD_API_URL=http://127.0.0.1:8002
VIOLATION_RECORDS_USERNAME=admin
VIOLATION_RECORDS_PASSWORD=password
DETECT_API_URL=http://127.0.0.1:8000
LINE_CHANNEL_ACCESS_TOKEN=...
TELEGRAM_BOT_TOKEN=...
BROADCAST_URL=http://127.0.0.1:8080/broadcast
API_REQUEST_TIMEOUT=30
```

MCP server 會直接 import `mcp.server.fastmcp.FastMCP`。請透過專案 dependency set 安裝
FastMCP package。

## 注意事項

- production service 啟動順序請以根目錄 README 為準。
- 會執行 YOLO inference 的 tool call 可能載入模型依賴；除非資源足夠，否則 MCP inference
  請和主多攝影機 GPU workflow 分開使用。
- 此 server 適合受控 agent access、診斷與管理流程，不作為直播影像 transport。
