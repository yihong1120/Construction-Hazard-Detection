🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# MCP Server

FastMCP server that exposes construction-hazard tools to MCP clients and agent
workflows. It wraps selected project modules and HTTP APIs; it is not part of
the main camera processing loop.

## Tool Groups

- `inference_*`: detect objects in an image.
- `hazard_*`: derive construction-safety warnings from detections.
- `violations_*`: search records, fetch record details, and read images.
- `notify_*`: send LINE, Telegram, or broadcast messages.
- `record_*`: upload or queue violation records.
- `streaming_*`: inspect or control stream-oriented helpers.
- `model_*`: fetch and inspect model files.
- `utils_*`: geometry and detection validation helpers.

## Run

From the repository root:

```bash
python -m examples.mcp_server.main
```

Default transport is `streamable-http`.

```dotenv
MCP_TRANSPORT=streamable-http
MCP_HOST=0.0.0.0
MCP_PORT=8000
```

Supported transports are `stdio`, `sse`, and `streamable-http`.

## Service Settings

Configure only the integrations you need:

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

The MCP server imports `mcp.server.fastmcp.FastMCP` directly from the MCP
Python SDK. Keep the `mcp` package installed through the project dependency
set.

## Notes

- Prefer the root README for production service startup order.
- Tool calls that run YOLO inference may load model dependencies; keep MCP
  inference usage separate from the main multi-camera GPU workflow unless you
  have enough resources.
- Use this server for controlled agent access, diagnostics, and administrative
  workflows, not as the live video transport.
