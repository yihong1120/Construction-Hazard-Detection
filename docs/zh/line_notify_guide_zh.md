[English](../en/line_notify_guide_en.md) | [繁體中文](line_notify_guide_zh.md)

# LINE Messaging API 通知

本 repo 使用 LINE 官方帳號的 Messaging API push message。
舊的 LINE Notify 於 2025-03-31 終止，詳見[官方公告](https://developers.line.biz/en/news/2025/04/01/line-notify/)。

## 設定

1. 準備已啟用 Messaging API 的 LINE 官方帳號。
2. 設定 `LINE_CHANNEL_ACCESS_TOKEN` 為該 channel 的 access token。
3. 取得要通知的 user／group／room ID，並確保官方帳號可向該對象發送訊息。
4. 從 repo 根目錄安裝 `uv sync --extra social-notifications`。

圖片推播透過 Cloudinary 取得可存取的 HTTPS 圖片 URL，另需設定
`CLOUDINARY_CLOUD_NAME`、`CLOUDINARY_API_KEY`、`CLOUDINARY_API_SECRET`。
純文字推播不需這三項設定。

## Python 呼叫

```python
import asyncio
import aiohttp
from src.notifiers.line_notifier_message_api import LineMessenger

async def send():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        sender = LineMessenger(session=session)
        status = await sender.push_message('recipient-id', '工地安全警示')
        print(status)

asyncio.run(send())
```

透過 MCP 使用時，安裝 `uv sync --extra mcp --extra social-notifications`，
再呼叫 `notify_line_push(recipient_id, message, image_base64=None)`。
連線由 MCP lifespan 管理；不需要自行建立 session。

可發送對象、訊息額度及權限以 [LINE Messaging API 文件](https://developers.line.biz/en/docs/messaging-api/sending-messages/)為準。
不要使用舊的 LINE Notify token 或 `notify-api.line.me` 端點。
