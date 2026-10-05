[English](line_notify_guide_en.md) | [繁體中文](../zh/line_notify_guide_zh.md)

# LINE Messaging API notifications

This repository sends push messages from a LINE Official Account through Messaging API.
LINE Notify ended on 2025-03-31; see the [official notice](https://developers.line.biz/en/news/2025/04/01/line-notify/).

## Configuration

1. Prepare a LINE Official Account with Messaging API enabled.
2. Set `LINE_CHANNEL_ACCESS_TOKEN` to its channel access token.
3. Obtain the target user/group/room ID and ensure the account can message that recipient.
4. Install `uv sync --extra social-notifications` from the repository root.

Image messages use Cloudinary for a publicly accessible HTTPS image URL. Configure
`CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY` and `CLOUDINARY_API_SECRET` for images.
Text-only messages do not need Cloudinary credentials.

## Python usage

```python
import asyncio
import aiohttp
from src.notifiers.line_notifier_message_api import LineMessenger

async def send():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        sender = LineMessenger(session=session)
        status = await sender.push_message('recipient-id', 'Construction safety alert')
        print(status)

asyncio.run(send())
```

For MCP, install `uv sync --extra mcp --extra social-notifications` and call
`notify_line_push(recipient_id, message, image_base64=None)`. MCP owns the session lifecycle.

Recipient eligibility, quotas and permissions follow the [official Messaging API documentation](https://developers.line.biz/en/docs/messaging-api/sending-messages/).
Use Messaging API credentials rather than LINE Notify tokens or `notify-api.line.me` endpoints.
