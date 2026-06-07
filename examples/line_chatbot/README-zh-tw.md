🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# LINE Chatbot Example

LINE Messaging API 的小型 FastAPI webhook 範例。它和主違規通知流程分離；正式違規通知
由 `src/notifiers/` 與 `examples/local_notification_server/` 處理。

## 功能

- 在 `POST /webhook` 接收 LINE webhook events。
- 驗證 `X-Line-Signature` header。
- 透過 LINE Messaging API v3 回覆文字訊息。

## 設定

部署前請設定 LINE channel credentials。目前範例在 `line_bot.py` 保留 placeholder，公開
endpoint 前請替換或改為環境變數：

```text
LINE_CHANNEL_ACCESS_TOKEN
LINE_CHANNEL_SECRET
```

## 執行

```bash
uvicorn examples.line_chatbot.line_bot:app \
  --host 127.0.0.1 \
  --port 8000
```

本機測試 webhook 時，請透過 HTTPS tunnel 暴露服務，並在 LINE Developer Console 設定：

```text
https://<public-host>/webhook
```

## 注意事項

- 此範例不要放進高吞吐攝影機主路徑。
- `main.py` 的正式違規 LINE 通知請使用
  `src/notifiers/line_notifier_message_api.py`。
- 公開 webhook 部署需保留 LINE signature verification、TLS 與正常 access logging。
