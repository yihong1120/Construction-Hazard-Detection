🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# LINE Chatbot Example

Small FastAPI webhook example for the LINE Messaging API. It is independent of
the main violation-notification flow, which is handled by `src/notifiers/` and
`examples/local_notification_server/`.

## What It Does

- Receives LINE webhook events at `POST /webhook`.
- Verifies the `X-Line-Signature` header.
- Replies to text messages through the LINE Messaging API v3.

## Configure

Set the LINE channel credentials before deployment. The current example keeps
placeholders in `line_bot.py`, so replace them or wire them to environment
variables before exposing the endpoint publicly:

```text
LINE_CHANNEL_ACCESS_TOKEN
LINE_CHANNEL_SECRET
```

## Run

```bash
uvicorn examples.line_chatbot.line_bot:app \
  --host 127.0.0.1 \
  --port 8000
```

For local webhook testing, expose the server through an HTTPS tunnel and set
the LINE Developer Console webhook URL to:

```text
https://<public-host>/webhook
```

## Notes

- Keep this example out of the high-throughput camera path.
- Use `src/notifiers/line_notifier_message_api.py` for production violation
  alerts from `main.py`.
- Protect public webhook deployments with LINE signature verification, TLS, and
  normal access logging.
