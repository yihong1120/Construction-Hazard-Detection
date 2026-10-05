# Construction Hazard Detection MCP

[English](README.md) | [繁體中文](README-zh-tw.md)

This optional MCP entry point exposes detection, violation records, models and multi-platform
notifications to AI clients. It uses existing APIs, OIDC identities and site permissions.

## Install and run

From the repository root:

```bash
uv sync --extra mcp
MCP_TRANSPORT=stdio uv run --no-sync python -m examples.mcp_server.main
```

Queries, uploads, FCM, model downloads and geometry tools need no GPU or image packages.
For local inference and hazard analysis, use `uv sync --extra mcp --extra yolo`.
Frame capture requires the `streaming` or `yolo` extra. Heavy libraries load only when used.

```bash
MCP_TRANSPORT=streamable-http uv run --no-sync python -m examples.mcp_server.main
```

HTTP defaults to `http://127.0.0.1:8092/mcp`; SSE is also supported at `/sse`.
Configure `MCP_HOST`, `MCP_PORT`, `MCP_PATH`, `MCP_SSE_PATH` and `MCP_DEBUG` as needed.
An invalid transport falls back to stdio and logs to stderr, keeping protocol stdout clean.

Clients use the server's configured identity. Use stdio or loopback HTTP for trusted
operators. The HTTP transport has no separate user login; remote access requires an
authenticated proxy and restricted network access.

## API configuration

- `VIOLATION_RECORD_API_URL`: defaults to `http://127.0.0.1:8002`.
- `FCM_API_URL`: defaults to `http://127.0.0.1:8003`.
- `WORKER_OIDC_CLIENT_ID`, `WORKER_OIDC_CLIENT_SECRET`, and
  `WORKER_OIDC_TOKEN_ENDPOINT` (or `OIDC_ISSUER_URL`): OIDC client credentials.
  Provision the service account, tenant and site permissions through the existing
  deployment flow; configuring a Keycloak client alone does not grant API access.
- `MCP_STATIC_BEARER`: optional OIDC access token for violation queries only.
  Uploads and FCM still use the worker identity; static tokens require manual renewal.
- `MODEL_FETCH_API_URL`, `MODEL_FETCH_BEARER_TOKEN`: authenticated model download
  settings, as used by `src/model_fetcher.py`.
- `MCP_UPSTREAM_TIMEOUT_SECONDS`, `MCP_REMOTE_IMAGE_TIMEOUT_SECONDS`: default 20.
- `MCP_MAX_REMOTE_IMAGE_BYTES`: inference input/upload limit, default 10 MiB.

Local passwords and unauthenticated queries are unsupported. Network clients reuse
connection pools, which close on shutdown even when the transport task group is cancelled.

## Tools

| Capability | Tools |
| --- | --- |
| Local inference | `inference_detect_frame` |
| Hazard analysis | `hazard_detect_violations` |
| Queries | `violations_search`, `violations_get`, `violations_get_image`, `violations_my_sites` |
| Notifications | `notify_fcm_send`, `notify_line_push`, `notify_messenger_send`, `notify_telegram_send`, `notify_wechat_send`, `notify_broadcast_send` |
| Uploads | `record_send_violation`, `record_batch_send_violations` |
| Frame capture | `streaming_capture_frame` |
| Models | `model_sync`, `model_list_available`, `model_get_local` |
| Geometry/validation | `utils_calculate_polygon_area`, `utils_point_in_polygon`, `utils_bbox_intersection`, `utils_validate_detections` |

Inference accepts `image_base64` and optional `model_key`, `use_ultralytics`,
`movement_thr`. Repeated settings reuse the detector; changed settings close and
replace it. Access to the shared tracker is serialized.

Hazard analysis accepts rows with at least six values `[x1,y1,x2,y2,confidence,class_id]`
or objects with `bbox`/`box`, `confidence`/`conf`, `class_`/`cls`. Per-call
`detection_items` switches are respected; malformed numbers raise errors.

Uploads require `image_base64`, `detections`, `warnings`, `site`, `stream_name`,
`model_id`, `model_version`. Optional fields are ISO `timestamp`, `cone_polygon`
and `pole_polygon`. Detections use the API's tracked rows with at least seven values:
`[x1,y1,x2,y2,confidence,class_id,track_id]`; warnings use the structured hazard output.
The model snapshot must exist in the registry and be accessible to the account.
Batches accept 1–100 objects with those fields and retain individual failures in order.
There are no demo successes, invented site/camera identities or placeholder statistics.

Frame capture returns JPEG `base64` or an `array`. Opening and reading each have
10-second timeouts. Capture runs off the event loop and always releases resources.
Continuous detection belongs to the existing `main.py` worker; the unsupported
stream control tools were removed. Image URLs still require API Bearer authorization;
use `as_base64=true` to retrieve image content through the tool.

Install `uv sync --extra mcp --extra social-notifications` for all notification
channels. Platform SDKs and image libraries load only on related tool calls;
see `.env.example` for credentials. FCM uses the OIDC worker identity; other
platforms use their own tokens/secrets. An unconfigured platform fails when called
without preventing MCP startup. Training, evaluation and augmentation remain
available in their `examples/YOLO_*` directories. Verify changes with `uv run --no-sync pytest tests/examples/mcp_server`.
