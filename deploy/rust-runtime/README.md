🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# Construction Rust runtime preview

This separate release bundles the camera pipeline and the preview HTTP API. Both launch as native Rust executables. Use an isolated database, Redis instance and identity realm to validate the preview. PostgreSQL, Redis, Keycloak, MediaMTX and the reverse proxy remain infrastructure services.

| Component | Preview coverage |
| --- | --- |
| `construction-pipeline` | CPU replay or TensorRT inference, tracking, hazard analysis, overlays, stream publishing and PostgreSQL outbox |
| `construction-api` | Health and readiness, authenticated upload and evidence access, site discovery, WebRTC ICE configuration, live metadata over SSE and WebSocket, opt-in media authorisation and optional TensorRT inference over HTTP and WebSocket |
| Opt-in ports | OIDC BFF, single-camera and wall playback, scoped account and site management, notification centre and FCM, PostgreSQL violation history and review |

The dedicated systemd examples start these two executables directly. The base preview serves an isolated schema and uses separate ports, users and paths. The opt-in ports can be validated against a restored copy of the application schema. Offline development and host administration tools are outside the preview runtime.

The executables use Rust source from `native/` and link to native SDK libraries. `native/postprocess-core/` contains the shared tracking and hazard algorithms. `native/stream-contract/` shares the Redis metadata keys, identifier encoding and overlay palette. PyO3 compatibility wrappers for `main.py` and the legacy services live in `bindings/python/` and are separate builds.

| Python entry point | Rust implementation in this release |
| --- | --- |
| `main.py` and the camera modules under `src/` | `native/pipeline/` supervises cameras from JSON or the PostgreSQL catalogue, applies each camera's working hours, and records events in the PostgreSQL outbox. |
| `examples/streaming_web/` and `examples/db_management/routers/playback.py` | `native/api/src/streaming/` serves site discovery, ICE, metadata, HLS sessions, viewer leases and the Flutter playback API. |
| `examples/db_management/` | `native/api/src/accounts.rs` lists accounts, groups and sites in the operator's group and manages site membership. |
| `examples/bff/` | `native/api/src/bff/` handles OIDC sign-in, session refresh, CSRF, sign-out and forwarding to configured loopback services. |
| `examples/local_notification_server/` | `native/api/src/notifications/` handles encrypted device registration, the inbox, site preferences and FCM HTTP v1 delivery. |
| `examples/YOLO_server_api/` | `native/api/src/detection/` serves configured engine models and image inference with bounded admission. |
| `examples/violation_records/` | `native/api/src/violations/` uses the existing history and review tables. The base preview evidence API uses its isolated schema. |

## Build a release

Use Linux, Rust 1.93.1, a C compiler, `pkg-config` and FFmpeg development libraries. Install `libavformat-dev`, `libavcodec-dev`, `libavutil-dev` and `libswscale-dev` on Debian or Ubuntu. Match the major version of the development and runtime libraries. Set `FFMPEG_ROOT` for a non-system SDK. Install Noto fonts when rendering multilingual overlays.

The pipeline discovers source dimensions in the process through libavformat. A source is probed only when its dimensions are missing from the configuration and catalogue cache.

From the repository root, choose a new release directory.

```bash
deploy/rust-runtime/build.sh /tmp/construction-rust-preview-cpu
```

Use the same command with `--tensorrt` before the output directory to build for GPU inference. Set `TRT_INCLUDE`, `TRT_LIB`, `CUDA_INCLUDE` and `CUDA_LIB` to the compatible SDK paths. Set `OPENCV_ROOT` for OpenCV outside the system directories. Make the dynamic libraries and their dependencies available during linking and at runtime. For an SDK outside the system library directories, set `LD_LIBRARY_PATH` in the build environment and in the service environment file. Use an existing engine that matches the host GPU and TensorRT version. The CPU build can run the hazard analyser or replay detections.

```bash
deploy/rust-runtime/build.sh --tensorrt /tmp/construction-rust-preview-gpu
```

The new directory contains `bin/`, the API configuration example under `config/`, the pipeline guide under `docs/pipeline/` and the sample units under `systemd/`. Cargo builds both binaries with their lockfiles before the script publishes the directory.

## Run the API on loopback

Copy `config/api.example.json` to a private configuration file. Keep the preview listen address on loopback. Set the issuer, JWKS URL and accepted audience for the isolated realm. Edit the subject (`sub`), client ID (`azp`), permissions and sites for each approved principal. Configure `RUST_API_DATABASE_URL` and `RUST_API_REDIS_URL` in the process environment. The schema initialisation command requires the database URL and a role that can create the preview schema.

```bash
release=/tmp/construction-rust-preview-cpu
api_config=/path/to/private-api.json
"$release/bin/construction-api" --config "$api_config" --init-db
"$release/bin/construction-api" --config "$api_config"
```

From another shell, check the endpoints below. Evidence endpoints require a signed access token with the configured audience and grant.

```bash
curl --fail http://127.0.0.1:18800/health
curl --fail http://127.0.0.1:18800/ready
```

To watch live metadata, grant `stream_read` for the camera's site. `GET /labels` lists the sites in the configured grant. Set the API's `metadata_prefix` to the same value as the pipeline's `redis.prefix`. Use the site name and the URL-safe Base64 camera name with its padding. `/metadata/stream-id/{site}/{stream_id}` uses SSE and `/ws/metadata-id/{site}/{stream_id}` uses WebSocket. Viewers share one Redis reader per camera, and the API reuses its existing Redis connection for subscription checks. Slow clients receive the latest frame.

Flutter selects the `overlay` profile and a language. PostgreSQL-managed cameras publish video with annotations drawn by the backend. Enable `capture.host_yuv: true` with CPU decoding to draw directly into the encoder's YUV planes. The Rust playback facade supports up to 32 cameras on a wall.

```bash
curl --no-buffer --fail -H "Authorization: Bearer $PREVIEW_ACCESS_TOKEN" \
  'http://127.0.0.1:18800/metadata/stream-id/demo/Y2FtZXJhLTE='
```

For `GET /webrtc/ice-servers`, set `STREAMING_WEBRTC_STUN_URLS` to a comma-separated list of STUN URLs. To enable time-limited TURN credentials, also set `STREAMING_WEBRTC_TURN_URLS`, `STREAMING_WEBRTC_TURN_SHARED_SECRET` and `STREAMING_WEBRTC_TURN_TTL_SECONDS`. This endpoint also requires `stream_read`.

WebSocket upgrades accept an `Authorization: Bearer` header. Browser clients can use the `token` query parameter. Configure any reverse proxy and access logs to omit the query string for these routes. `WS_MAX_SESSION_SECONDS` controls the maximum WebSocket duration and defaults to 1800 seconds. The server revalidates active sessions.

### Validate the migrated APIs

Start with `config/api.migration.example.json` in the release. Replace its placeholders with an isolated Keycloak realm and an active tenant and deployment from the restored PostgreSQL schema. Identity links use `user_identities.provider_user_id`. Keep the role and site in PostgreSQL in scope as well as the corresponding static API grants. Initialise `rust_api` once with `--init-db`. That command creates the preview evidence schema. The migration profile expects the existing application tables and does not apply application schema migrations.

- Enable `account_management` with `deployment_scope` for group-scoped account and site lists and site membership updates. Use `account_read` for lists and `account_manage` for membership writes.
- Register the BFF callback at `/bff/auth/oidc/callback` in Keycloak. The public API base URL must match the deployment row. Route `/bff/` to the Rust API. Set the BFF's client secret and a separate session encryption secret of at least 32 bytes through the named environment variables. Point each enabled upstream at the Rust listener. A browser session forwards its signed access token to the domain handler.
- Set `media_auth` and `playback`. Route the public session playlist path to the Rust API and media paths through the media proxy and its `/media-auth` subrequest. Match `playback.demand_prefix` and `metadata_prefix` to the pipeline Redis prefix. Camera listings, new playback and active metadata re-check PostgreSQL membership when `deployment_scope` is set. Existing media capabilities have a bounded lifetime and renew through the authorised session.
- Set `RUST_API_DATABASE_SCHEMA` for deployments where application tables live outside `public`. The API uses this schema before `public` in the PostgreSQL search path. The database role must be able to read both schemas as required by the enabled routes.
- Set `violation_history` to enable history, analytics, evidence images, feedback and audit-backed reviews. Set `write_uploads` to `true` to write the Rust multipart `/upload` to the deployment's `violations` table. Provide a writable absolute `media_root` for this mode and allow the API service account to write it through systemd. Use the existing `evidence_write_operations` journal and published or archived model catalogue. The camera worker needs `evidence_write` and an active, linked PostgreSQL account with site and model access.
- Set `notifications.token_key_env` to the existing Fernet key environment variable. To enable sends, add `"fcm": {"service_account_path_env": "GOOGLE_APPLICATION_CREDENTIALS"}` under `notifications`. Point that variable at an isolated Firebase service-account file. The inbox and device routes use `notification_read`. [Firebase HTTP v1 documentation](https://firebase.google.com/docs/cloud-messaging/send/v1-api)

Account creation, approval, password resets, role changes, site and group lifecycle operations and model management still use the existing services. The isolated migration profile exercises the APIs above.

### Send a violation notification

Add `notification_dispatch` to the camera worker's `grants[].permissions` in the API configuration and include the target site in `grants[].sites`. The worker's Bearer token must match the grant's `subject` (`sub`) and `client_id` (`azp`).

Send `POST /send_fcm_notification` as JSON. Supply the configured `site_id` and the `violation_id` returned by a successful `POST /upload` for that site. Both IDs must be positive JSON integers. The `site` name must match the site's current PostgreSQL record and the caller's grant. Repeat the same IDs when retrying so the API can use its PostgreSQL delivery receipts to skip sends already accepted by FCM.

```bash
curl --fail -H "Authorization: Bearer $PREVIEW_ACCESS_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"site":"demo","site_id":1,"violation_id":123,"stream_name":"Camera 1",
       "type":"violation","title":"Safety violation alert",
       "body":{"warning_no_hardhat":{"count":1}},
       "deep_link":"/violations?violation_id=123","metadata":{"violation_id":123}}' \
  http://127.0.0.1:18800/send_fcm_notification
```

Use `POST /notifications/test` with a `notification_read` access token to check delivery to the signed-in user's registered devices. This request has an empty body.

The camera dispatcher expects a positive JSON integer in the upload response's `violation_id` field. If notification delivery is enabled, the notification response must include an `outcome` of `completed`, `skipped`, `retryable` or `permanent_failure`.

### Optional TensorRT image API

With the `--tensorrt` release, add a `detection` object to the API configuration. Choose the CUDA `device`, optional `default_model_id` and `models`. Each entry specifies `id`, `display_name`, `version`, the absolute `.engine` path in `engine`, its `sha256`, `imgsz`, `classes` and `allowed_subjects`. A class contains `id`, `code`, `display_name` and a `#RRGGBB` `color`. The token subject needs an `image_detection` grant and a matching entry in that model's subject allowlist. With a configured model, the preview serves `GET /models`, `POST /detect`, `POST /v2/detect` and `/ws/detect`.

WebSocket clients select the model with `X-Model-Key` and optionally `X-Model-Version`, or send `{"model_key":"preview-model"}` as their first text message. After the server sends a message with `"status":"ready"`, send a JPEG or PNG as a binary message for each inference request. Both transports use the same bounded inference queue. The camera pipeline has its own model worker.

The preview `/v2/detect` route supports the object view and expects `view=objects` and `compute_regions=false`. Supply a JPEG or PNG with its configured model ID.

```bash
curl --fail -H "Authorization: Bearer $PREVIEW_ACCESS_TOKEN" \
  -F 'image=@/path/to/test.jpg' -F 'model=preview-model' \
  -F 'view=objects' -F 'compute_regions=false' \
  http://127.0.0.1:18800/v2/detect
```

## Run the camera pipeline

Set up a standalone camera JSON configuration using `docs/pipeline/README.md` in the release or `native/pipeline/README.md` in the checkout. Use a separate absolute `state_dir` and an isolated PostgreSQL database. Provide its connection through `RUST_OUTBOX_DATABASE_URL` when that variable is selected by `outbox.url_env`. Go through `validate` and the outbox check before starting capture.

```bash
release=/tmp/construction-rust-preview-gpu
pipeline_config=/path/to/private-pipeline.json
"$release/bin/construction-pipeline" validate --config "$pipeline_config"
"$release/bin/construction-pipeline" outbox --config "$pipeline_config"
"$release/bin/construction-pipeline" run --config "$pipeline_config"
```

For an isolated evidence delivery test, point `delivery.upload_url` at the Rust `/upload` endpoint, configure `token_url` and the credential environment names, and grant the worker `evidence_write` for its sites. To test the notification path with matching history IDs, use `violation_history.write_uploads` and set `delivery.notification_url` to `/send_fcm_notification` on the Rust API. Without a notification URL, the pipeline completes the outbox item after upload.

## Systemd example

The units expect the selected release at `/opt/construction-rust-preview/current`, the service accounts `chd-rust-preview-api` and `chd-rust-preview-pipeline`, and private configuration and environment files under `/etc/construction-rust-preview/`. Initialise the database schema once before starting the API unit. Keep the camera state and media output under `/var/lib/construction-rust-preview-pipeline` and make models readable by the appropriate service accounts. For a TensorRT build, give the accounts access to the required GPU devices and native libraries. Use the isolated units to validate the release before any production routing changes.
