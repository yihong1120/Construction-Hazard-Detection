🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Database Management Backend

FastAPI service for user, group, feature, site, and stream-configuration
management. `main.py` reads active stream configurations from this service, so
this backend is the control plane for the live detection runtime.

## Responsibilities

- Authenticate users with a username or e-mail plus password, and refresh JWT
  tokens.
- Manage users, pending signups, roles, and groups.
- Manage feature permissions by group.
- Manage construction sites and user-site access.
- Manage stream configurations, including camera URL, stream name, model key,
  detection options, work-hour settings, and live publishing flags.

## Files

- `app.py`: FastAPI application.
- `deps.py`: JWT, role, and site-permission dependencies.
- `routers/`: route modules for auth, users, groups, features, sites, and
  streams.
- `schemas/`: Pydantic request and response models.
- `services/`: async SQLAlchemy service layer.

## Run

From the repository root:

```bash
uvicorn examples.db_management.app:app \
  --host 127.0.0.1 \
  --port 8005 \
  --workers 2 \
  --timeout-graceful-shutdown 10
```

OpenAPI docs are available at `http://127.0.0.1:8005/docs`.

## Required Settings

```dotenv
DATABASE_URL=postgresql+asyncpg://username:password@127.0.0.1/construction_hazard_detection
DB_POOL_SIZE=2
DB_MAX_OVERFLOW=1
DB_POOL_TIMEOUT_SECONDS=10
DB_POOL_RECYCLE_SECONDS=1800
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=password
JWT_SECRET_KEY=replace-with-a-long-random-secret
HCAPTCHA_ENABLED=true
HCAPTCHA_SECRET_KEY=replace-with-your-hcaptcha-secret
HCAPTCHA_SITE_KEY=3e5cc8c8-0e36-4316-8416-63f0e4c635d0
HCAPTCHA_BYPASS_KEY=local-script-only-random-secret
BFF_TOKEN_ENCRYPTION_KEY=replace-with-an-independent-random-secret
BFF_SESSION_COOKIE_NAME=__Host-vn_session
BFF_SESSION_COOKIE_SECURE=true
BFF_SESSION_TTL_SECONDS=2592000
MEDIA_SESSION_TTL_SECONDS=600
PLAYBACK_STREAMING_API_URL=http://127.0.0.1:8800
CORS_ALLOWED_ORIGINS=https://changdar-server.mooo.com,https://visionnaire-cda17.web.app,http://localhost:3000,http://127.0.0.1:3000,http://localhost:5000,http://127.0.0.1:5000,http://localhost:8080,http://127.0.0.1:8080
BREVO_API_KEY=replace-with-your-brevo-api-key
MAIL_FROM=verified-sender@example.com
MAIL_FROM_NAME=Visionnaire
APP_PUBLIC_URL=https://changdar-server.mooo.com
PASSWORD_RESET_TOKEN_TTL_SECONDS=1800
```

The same `JWT_SECRET_KEY` must be used by every API service.
Keep `HCAPTCHA_SECRET_KEY` only in backend environment variables.
Use `HCAPTCHA_BYPASS_KEY` only for trusted backend scripts that cannot solve
hCaptcha. Never expose it to browser code.
Keep `BREVO_API_KEY` only in backend environment variables.
`BFF_TOKEN_ENCRYPTION_KEY` encrypts server-side access/refresh tokens. It must
not be exposed to any client. Flutter Web receives only an opaque HttpOnly
session cookie. Native clients use OAuth Authorization Code + PKCE.
`CORS_ALLOWED_ORIGINS` must list the exact Web origins that may send cookies;
do not use `*` with credentialed requests.
When running with Docker Compose, set `PLAYBACK_STREAMING_API_URL` to
`http://streaming-web-backend:8000`. Use `http://127.0.0.1:8800` only when the
services are running directly on the host.

## Login

`POST /auth/login` uses one account identifier field that accepts a username or
e-mail address:

```json
{
  "identifier": "user@example.com",
  "password": "password",
  "hcaptcha_token": "frontend-hcaptcha-token"
}
```

`POST /auth/google` and `POST /auth/apple` accept provider login tokens and
return the same JWT response shape when the linked local user is active. New
provider accounts are created as pending users and must be approved before
they can receive local JWTs.
If the provider token e-mail already belongs to an existing account, the
backend returns `account_link_required` instead of automatically merging
accounts. The user must sign in with an existing method and link the provider.

Authenticated users can manage login methods with:

- `GET /auth/identities`
- `POST /auth/identities/google/link`
- `POST /auth/identities/apple/link`
- `DELETE /auth/identities/{identity_id}`

Unlinking the last remaining login method is rejected with `last_login_method`.

## Native OAuth and Unified Playback

Flutter Web BFF routes are provided by the `examples/bff` module mounted in
this same process. This service also retains Native OAuth and exposes the
Flutter Web/iOS/Android playback facade.

Native apps use `GET /hazard/api/db_management/oauth/authorize`,
`POST /hazard/api/db_management/oauth/token`,
`GET /hazard/api/db_management/me`, and
`POST /hazard/api/db_management/oauth/revoke`. Only S256 PKCE and configured
client/redirect pairs are
accepted. Access tokens last 15 minutes; refresh tokens rotate and retain
family reuse-detection state.

Native clients use the playback facade under the existing
`/hazard/api/db_management/` base path. Flutter Web uses the authenticated
BFF proxy instead:

```text
Web:    POST   /bff/db_management/api/playback/sessions
Web:    POST   /bff/db_management/api/playback/walls
Web:    POST   /bff/db_management/api/playback/sessions/renew
Web:    DELETE /bff/db_management/api/playback/sessions/{id}

Native: POST   /hazard/api/db_management/api/playback/sessions
Native: POST   /hazard/api/db_management/api/playback/walls
Native: POST   /hazard/api/db_management/api/playback/sessions/renew
Native: DELETE /hazard/api/db_management/api/playback/sessions/{id}
```

Single-camera playback returns `mode: "single"` and `hls_url`.
Multi-camera walls return `mode: "multi_stream"`, `layout: "responsive"`, and
`items[*].preview_hls_url`. `hls_url` and `preview_hls_url` point at stable
playback playlists carrying a short-lived `mt` media token; streaming_web
rewrites the playlist fragment URLs with that same `mt`, so players no longer
need Web-cookie or Native-Bearer-specific HLS handling.

Wall requests may provide only the site or an explicit camera-name list:

```json
{
  "site": "Site A",
  "cameras": ["Cam 1", "Cam 2"],
  "profile": "overlay"
}
```

The endpoint fixes wall quality to a dedicated low-bitrate `preview`
rendition; it is not an alias for the detail HLS path. `profile` controls only
the visual mode: send `"overlay"` for server-rendered detection results or
`"clean"` when the user turns them off. Single-camera sessions always use the
detail rendition.

`POST /api/playback/sessions/renew` accepts `{"id":"..."}`. It extends the
existing media capability TTL in place: `hls_url` and
`items[*].preview_hls_url` do not change, so the client must not rebuild a
player after a successful renewal.

The older Web/Native public media-session APIs have been removed. Flutter no
longer creates a Cookie or Bearer media session before playback. Wall scopes
are limited to 24 unique cameras and never accept wildcards.
Only configurations with `recognition_enabled` enabled are returned to
live-view clients.

```dotenv
GOOGLE_WEB_CLIENT_ID=860473757501-c1gtkrqr4lsa52vgoq7vclprm8atjvtv.apps.googleusercontent.com
GOOGLE_IOS_CLIENT_ID=860473757501-s53qldp7i294qbg1ia8aq822oa0rudj2.apps.googleusercontent.com
GOOGLE_ANDROID_CLIENT_ID=860473757501-088t4flpgv0kdds6pu4a5m1fntamf1ht.apps.googleusercontent.com
APPLE_TEAM_ID=5DU8R27949
APPLE_KEY_ID=NGC4QBS7ZY
APPLE_SERVICE_ID=com.changdar.visionnaire.signin
APPLE_BUNDLE_ID=com.changdar.visionnaire
APPLE_REDIRECT_URI=https://changdar-server.mooo.com/hazard/api/db_management/auth/apple/callback
APPLE_PRIVATE_KEY_PATH=config/secrets/apple/AuthKey_NGC4QBS7ZY.p8
```

## Password Reset

`POST /password/forgot` always returns:

```json
{
  "message": "If the email exists, a reset link has been sent."
}
```

If the e-mail belongs to a user, the backend stores only a SHA-256 hash of a
one-time reset token in Redis with `PASSWORD_RESET_TOKEN_TTL_SECONDS`, then
sends `APP_PUBLIC_URL/reset_password?token={raw_reset_token}` through Brevo.

`POST /password/reset` accepts the raw token from the URL and a new password.
The token is deleted after a successful reset, and the user's cached JWT
session data is removed from Redis.

## Stream Configuration And Runtime

Stream rows with recognition enabled during their configured working hours
drive the main detection workflow:

```text
database stream_configs -> main.py -> src/stream_processor.py
```

Key fields include:

- source URL and stream display name;
- site label and stream ID;
- `model_key`, which maps to `models/pt/best_<model_key>.pt`;
- `recognition_enabled`, which saves a camera configuration without starting
  capture, inference, or violation processing when disabled;
- detection item switches and warning thresholds;
- working-hour schedule;
- clean and annotated MediaMTX publishing options.

For multi-camera deployments, prefer database mode over local JSON config so
updates can be applied without editing `main.py`.
