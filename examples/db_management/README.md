🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Database Management Backend

FastAPI service for user, group, feature, site, and stream-configuration
management. `main.py` reads active stream configurations from this service, so
this backend is the control plane for the live detection runtime.

## Responsibilities

- Verify OIDC access tokens and manage Keycloak identities.
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
OIDC_ENABLED=true
OIDC_ISSUER_URL=https://sso.example.com/realms/visionnaire
OIDC_JWKS_URL=https://sso.example.com/realms/visionnaire/protocol/openid-connect/certs
OIDC_AUDIENCE=visionnaire-api
OIDC_ACCOUNT_URL=https://sso.example.com/realms/visionnaire/account
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
```

API services share the configured OIDC issuer, JWKS URL and API audience.

## Login

Web uses Keycloak Authorization Code + PKCE through `GET /bff/auth/oidc/login`.
Native clients use Keycloak Authorization Code + PKCE directly. APIs accept
OIDC access tokens only. Administrator account creation, password resets and
identity updates use the Keycloak Admin API. Users manage their passwords and
account recovery through `/bff/auth/account`.

Local login, direct Google/Apple login, local JWT/refresh issuance and the
password migration bridge have been removed. Existing databases need
`scripts/migrate_remove_local_passwords.sql`.

## Native OAuth and Unified Playback

Flutter Web BFF routes are provided by the `examples/bff` module mounted in
this same process. This service also retains Native OAuth and exposes the
Flutter Web/iOS/Android playback facade.

Native apps use Keycloak Authorization Code + PKCE and `GET /me` for local business permissions.

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

## Retired routes

Stream configuration reads use `GET /sites/{site_id}/stream-config`.
`/list_stream_configs`, `/auth/verify-email`, and `/auth/resend-verification`
are removed. Email verification and mail delivery belong to the configured
Keycloak realm; configure its SMTP and verification policy before enabling
that workflow. Retiring local verification never activates existing accounts.
