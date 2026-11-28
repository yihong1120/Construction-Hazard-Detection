🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# Web Backend-for-Frontend

The Web BFF is a separate code module mounted by
`examples.db_management.app` in the shared FastAPI process on port `8005`.
Nginx exposes it at `/bff/`. It owns the opaque Web session cookie, CSRF
checks, JWT refresh and the allowlisted HTTP proxy to backend services.

Run the shared process with:

```bash
uvicorn examples.db_management.app:app \
  --host 127.0.0.1 \
  --port 8005 \
  --workers 4 \
  --timeout-graceful-shutdown 10
```

Public routes:

- `GET /bff/auth/oidc/login?return_to=/...`
- `GET /bff/auth/oidc/callback` (Keycloak callback only)
- `GET /bff/auth/oidc/logout?state=...` (one-use global logout bridge)
- `GET /bff/auth/account` (Keycloak Account Console)
- `GET /bff/auth/session`
- `GET /bff/auth/csrf`
- `POST /bff/auth/logout`
- `/bff/{allowlisted-service}/*`

The main application's public login page at `https://<app-host>/login` redirects
to `/bff/auth/oidc/login`. Keycloak then handles credentials, hCaptcha and any
configured multi-factor authentication.

The canonical service names are `chat`, `db_management`, `detection`, `fcm`,
`files`, `streaming`, `streaming_web`, and `violations`. For example, sites are
listed through `GET /bff/db_management/list_sites`.

`GET /bff/auth/session` verifies the current access token and resolves the local
identity, profile and feature grants again. Navigation therefore reflects grants
and revocations on page refresh, including sessions created before a permission
change. Login-time Redis metadata is not the authorisation source. Session
summary reads never rewrite encrypted tokens held by a concurrent refresh.

For violation list and detail responses, the BFF rewrites `image_url` and
`thumbnail_url` to `/bff/violations/...`. This keeps evidence requests on the
same cookie-authenticated route instead of exposing the violation service's
internal root paths to the browser.

Unsafe requests require an allowed `Origin` and `X-CSRF-Token`.

## Keycloak and OpenID Connect single sign-on

Web clients start the authorisation-code flow by navigating to a local return
path, for example `GET /bff/auth/oidc/login?return_to=/violations`. The BFF
creates a single-use `state` and an S256 PKCE challenge, and encrypts the
Keycloak access and refresh tokens in Redis. The browser receives an opaque
HttpOnly session cookie. The callback checks the access token's signature,
issuer and `OIDC_AUDIENCE`, then maps `sub` to the local
`user_identities(provider=keycloak)` record. The local tenant, role, group,
site and feature grants determine access.

The BFF requests a regular refresh token and omits `offline_access`. Refreshing
extends the same online SSO session shared by Visionnaire and the Keycloak
Account Console. When the idle or maximum SSO lifetime expires, both require
the user to sign in again.

Use a dedicated Keycloak audience such as `visionnaire-api` for
`OIDC_AUDIENCE`. Open WebUI and Visionnaire use separate clients. Open WebUI
tokens must exclude the Visionnaire audience. Branded Google and Apple
buttons link to `/bff/auth/oidc/login?return_to=/&idp_hint=google` or use
`idp_hint=apple`. The BFF forwards only allowlisted provider hints. Third-party
tokens stay with Keycloak. The account screen links password changes to
`/bff/auth/account`. Local password login, direct social login and the local
refresh endpoint have been retired. Keycloak's login theme and server-side
provider handle hCaptcha.

For OIDC sessions, the CSRF-protected `POST /bff/auth/logout` returns a one-use
`global_logout_url`. Flutter Web must navigate the top-level browser to that
URL after clearing its UI state. The BFF removes its own cookie first, then
redirects the browser to Keycloak RP-initiated logout using Keycloak's own SSO
cookie. No OAuth token is placed in browser storage, JavaScript, or the logout
URL. Flutter Web uses the returned URL for top-level navigation. Constructing a
Keycloak URL on the client or following the redirect in a background request
does not complete the same browser logout flow.

## Push-device registration

After a successful BFF login, Flutter Web obtains the CSRF token from
`GET /bff/auth/csrf` and registers its Firebase device token through:

```text
PUT /bff/fcm/devices
X-CSRF-Token: <csrf-token>
```

The BFF injects the server-side bearer token. Browser code must not access or
send an access token itself. The JSON body must contain only `device_token`,
`device_lang`, and `platform` (`web`).

Live playback does not use a dedicated BFF playback service. Flutter Web uses
the generic, allow-listed `db_management` proxy, while Native clients call the
same facade through the public API root:

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

Web clients keep renewal requests on the BFF path even if a Native-form
`renew_endpoint` appears in a response.

Keeping BFF code in this package preserves a clear security boundary without
adding another process, database pool or deployment unit.

OIDC token exchange classifies only an explicit `invalid_grant` rejection with
HTTP 400/401/403 as a permanent grant failure. Throttling remains HTTP 429;
transport, server, malformed-response and client-configuration failures return
503. Temporary failures preserve the session and propagate `Retry-After` when
present. Token persistence still checks refresh ownership and session revision.
