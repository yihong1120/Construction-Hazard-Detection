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

The canonical service names are `chat`, `db_management`, `detection`, `fcm`,
`files`, `streaming`, `streaming_web`, and `violations`. For example, sites are
listed through `GET /bff/db_management/list_sites`.

For violation list and detail responses, the BFF rewrites `image_url` and
`thumbnail_url` to `/bff/violations/...`. This keeps evidence requests on the
same cookie-authenticated route instead of exposing the violation service's
internal root paths to the browser.

Unsafe requests require an allowed `Origin` and `X-CSRF-Token`.

For OIDC sessions, `POST /bff/auth/logout` returns a one-use
`global_logout_url`. Flutter Web must navigate the top-level browser to that
URL after clearing its UI state. The BFF removes its own cookie first, then
redirects the browser to Keycloak RP-initiated logout using Keycloak's own SSO
cookie. No OAuth token is placed in browser storage, JavaScript, or the logout
URL.

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
