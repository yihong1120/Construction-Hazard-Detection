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

- `POST /bff/auth/login`
- `GET /bff/auth/session`
- `GET /bff/auth/csrf`
- `POST /bff/auth/logout`
- `/bff/{allowlisted-service}/*`

The canonical service names are `chat`, `db_management`, `detection`, `fcm`,
`files`, `streaming`, `streaming_web`, and `violations`. For example, sites are
listed through `GET /bff/db_management/list_sites`.

Unsafe requests require an allowed `Origin` and `X-CSRF-Token`.

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

Live playback is no longer exposed through BFF media-session endpoints.
Flutter Web and Native clients use the db_management playback facade:

```text
POST   /hazard/api/db_management/api/playback/sessions
POST   /hazard/api/db_management/api/playback/walls
POST   /hazard/api/db_management/api/playback/sessions/renew
DELETE /hazard/api/db_management/api/playback/sessions/{id}
```

Keeping BFF code in this package preserves a clear security boundary without
adding another process, database pool or deployment unit.
