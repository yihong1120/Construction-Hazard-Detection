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
  --workers 4
```

Public routes:

- `POST /bff/auth/login`
- `GET /bff/auth/session`
- `GET /bff/auth/csrf`
- `POST /bff/auth/logout`
- `/bff/{allowlisted-service}/*`

Unsafe requests require an allowed `Origin` and `X-CSRF-Token`.

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
