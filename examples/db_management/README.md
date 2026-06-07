🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Database Management Backend

FastAPI service for user, group, feature, site, and stream-configuration
management. `main.py` reads active stream configurations from this service, so
this backend is the control plane for the live detection runtime.

## Responsibilities

- Authenticate users and refresh JWT tokens.
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
  --workers 2
```

OpenAPI docs are available at `http://127.0.0.1:8005/docs`.

## Required Settings

```dotenv
DATABASE_URL=postgresql+asyncpg://username:password@127.0.0.1/construction_hazard_detection
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=password
JWT_SECRET_KEY=replace-with-a-long-random-secret
```

The same `JWT_SECRET_KEY` must be used by every API service.

## Stream Configuration And Runtime

Active stream rows drive the main detection workflow:

```text
database stream_configs -> main.py -> src/stream_processor.py
```

Key fields include:

- source URL and stream display name;
- site label and stream ID;
- `model_key`, which maps to `models/pt/best_<model_key>.pt`;
- detection item switches and warning thresholds;
- working-hour schedule;
- clean and annotated MediaMTX publishing options.

For multi-camera deployments, prefer database mode over local JSON config so
updates can be applied without editing `main.py`.
