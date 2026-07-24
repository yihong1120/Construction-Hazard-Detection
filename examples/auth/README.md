🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Authentication And Authorisation

Shared authentication modules used by the FastAPI services in `examples/`.
They provide JWT validation, Redis-backed token/session cache, password
hashing, database sessions, and user/site models.

This directory is a shared library for the example services rather than a
standalone application.

## Main Files

- `config.py`: environment-backed settings.
- `database.py`: async SQLAlchemy engine, session factory, and base model.
- `models.py`: users, groups, sites, stream configs, features, and violations.
- `jwt_config.py`: JWT access and refresh dependencies.
- `cache.py`: Redis user cache, token cache, and lightweight rate helper.
- `redis_pool.py`: shared async Redis pool for HTTP and WebSocket handlers.
- `user_service.py`: user/site access helpers and cache invalidation.
- `token_cleanup.py`: removes stale token cache entries.
- `security.py` and `jwt_scheduler.py`: optional secret-generation utilities.

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
```

Use one stable, high-entropy `JWT_SECRET_KEY` for all API services in the same
deployment. Rotating the key immediately invalidates existing access and
refresh tokens, so do it only as a planned security operation.

## Passwords

The project uses `pwdlib[argon2]` to hash and verify user passwords. Do not
store plaintext passwords and do not replace Argon2 with reversible encryption.

## Redis Usage

Redis stores small authentication and authorisation state:

- user cache entries;
- refresh-token references;
- access-token `jti` lists;
- effective site access cache;
- Lua script state used by the shared cache helpers.

Redis is not used for live video frames.

## Used By

- `examples/db_management/`
- `examples/local_notification_server/`
- `examples/streaming_web/`
- `examples/violation_records/`
- `examples/YOLO_server_api/`

Start those services directly; they import this directory through their
lifespan and dependency wiring.
