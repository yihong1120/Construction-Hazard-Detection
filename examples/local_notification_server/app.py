from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from examples.auth.lifespan import global_lifespan
from examples.local_notification_server.fcm_service import init_firebase_app
from examples.local_notification_server.routers import (
    router as notification_router,
)
load_dotenv()

logger = logging.getLogger(__name__)
_sensitive_payload_keys = {
    'device_token',
    'fcm_token',
    'token',
    'access_token',
    'refresh_token',
}
_sensitive_payload_key_aliases = {
    key.replace('_', '').replace('-', '').lower()
    for key in _sensitive_payload_keys
}


def _is_sensitive_payload_key(key: object) -> bool:
    """Return whether a request-body key may contain token material."""
    if not isinstance(key, str):
        return False
    normalised = key.replace('_', '').replace('-', '').lower()
    return key.lower() in _sensitive_payload_keys or (
        normalised in _sensitive_payload_key_aliases
    )


def _redact_sensitive_payload(value: object) -> object:
    """Redact token-like values before logging request bodies."""
    if isinstance(value, dict):
        return {
            key: (
                '<redacted>'
                if _is_sensitive_payload_key(key)
                else _redact_sensitive_payload(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_payload(item) for item in value]
    return value


def _safe_body_preview(body: bytes) -> str:
    """Return a bounded, token-redacted request body preview."""
    try:
        parsed = json.loads(body.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body[:2000].decode('utf-8', errors='replace')
    redacted = _redact_sensitive_payload(parsed)
    return json.dumps(redacted, ensure_ascii=False)[:2000]


@asynccontextmanager
async def notification_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise notification-server resources during application lifespan.

    Args:
        app: FastAPI application instance receiving shared resources.

    Yields:
        Control to FastAPI while shared database, Redis, and Firebase resources
        are available.
    """
    cred_path = os.getenv(
        'FIREBASE_CRED_PATH',
        'path/to/your/firebase/credentials.json',
    )
    project_id = os.getenv('FIREBASE_PROJECT_ID', 'your-firebase-project-id')

    async with global_lifespan(app):
        init_firebase_app(cred_path=cred_path, project_id=project_id)
        yield

app: FastAPI = FastAPI(lifespan=notification_lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Log request validation failures and return a structured response.

    Args:
        request: Incoming FastAPI request that failed validation.
        exc: Validation exception raised by FastAPI.

    Returns:
        JSON response containing FastAPI's validation details.
    """
    body = await request.body()
    body_preview = _safe_body_preview(body)
    logger.warning(
        'Request validation failed path=%s errors=%s body=%s',
        request.url.path,
        exc.errors(),
        body_preview,
    )
    return JSONResponse(status_code=422, content={'detail': exc.errors()})

# Include routers for  notification services
app.include_router(notification_router)


def main() -> None:
    """Run the notification FastAPI application with Uvicorn."""
    uvicorn.run(app, host='127.0.0.1', port=8003)


if __name__ == '__main__':
    main()

"""
uvicorn examples.local_notification_server.app:app\
    --host 127.0.0.1 \
    --port 8003 \
    --workers 4

uv run uvicorn examples.local_notification_server.app:app\
    --host 127.0.0.1 \
    --port 8003 \
    --workers 4
"""
