from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi import Response
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.redis_pool import get_redis_pool
from examples.db_management.schemas.playback import PlaybackRenewRequest
from examples.db_management.schemas.playback import PlaybackSessionRequest
from examples.db_management.schemas.playback import PlaybackWallRequest
from examples.db_management.services.playback_services import (
    delete_playback_response,
)
from examples.db_management.services.playback_services import (
    playback_session_response,
)
from examples.db_management.services.playback_services import (
    playback_wall_response,
)
from examples.db_management.services.playback_services import (
    renew_playback_response,
)

router = APIRouter(prefix='/api/playback', tags=['playback'])


@router.post('/sessions')
async def create_playback_session(
    payload: PlaybackSessionRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> JSONResponse:
    """Create one signed single-camera playback session.

    Args:
        payload: Validated session request.
        request: HTTP request containing the caller's bearer token.
        db: Database session reserved for the request lifecycle.
        redis: Redis connection storing the playback session.

    Returns:
        The streaming service session response.
    """
    return await playback_session_response(payload, request, redis)


@router.post('/walls')
async def create_playback_wall(
    payload: PlaybackWallRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> JSONResponse:
    """Create signed playback sessions for a camera wall.

    Args:
        payload: Validated multi-camera wall request.
        request: HTTP request containing the caller's bearer token.
        db: Database session reserved for the request lifecycle.
        redis: Redis connection storing the playback sessions.

    Returns:
        The streaming service wall response.
    """
    return await playback_wall_response(payload, request, redis)


@router.post('/sessions/renew')
async def renew_playback_session(
    payload: PlaybackRenewRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> JSONResponse:
    """Renew the signed URLs for a playback session.

    Args:
        payload: Validated renewal request.
        request: HTTP request containing the caller's bearer token.
        db: Database session reserved for the request lifecycle.
        redis: Redis connection storing the playback session.

    Returns:
        The renewed playback response.
    """
    return await renew_playback_response(payload, request, redis)


@router.delete('/sessions/{session_id}', status_code=204)
async def delete_playback_session(
    session_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> Response:
    """Delete a public playback session.

    Args:
        session_id: Public playback session identifier.
        request: HTTP request containing the caller's bearer token.
        db: Database session reserved for the request lifecycle.
        redis: Redis connection storing the playback session.

    Returns:
        An empty successful deletion response.
    """
    return await delete_playback_response(session_id, request, redis)
