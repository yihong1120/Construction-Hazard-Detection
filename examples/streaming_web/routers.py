from __future__ import annotations

import redis.asyncio as redis
from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi import Response
from fastapi import Security
from fastapi import WebSocket
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.redis_pool import get_redis_pool
from examples.auth.redis_pool import get_redis_pool_ws
from examples.streaming_web import stream_catalog_service
from examples.streaming_web import streaming_api_service
from examples.streaming_web import streaming_metadata_service
from examples.streaming_web.schemas import LabelListResponse
from examples.streaming_web.schemas import OverlayLanguageListResponse
from examples.streaming_web.schemas import StreamPlaybackBatchRequest
from examples.streaming_web.schemas import StreamPlaybackRequest

router: APIRouter = APIRouter()


@router.get('/labels', response_model=LabelListResponse)
async def get_labels_route(
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    db: AsyncSession = Depends(get_db),
) -> LabelListResponse:
    """Return the labels visible to the authenticated user.

    Args:
        credentials: Verified JWT credentials for the caller.
        db: Request-scoped database session.

    Returns:
        Validated list of site labels visible to the caller.
    """
    return await stream_catalog_service.get_visible_labels(credentials, db)


@router.get('/overlay-languages', response_model=OverlayLanguageListResponse)
async def get_overlay_languages(
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> OverlayLanguageListResponse:
    """Return canonical overlay-language options for clients.

    Args:
        credentials: Verified JWT credentials for the caller.

    Returns:
        Backend-supported language and translation capabilities.
    """
    return streaming_api_service.get_overlay_languages(credentials)


@router.get('/media-auth', include_in_schema=False)
async def authorise_media_request(
    request: Request,
    rds: redis.Redis = Depends(get_redis_pool),
) -> Response:
    """Authorise a MediaMTX request through a scoped opaque capability.

    Args:
        request: Incoming media-proxy authorisation request.
        rds: Request-scoped Redis connection.

    Returns:
        Empty success response or a rejected authorisation response.
    """
    return await streaming_api_service.authorise_media_request(request, rds)


@router.get('/stream-playback/sessions/{session_id}/index.m3u8')
async def stream_playback_session_playlist(
    session_id: str,
    request: Request,
    rds: redis.Redis = Depends(get_redis_pool),
) -> Response:
    """Serve the stable authorised HLS playlist for a playback session.

    Args:
        session_id: Opaque playback session identifier.
        request: Incoming client request containing its session token.
        rds: Request-scoped Redis connection.

    Returns:
        Rewritten HLS playlist response for the session.
    """
    return await streaming_api_service.stream_playback_session_playlist(
        session_id,
        request,
        rds,
    )


@router.post('/stream-playback')
async def request_stream_playback(
    request_body: StreamPlaybackRequest,
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> JSONResponse:
    """Create or update a stable playback session for one camera.

    Args:
        request_body: Validated one-stream playback request.
        credentials: Verified JWT credentials for the caller.
        db: Request-scoped database session.
        rds: Request-scoped Redis connection.

    Returns:
        JSON body describing the stable playback session.
    """
    return await streaming_api_service.request_stream_playback(
        request_body,
        credentials,
        db,
        rds,
    )


@router.post('/stream-playback/batch')
async def request_stream_playback_batch(
    request_body: StreamPlaybackBatchRequest,
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> JSONResponse:
    """Create stable playback sessions for a batch of cameras.

    Args:
        request_body: Validated batch playback request.
        credentials: Verified JWT credentials for the caller.
        db: Request-scoped database session.
        rds: Request-scoped Redis connection.

    Returns:
        JSON body containing playback sessions and batch metadata.
    """
    return await streaming_api_service.request_stream_playback_batch(
        request_body,
        credentials,
        db,
        rds,
    )


@router.post('/stream-playback/release')
async def release_stream_playback(
    request_body: StreamPlaybackRequest,
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    rds: redis.Redis = Depends(get_redis_pool),
) -> JSONResponse:
    """Release one playback session and its demand lease.

    Args:
        request_body: Request identifying the session to release.
        credentials: Verified JWT credentials for the caller.
        rds: Request-scoped Redis connection.

    Returns:
        JSON status confirming whether the session was released.
    """
    return await streaming_api_service.release_stream_playback(
        request_body,
        credentials,
        rds,
    )


@router.get('/streams/{label}')
async def get_streams_for_label_route(
    label: str,
    overlay: str | None = None,
    language: str | None = None,
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> JSONResponse:
    """Return stable playback details for accessible site streams.

    Args:
        label: Site label to enumerate.
        overlay: Optional overlay mode requested by the client.
        language: Optional overlay label language.
        credentials: Verified JWT credentials for the caller.
        db: Request-scoped database session.
        rds: Request-scoped Redis connection.

    Returns:
        JSON body containing visible stream playback descriptors.
    """
    return await streaming_api_service.get_streams_for_label(
        label,
        overlay,
        language,
        credentials,
        db,
        rds,
    )


@router.get('/webrtc/ice-servers')
async def get_webrtc_ice_servers(
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> dict[str, list[dict[str, object]]]:
    """Return authenticated ICE servers for peer-connection clients.

    Args:
        credentials: Verified JWT credentials for the caller.

    Returns:
        STUN and optional TURN server configuration.
    """
    return streaming_api_service.get_webrtc_ice_servers(credentials)


@router.get('/metadata/stream-id/{label}/{stream_id}')
async def metadata_stream_id(
    request: Request,
    label: str,
    stream_id: str,
    overlay: str | None = None,
    language: str | None = None,
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> StreamingResponse:
    """Stream live metadata by stable stream identifier as SSE.

    Args:
        request: Incoming SSE request used for disconnect detection.
        label: Site label containing the stream.
        stream_id: Encoded configured stream identifier.
        overlay: Optional requested overlay mode.
        language: Optional requested overlay language.
        credentials: Verified JWT credentials for the caller.
        db: Request-scoped database session.
        rds: Request-scoped Redis connection.

    Returns:
        Streaming server-sent-event response.
    """
    return await streaming_metadata_service.metadata_stream_response(
        request,
        label,
        stream_id,
        overlay,
        language,
        credentials,
        db,
        rds,
    )


@router.websocket('/ws/metadata-id/{label}/{stream_id}')
async def websocket_metadata_stream_id(
    websocket: WebSocket,
    label: str,
    stream_id: str,
    rds: redis.Redis = Depends(get_redis_pool_ws),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delegate metadata WebSocket lifecycle handling to its service.

    Args:
        websocket: WebSocket connection to accept and serve.
        label: Site label containing the stream.
        stream_id: Encoded configured stream identifier.
        rds: WebSocket-scoped Redis connection.
        db: Request-scoped database session.
    """
    await streaming_metadata_service.metadata_stream_websocket(
        websocket,
        label,
        stream_id,
        rds,
        db,
    )
