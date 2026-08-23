from __future__ import annotations

import redis.asyncio as redis
from fastapi import HTTPException
from fastapi import Request
from fastapi import WebSocket
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.config import Settings
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.streaming_web import playback_service
from examples.streaming_web.media_paths import build_annotated_media_path
from examples.streaming_web.media_paths import build_media_hls_url
from examples.streaming_web.media_paths import build_media_path
from examples.streaming_web.media_paths import build_overlay_demand_key
from examples.streaming_web.media_paths import build_overlay_ready_key
from examples.streaming_web.media_paths import decode_media_segment
from examples.streaming_web.metadata_keys import (
    build_metadata_key_from_stream_id,
)
from examples.streaming_web.metadata_keys import get_metadata_site_generation
from examples.streaming_web.overlay_renderer import normalise_label_language
from examples.streaming_web.overlay_renderer import normalise_overlay_mode
from examples.streaming_web.playback_hls import authorise_label_access
from examples.streaming_web.streaming_metadata_handlers import (
    handle_metadata_stream_id_ws,
)
from examples.streaming_web.streaming_metadata_handlers import (
    metadata_stream_generator,
)

settings = Settings()


async def metadata_stream_websocket(
    websocket: WebSocket,
    label: str,
    stream_id: str,
    rds: redis.Redis,
    db: AsyncSession,
) -> None:
    """Serve an authorised metadata WebSocket through the handler service.

    Args:
        websocket: WebSocket connection to accept and serve.
        label: Site label containing the stream.
        stream_id: Encoded configured stream identifier.
        rds: Redis connection used to read metadata frames.
        db: Database session used to authorise the stream.
    """
    await handle_metadata_stream_id_ws(
        websocket=websocket,
        label=label,
        stream_id=stream_id,
        rds=rds,
        settings=settings,
        db=db,
    )


async def metadata_stream_response(
    request: Request,
    label: str,
    stream_id: str,
    overlay: str | None,
    language: str | None,
    credentials: JwtAuthorizationCredentials,
    db: AsyncSession,
    rds: redis.Redis,
) -> StreamingResponse:
    """Build an SSE stream with optional backend-overlay state updates.

    Args:
        request: Incoming request used to detect client disconnection.
        label: Site label containing the stream.
        stream_id: Encoded configured stream identifier.
        overlay: Optional requested overlay mode.
        language: Optional requested overlay label language.
        credentials: Verified JWT credentials for the caller.
        db: Database session used to authorise the requested site.
        rds: Redis connection used to read metadata and overlay state.

    Returns:
        Non-buffered server-sent-event response for live frame metadata.

    Raises:
        HTTPException: If site access or the overlay language is invalid.
    """
    await authorise_label_access(credentials, db, label)
    await db.close()
    generation = await get_metadata_site_generation(rds, label)
    redis_key = build_metadata_key_from_stream_id(
        label,
        stream_id,
        generation,
    )
    overlay_ready_key: str | None = None
    overlay_ready_payload: dict[str, object] | None = None
    overlay_demand_key: str | None = None
    if normalise_overlay_mode(overlay) == 'backend':
        # SSE metadata must advertise the exact producer path it has leased.
        overlay_language = normalise_label_language(language)
        if (
            overlay_language
            not in playback_service._allowed_overlay_languages()
        ):
            raise HTTPException(status_code=422, detail='unsupported_language')
        stream_name = decode_media_segment(stream_id)
        media_path = build_media_path(label, stream_name)
        overlay_path = build_annotated_media_path(media_path, overlay_language)
        overlay_hls_url = build_media_hls_url(overlay_path)
        overlay_ready_key = build_overlay_ready_key(overlay_path)
        overlay_demand_key = build_overlay_demand_key(
            media_path,
            overlay_language,
        )
        overlay_ready_payload = {
            'profile': 'overlay',
            'status': 'ready',
            'state': 'ready',
            'playback_ready': True,
            'overlay_ready': True,
            'language': overlay_language,
            'media_path': overlay_path,
            'playback_url': overlay_hls_url,
            'media_hls_url': overlay_hls_url,
        }
    return StreamingResponse(
        metadata_stream_generator(
            request,
            rds,
            redis_key,
            overlay_ready_key=overlay_ready_key,
            overlay_ready_payload=overlay_ready_payload,
            overlay_demand_key=overlay_demand_key,
            overlay_demand_ttl_seconds=(
                playback_service.OVERLAY_DEMAND_TTL_SECONDS
            ),
            overlay_demand_refresh_seconds=max(
                1.0,
                playback_service.OVERLAY_DEMAND_TTL_SECONDS / 2,
            ),
        ),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate',
            'Pragma': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )
