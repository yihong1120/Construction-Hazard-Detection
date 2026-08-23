from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from collections.abc import Mapping
from typing import Final

import redis.asyncio as redis
from fastapi import Request
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.config import Settings
from examples.auth.user_service import load_user_access_context
from examples.shared.ws_helpers import authenticate_ws_or_none
from examples.shared.ws_helpers import check_and_maybe_close_on_timeout
from examples.shared.ws_helpers import get_auto_register_jti
from examples.shared.ws_helpers import log_every_n
from examples.shared.ws_helpers import start_session_timer
from examples.shared.ws_utils import _is_websocket_connected
from examples.shared.ws_utils import _safe_websocket_receive_text
from examples.shared.ws_utils import _safe_websocket_send_json
from examples.shared.ws_utils import _safe_websocket_send_text
from examples.streaming_web.metadata_fanout import metadata_fanout
from examples.streaming_web.metadata_keys import build_metadata_key
from examples.streaming_web.metadata_keys import build_metadata_key_from_stream_id
from examples.streaming_web.metadata_keys import get_metadata_site_generation
from examples.streaming_web.redis_service import fetch_latest_metadata_for_key
from examples.streaming_web.schemas import FrameOutData

AUTO_REGISTER_JTI: Final[bool] = get_auto_register_jti()
_json_compact_separators: Final[tuple[str, str]] = (',', ':')
_metadata_client_tick_seconds: Final[float] = 1.0
_metadata_heartbeat_seconds: Final[float] = 15.0
_metadata_redis_error_interval_seconds: Final[float] = 15.0


logger = logging.getLogger(__name__)


def _metadata_has_warning(frame_data: FrameOutData) -> bool:
    """Read the compact warning state from viewer metadata.

    Args:
        frame_data: Decoded Redis metadata record.

    Returns:
        Whether the source frame contains a safety warning.
    """
    return frame_data['has_warning']


def _build_metadata_payload(frame_data: FrameOutData) -> dict[str, object]:
    """Build the payload for clients that render video natively.

    Args:
        frame_data: Decoded Redis metadata record.

    Returns:
        Minimal serialisable warning state without image data.
    """
    return {
        'has_warning': _metadata_has_warning(frame_data),
    }


def _encode_sse_event(
    payload: Mapping[str, object],
    event_type: str = 'metadata',
) -> bytes:
    """Encode one Server-Sent Events message.

    Args:
        payload: Serialisable event payload.
        event_type: SSE event name sent to the client.

    Returns:
        UTF-8 SSE wire message including the required blank terminator.
    """
    event_id = str(payload.get('id') or '')
    lines = []
    if event_id:
        lines.append(f'id: {event_id}')
    lines.append(f'event: {event_type}')
    lines.append(
        'data: '
        + json.dumps(payload, separators=_json_compact_separators),
    )
    return ('\n'.join(lines) + '\n\n').encode('utf-8')


async def _refresh_overlay_demand(
    rds: redis.Redis,
    overlay_demand_key: str | None,
    overlay_demand_ttl_seconds: int | None,
) -> None:
    """Keep an on-demand overlay stream alive for an SSE viewer.

    Args:
        rds: Redis connection used to renew the demand key.
        overlay_demand_key: Optional demand key for the overlay producer.
        overlay_demand_ttl_seconds: Optional renewed lease duration.
    """
    if not overlay_demand_key or not overlay_demand_ttl_seconds:
        return
    await rds.set(
        overlay_demand_key,
        b'1',
        ex=overlay_demand_ttl_seconds,
    )


async def _overlay_ready_event(
    rds: redis.Redis,
    overlay_ready_key: str | None,
    overlay_ready_payload: Mapping[str, object] | None,
) -> bytes | None:
    """Build an overlay-ready SSE message when the ready key is present.

    Args:
        rds: Redis connection used to inspect readiness.
        overlay_ready_key: Optional key set by the overlay producer.
        overlay_ready_payload: Optional client payload for the ready event.

    Returns:
        Encoded ready event, or ``None`` while the producer is not ready.
    """
    if not overlay_ready_key or overlay_ready_payload is None:
        return None
    if not await rds.exists(overlay_ready_key):
        return None
    return _encode_sse_event(overlay_ready_payload, event_type='overlay_ready')


async def metadata_stream_generator(
    request: Request,
    rds: redis.Redis,
    redis_key: str,
    overlay_ready_key: str | None = None,
    overlay_ready_payload: Mapping[str, object] | None = None,
    overlay_demand_key: str | None = None,
    overlay_demand_ttl_seconds: int | None = None,
    overlay_demand_refresh_seconds: float = 30.0,
) -> AsyncIterator[bytes]:
    """Yield compact warning metadata for native MediaMTX video viewers.

    Args:
        request: HTTP request used to detect client disconnection.
        rds: Redis connection used to consume metadata and overlay state.
        redis_key: Canonical metadata Redis Stream key.
        overlay_ready_key: Optional overlay readiness marker key.
        overlay_ready_payload: Optional payload for the first ready event.
        overlay_demand_key: Optional producer demand lease key.
        overlay_demand_ttl_seconds: Optional demand lease duration.
        overlay_demand_refresh_seconds: Minimum interval between lease renewals.

    Yields:
        Encoded SSE events, heartbeats, and rate-limited Redis-error events.
    """
    last_heartbeat = asyncio.get_running_loop().time()
    last_overlay_demand_refresh = 0.0
    last_redis_error_event = float('-inf')
    last_redis_error_log = float('-inf')
    overlay_ready_sent = False
    # Tell browsers to retry conservatively before the first Redis read blocks.
    yield b'retry: 15000\n: connected\n\n'
    subscription = await metadata_fanout.subscribe(
        rds,
        redis_key,
        fetcher=fetch_latest_metadata_for_key,
    )
    try:
        while not await request.is_disconnected():
            now = asyncio.get_running_loop().time()
            try:
                last_overlay_demand_refresh = (
                    await _refresh_overlay_demand_if_due(
                        rds,
                        overlay_demand_key,
                        overlay_demand_ttl_seconds,
                        overlay_demand_refresh_seconds,
                        last_overlay_demand_refresh,
                        now,
                    )
                )
                overlay_ready_sent, overlay_event = (
                    await _next_overlay_ready_event(
                        rds,
                        overlay_ready_key,
                        overlay_ready_payload,
                        overlay_ready_sent,
                    )
                )
                if overlay_event is not None:
                    last_heartbeat = now
                    yield overlay_event
                    continue
                frame_data = await asyncio.wait_for(
                    subscription.get(),
                    timeout=_metadata_client_tick_seconds,
                )
                if isinstance(frame_data, Exception):
                    raise frame_data
            except asyncio.TimeoutError:
                last_heartbeat, heartbeat = _heartbeat_event(now, last_heartbeat)
                if heartbeat is not None:
                    yield heartbeat
                continue
            except Exception as exc:
                (
                    last_redis_error_event,
                    last_redis_error_log,
                    last_heartbeat,
                    event,
                ) = _metadata_read_error_event(
                    redis_key,
                    exc,
                    now,
                    last_redis_error_event,
                    last_redis_error_log,
                    last_heartbeat,
                )
                if event is not None:
                    yield event
                await asyncio.sleep(1.0)
                continue

            _last_id, last_heartbeat, event = _metadata_frame_event(
                frame_data,
                redis_key,
                '',
                now,
                last_heartbeat,
            )
            if event is not None:
                yield event
    finally:
        await subscription.close()


async def _refresh_overlay_demand_if_due(
    rds: redis.Redis,
    overlay_demand_key: str | None,
    overlay_demand_ttl_seconds: int | None,
    refresh_seconds: float,
    last_refresh: float,
    now: float,
) -> float:
    """Refresh overlay demand only when the viewer lease is due.

    Args:
        rds: Redis connection used to renew the demand key.
        overlay_demand_key: Optional producer demand lease key.
        overlay_demand_ttl_seconds: Optional renewed lease duration.
        refresh_seconds: Minimum period between renewals.
        last_refresh: Monotonic timestamp of the previous renewal.
        now: Current monotonic timestamp.

    Returns:
        Timestamp of the renewal or the unchanged previous timestamp.
    """
    if (
        overlay_demand_key
        and now - last_refresh >= refresh_seconds
    ):
        await _refresh_overlay_demand(
            rds,
            overlay_demand_key,
            overlay_demand_ttl_seconds,
        )
        return now
    return last_refresh


async def _next_overlay_ready_event(
    rds: redis.Redis,
    overlay_ready_key: str | None,
    overlay_ready_payload: Mapping[str, object] | None,
    already_sent: bool,
) -> tuple[bool, bytes | None]:
    """Return the overlay-ready event once for the current SSE viewer.

    Args:
        rds: Redis connection used to inspect readiness.
        overlay_ready_key: Optional overlay readiness marker key.
        overlay_ready_payload: Optional client payload for the ready event.
        already_sent: Whether this viewer has received a ready event.

    Returns:
        Updated sent flag and optional encoded ready event.
    """
    if already_sent:
        return True, None
    event = await _overlay_ready_event(
        rds,
        overlay_ready_key,
        overlay_ready_payload,
    )
    return event is not None, event


def _heartbeat_event(
    now: float,
    last_heartbeat: float,
) -> tuple[float, bytes | None]:
    """Emit a keepalive only after the configured idle period.

    Args:
        now: Current monotonic timestamp.
        last_heartbeat: Monotonic timestamp of the prior heartbeat or event.

    Returns:
        Updated heartbeat timestamp and optional encoded keepalive event.
    """
    if now - last_heartbeat >= _metadata_heartbeat_seconds:
        return now, b': keepalive\n\n'
    return last_heartbeat, None


def _metadata_read_error_event(
    redis_key: str,
    exc: Exception,
    now: float,
    last_error_event: float,
    last_error_log: float,
    last_heartbeat: float,
) -> tuple[float, float, float, bytes | None]:
    """Rate-limit Redis errors while keeping SSE clients alive.

    Args:
        redis_key: Metadata key whose read failed.
        exc: Exception raised by the Redis read.
        now: Current monotonic timestamp.
        last_error_event: Timestamp of the previous error event.
        last_error_log: Timestamp of the previous error log.
        last_heartbeat: Timestamp of the prior heartbeat or event.

    Returns:
        Updated error, log, heartbeat timestamps and an optional SSE event.
    """
    if now - last_error_log >= _metadata_redis_error_interval_seconds:
        logger.info(
            f"[metadata] Redis read failed for {redis_key}: {exc}",
        )
        last_error_log = now
    if now - last_error_event >= _metadata_redis_error_interval_seconds:
        return (
            now,
            last_error_log,
            now,
            _encode_sse_event(
                {'source': 'redis', 'code': 'redis_unavailable'},
                event_type='redis_error',
            ),
        )
    last_heartbeat, heartbeat = _heartbeat_event(now, last_heartbeat)
    return last_error_event, last_error_log, last_heartbeat, heartbeat


def _metadata_frame_event(
    frame_data: FrameOutData | None,
    redis_key: str,
    last_id: str,
    now: float,
    last_heartbeat: float,
) -> tuple[str, float, bytes | None]:
    """Encode one metadata update or idle heartbeat for SSE clients.

    Args:
        frame_data: Optional next Redis metadata record.
        redis_key: Metadata key used for diagnostic output.
        last_id: Previously delivered Redis Stream identifier.
        now: Current monotonic timestamp.
        last_heartbeat: Timestamp of the prior heartbeat or event.

    Returns:
        Updated message identifier, heartbeat timestamp, and optional SSE event.
    """
    if frame_data:
        last_id = str(frame_data['id'])
        payload = _build_metadata_payload(frame_data)
        payload['id'] = last_id
        logger.debug(
            '[SSE-Metadata] send key=%s id=%s has_warning=%s',
            redis_key,
            last_id,
            payload['has_warning'],
        )
        return last_id, now, _encode_sse_event(payload)
    last_heartbeat, heartbeat = _heartbeat_event(now, last_heartbeat)
    return last_id, last_heartbeat, heartbeat


async def metadata_push_loop(
    websocket: WebSocket,
    rds: redis.Redis,
    redis_key: str,
    client_ip: str,
    username: str,
) -> int:
    """Push lightweight stream metadata as JSON whenever Redis advances.

    Args:
        websocket: Accepted authenticated WebSocket connection.
        rds: Redis connection used to consume metadata records.
        redis_key: Canonical metadata Redis Stream key.
        client_ip: Peer IP address for diagnostic output.
        username: Authenticated username for diagnostic output.

    Returns:
        Count of successfully delivered metadata updates.
    """
    update_count = 0
    session_start = start_session_timer()
    receive_task: asyncio.Task[str | None] | None = asyncio.create_task(
        _safe_websocket_receive_text(websocket, f"{client_ip} ({username})"),
    )
    subscription = await metadata_fanout.subscribe(
        rds,
        redis_key,
        fetcher=fetch_latest_metadata_for_key,
    )

    try:
        while True:
            client_tag = f"[WebSocket-Metadata] {client_ip} ({username})"
            if not await _metadata_websocket_is_active(
                websocket,
                session_start,
                client_tag,
            ):
                break

            if receive_task and receive_task.done():
                if not await _handle_metadata_receive_task(
                    websocket,
                    receive_task,
                    client_tag,
                ):
                    break
                receive_task = asyncio.create_task(
                    _safe_websocket_receive_text(
                        websocket,
                        f"{client_ip} ({username})",
                    ),
                )

            try:
                frame_data = await asyncio.wait_for(
                    subscription.get(),
                    timeout=_metadata_client_tick_seconds,
                )
            except asyncio.TimeoutError:
                continue

            if isinstance(frame_data, Exception):
                logger.warning(
                    '[WebSocket-Metadata] reader failed key=%s error_type=%s',
                    redis_key,
                    type(frame_data).__name__,
                )
                continue

            if frame_data:
                last_id = str(frame_data['id'])
                if not await _send_metadata_websocket_frame(
                    websocket,
                    frame_data,
                    redis_key,
                    last_id,
                    client_tag,
                ):
                    break
                update_count += 1
                log_every_n(
                    f"[WebSocket-Metadata] {client_ip} ({username})",
                    update_count,
                    unit='metadata updates',
                )
    finally:
        await subscription.close()
        if receive_task and not receive_task.done():
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass

    return update_count


async def _metadata_websocket_is_active(
    websocket: WebSocket,
    session_start: float,
    client_tag: str,
) -> bool:
    """Determine whether a metadata WebSocket may receive more updates.

    Args:
        websocket: Accepted client WebSocket.
        session_start: Monotonic session start timestamp.
        client_tag: Diagnostic label for timeout handling.

    Returns:
        ``True`` while the connection remains active and within its time limit.
    """
    timed_out = await check_and_maybe_close_on_timeout(
        websocket,
        session_start,
        client_tag,
        use_text=True,
    )
    return not timed_out and _is_websocket_connected(websocket)


async def _handle_metadata_receive_task(
    websocket: WebSocket,
    receive_task: asyncio.Task[str | None],
    client_tag: str,
) -> bool:
    """Process a completed client message and respond to a ping.

    Args:
        websocket: Accepted client WebSocket.
        receive_task: Completed task that received one client message.
        client_tag: Diagnostic label for safe sending.

    Returns:
        ``False`` when the client has disconnected; otherwise ``True``.
    """
    message = receive_task.result()
    if message is None:
        return False
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        data = {}
    action = data.get('action') or data.get('type')
    if action == 'ping':
        await _safe_websocket_send_text(
            websocket,
            json.dumps(
                {'action': 'pong'},
                separators=_json_compact_separators,
            ),
            client_tag.removeprefix('[WebSocket-Metadata] '),
        )
    return True


async def _send_metadata_websocket_frame(
    websocket: WebSocket,
    frame_data: FrameOutData,
    redis_key: str,
    last_id: str,
    client_tag: str,
) -> bool:
    """Send one Redis metadata update with diagnostic output.

    Args:
        websocket: Accepted client WebSocket.
        frame_data: Decoded metadata record to serialise.
        redis_key: Source metadata key for diagnostics.
        last_id: Redis Stream identifier for diagnostics.
        client_tag: Diagnostic label for safe sending.

    Returns:
        Whether the JSON frame was sent successfully.
    """
    payload = _build_metadata_payload(frame_data)
    if not await _safe_websocket_send_json(
        websocket,
        payload,
        client_tag.removeprefix('[WebSocket-Metadata] '),
    ):
        return False
    logger.debug(
        '[WebSocket-Metadata] send key=%s id=%s has_warning=%s',
        redis_key,
        last_id,
        payload['has_warning'],
    )
    return True


async def handle_metadata_ws(
    websocket: WebSocket,
    label: str,
    key: str,
    rds: redis.Redis,
    settings: Settings,
    db: AsyncSession | None = None,
    redis_key_override: str | None = None,
) -> None:
    """Serve metadata over WebSocket without duplicating image bytes.

    Args:
        websocket: New client WebSocket to accept and authenticate.
        label: Site label containing the stream.
        key: Decoded stream name for the default metadata key.
        rds: Redis connection used to consume metadata records.
        settings: Authentication and session settings.
        db: Optional database session used to enforce site access.
        redis_key_override: Optional canonical key for encoded stream routes.
    """
    client_ip = websocket.client.host if websocket.client else 'unknown'
    logger.info(
        (
            f"[WebSocket-Metadata] New connection from {client_ip} for "
            f"{label}/{key}"
        ),
    )
    await websocket.accept()
    username, _ = await authenticate_ws_or_none(
        websocket,
        rds,
        settings,
        auto_register_jti=AUTO_REGISTER_JTI,
        client_tag=f"[WebSocket-Metadata] {client_ip}",
    )
    if not username:
        return
    logger.info(f"[WebSocket-Metadata] {client_ip}: Authenticated as {username}")

    if db is not None:
        try:
            _, user_site_names, user_role = await load_user_access_context(
                db, username,
            )
        except Exception:
            await websocket.close(code=4001, reason='User not found')
            return
        finally:
            # Redis serves the long-lived data stream after SQL authorisation.
            await db.close()
        if user_role != 'super_admin' and label not in user_site_names:
            logger.info(
                f"[WebSocket-Metadata] {client_ip} ({username}): "
                f"Access denied to label '{label}'",
            )
            await websocket.close(code=4003, reason='Access denied')
            return

    if redis_key_override is None:
        generation = await get_metadata_site_generation(rds, label)
        redis_key = build_metadata_key(label, key, generation)
    else:
        redis_key = redis_key_override

    update_count = 0
    try:
        update_count = await metadata_push_loop(
            websocket,
            rds,
            redis_key,
            client_ip,
            username,
        )
    except WebSocketDisconnect:
        logger.info(
            (
                f"[WebSocket-Metadata] {client_ip} ({username}): Client "
                f"disconnected after {update_count} metadata updates"
            ),
        )
    finally:
        logger.info(
            (
                f"[WebSocket-Metadata] {client_ip} ({username}): "
                f'Connection closed, total updates: {update_count}'
            ),
        )


async def handle_metadata_stream_id_ws(
    websocket: WebSocket,
    label: str,
    stream_id: str,
    rds: redis.Redis,
    settings: Settings,
    db: AsyncSession | None = None,
) -> None:
    """Serve metadata using the stable encoded stream identifier.

    Args:
        websocket: New client WebSocket to accept and authenticate.
        label: Site label containing the stream.
        stream_id: Encoded configured stream identifier.
        rds: Redis connection used to consume metadata records.
        settings: Authentication and session settings.
        db: Optional database session used to enforce site access.
    """
    generation = await get_metadata_site_generation(rds, label)
    redis_key = build_metadata_key_from_stream_id(
        label,
        stream_id,
        generation,
    )
    await handle_metadata_ws(
        websocket=websocket,
        label=label,
        key=stream_id,
        rds=rds,
        settings=settings,
        db=db,
        redis_key_override=redis_key,
    )
