from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from typing import cast
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
from examples.streaming_web.backend.redis_service import (
    fetch_latest_metadata_for_key,
)
from examples.streaming_web.backend.schemas import FrameOutData
from examples.streaming_web.backend.utils import Utils

# Module-level alias retained for test patching.
get_user_and_sites = load_user_access_context

AUTO_REGISTER_JTI: Final[bool] = get_auto_register_jti()
_json_compact_separators: Final[tuple[str, str]] = (',', ':')
_metadata_poll_interval: Final[float] = 0.01


def _metadata_has_warning(frame_data: FrameOutData) -> bool:
    """Read compact warning state from MediaMTX viewer metadata."""
    raw_value = frame_data.get('has_warning')
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        value = raw_value.strip().lower()
        if value in {'1', 'true', 'yes', 'on'}:
            return True
    return False


def _build_metadata_payload(frame_data: FrameOutData) -> dict[str, object]:
    """Build a lightweight payload for clients that render video natively."""
    return {
        'has_warning': _metadata_has_warning(frame_data),
    }


def _encode_sse_event(payload: dict[str, object]) -> bytes:
    """Encode metadata as one Server-Sent Events message."""
    event_id = str(payload.get('id') or '')
    lines = []
    if event_id:
        lines.append(f'id: {event_id}')
    lines.append('event: metadata')
    lines.append(
        'data: '
        + json.dumps(payload, separators=_json_compact_separators),
    )
    return ('\n'.join(lines) + '\n\n').encode('utf-8')


async def metadata_stream_generator(
    request: Request,
    rds: redis.Redis,
    redis_key: str,
) -> AsyncIterator[bytes]:
    """Yield compact warning metadata for native MediaMTX video viewers."""
    last_id = '0'
    last_heartbeat = asyncio.get_running_loop().time()
    yield b'retry: 15000\n: connected\n\n'
    while not await request.is_disconnected():
        try:
            _aw_frame: Awaitable[FrameOutData | None] = cast(
                Awaitable[FrameOutData | None],
                fetch_latest_metadata_for_key(rds, redis_key, last_id),
            )
            frame_data = await asyncio.wait_for(_aw_frame, timeout=2.0)
        except asyncio.TimeoutError:
            await asyncio.sleep(_metadata_poll_interval)
            continue
        except Exception as exc:
            print(f"[metadata] Redis read failed for {redis_key}: {exc}")
            await asyncio.sleep(1.0)
            continue

        if frame_data:
            last_id = str(frame_data['id'])
            last_heartbeat = asyncio.get_running_loop().time()
            payload = _build_metadata_payload(frame_data)
            payload['id'] = last_id
            yield _encode_sse_event(payload)
        else:
            now = asyncio.get_running_loop().time()
            if now - last_heartbeat >= 15.0:
                last_heartbeat = now
                yield b': keepalive\n\n'
            await asyncio.sleep(_metadata_poll_interval)


async def metadata_push_loop(
    websocket: WebSocket,
    rds: redis.Redis,
    redis_key: str,
    client_ip: str,
    username: str,
) -> int:
    """Push lightweight stream metadata as JSON whenever Redis advances."""
    update_count = 0
    last_id = '0'
    session_start = start_session_timer()
    receive_task: asyncio.Task[str | None] | None = asyncio.create_task(
        _safe_websocket_receive_text(websocket, f"{client_ip} ({username})"),
    )

    try:
        while True:
            if await check_and_maybe_close_on_timeout(
                websocket,
                session_start,
                f"[WebSocket-Metadata] {client_ip} ({username})",
                use_text=True,
            ):
                break
            if not _is_websocket_connected(websocket):
                break

            if receive_task and receive_task.done():
                msg = receive_task.result()
                if msg is None:
                    break
                try:
                    data = json.loads(msg)
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
                        f"{client_ip} ({username})",
                    )
                receive_task = asyncio.create_task(
                    _safe_websocket_receive_text(
                        websocket,
                        f"{client_ip} ({username})",
                    ),
                )

            try:
                _aw_frame: Awaitable[FrameOutData | None] = cast(
                    Awaitable[FrameOutData | None],
                    fetch_latest_metadata_for_key(rds, redis_key, last_id),
                )
                frame_data = await asyncio.wait_for(_aw_frame, timeout=2.0)
            except asyncio.TimeoutError:
                await asyncio.sleep(_metadata_poll_interval)
                continue

            if frame_data:
                last_id = str(frame_data['id'])
                sent = await _safe_websocket_send_json(
                    websocket,
                    _build_metadata_payload(frame_data),
                    f"{client_ip} ({username})",
                )
                if not sent:
                    break
                update_count += 1
                log_every_n(
                    f"[WebSocket-Metadata] {client_ip} ({username})",
                    update_count,
                    unit='metadata updates',
                )
            else:
                await asyncio.sleep(_metadata_poll_interval)
    finally:
        if receive_task and not receive_task.done():
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass

    return update_count


async def handle_metadata_ws(
    websocket: WebSocket,
    label: str,
    key: str,
    rds: redis.Redis,
    settings: Settings,
    db: AsyncSession | None = None,
    redis_key_override: str | None = None,
) -> None:
    """WebSocket endpoint: push warning metadata without image bytes."""
    client_ip = websocket.client.host if websocket.client else 'unknown'
    print(
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
    print(f"[WebSocket-Metadata] {client_ip}: Authenticated as {username}")

    if db is not None:
        try:
            _, user_site_names, user_role = await get_user_and_sites(
                db, username,
            )
        except Exception:
            await websocket.close(code=4001, reason='User not found')
            return
        if user_role != 'super_admin' and label not in user_site_names:
            print(
                f"[WebSocket-Metadata] {client_ip} ({username}): "
                f"Access denied to label '{label}'",
            )
            await websocket.close(code=4003, reason='Access denied')
            return

    redis_key: str = redis_key_override or (
        f"stream_metadata:{Utils.encode(label)}|{Utils.encode(key)}"
    )

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
        print(
            (
                f"[WebSocket-Metadata] {client_ip} ({username}): Client "
                f"disconnected after {update_count} metadata updates"
            ),
        )
    finally:
        print(
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
    """Metadata WebSocket using the stable encoded stream id from overview."""
    redis_key = f"stream_metadata:{Utils.encode(label)}|{stream_id}"
    await handle_metadata_ws(
        websocket=websocket,
        label=label,
        key=stream_id,
        rds=rds,
        settings=settings,
        db=db,
        redis_key_override=redis_key,
    )
