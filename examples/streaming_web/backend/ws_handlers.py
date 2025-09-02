from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from collections.abc import Sequence
from typing import cast
from typing import Final

import redis
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from examples.auth.config import Settings
from examples.shared.ws_helpers import authenticate_ws_or_none
from examples.shared.ws_helpers import check_and_maybe_close_on_timeout
from examples.shared.ws_helpers import get_auto_register_jti
from examples.shared.ws_helpers import log_every_n
from examples.shared.ws_helpers import start_session_timer
from examples.shared.ws_utils import _is_websocket_connected
from examples.shared.ws_utils import _safe_websocket_receive_bytes
from examples.shared.ws_utils import _safe_websocket_receive_text
from examples.shared.ws_utils import _safe_websocket_send_bytes
from examples.shared.ws_utils import _safe_websocket_send_json
from examples.shared.ws_utils import _safe_websocket_send_text
from examples.streaming_web.backend.redis_service import DELIMITER
from examples.streaming_web.backend.redis_service import (
    fetch_latest_frame_for_key,
)
from examples.streaming_web.backend.redis_service import fetch_latest_frames
from examples.streaming_web.backend.redis_service import get_keys_for_label
from examples.streaming_web.backend.redis_service import store_to_redis
from examples.streaming_web.backend.schemas import FrameInHeader
from examples.streaming_web.backend.schemas import FrameOutData
from examples.streaming_web.backend.utils import Utils

# Whether to auto-register JWT IDs for WebSocket sessions.
AUTO_REGISTER_JTI: Final[bool] = get_auto_register_jti()


async def prepare_label_keys(
    websocket: WebSocket,
    rds: redis.Redis,
    label: str,
    client_ip: str,
    username: str,
) -> dict[str, str] | None:
    """Resolve Redis stream keys for a label and notify the client.

    Args:
        websocket: The active WebSocket connection to the client.
        rds: The Redis client instance used to query keys.
        label: The logical label grouping multiple stream keys.
        client_ip: Remote client IP for logging context.
        username: Authenticated username for logging context.

    Returns:
        A mapping of ``key -> last_id`` initialised to ``'0'`` for each found
        key, or ``None`` if no keys were found (in which case the function also
        sends an error to the client and closes the WebSocket).

    Notes:
        This function only prepares state; it does not enter the streaming
        loop.
    """
    # Fetch all Redis keys associated with the given label.
    keys: list[str] = await get_keys_for_label(rds, label)
    if not keys:
        print(
            (
                f"[WebSocket-Labels] {client_ip} ({username}): No keys "
                f"found for label '{label}'"
            ),
        )
        # Inform the client that no streams are available for this label.
        await _safe_websocket_send_json(
            websocket,
            {'error': f"No keys found for label '{label}'"},
            f"{client_ip} ({username})",
        )
        await websocket.close(code=1000, reason='No streams available')
        return None
    # Start at ID '0' to get new entries on the first fetch.
    last_ids: dict[str, str] = {k: '0' for k in keys}
    print(
        (
            f"[WebSocket-Labels] {client_ip} ({username}): Found "
            f"{len(keys)} streams for label '{label}'"
        ),
    )
    return last_ids


async def send_updated_frames(
    websocket: WebSocket,
    updated_data: Sequence[FrameOutData],
    last_ids: dict[str, str],
    client_ip: str,
    username: str,
) -> int:
    """Send one or more updated frames to the client.

    Batches frames retrieved from Redis, building a small JSON header per
    frame, then streams them as binary messages in the form:

    ``<json-header-bytes> + DELIMITER + <frame-bytes>``

    Args:
        websocket: The active WebSocket connection.
        updated_data: A sequence of frame dictionaries fetched from Redis.
        last_ids: Mutable mapping of ``key -> last_id`` updated as frames are
            sent.
        client_ip: Remote client IP for logging context.
        username: Authenticated username for logging context.

    Returns:
        The count of frames successfully sent to the client.
    """
    sent = 0
    for data in updated_data:
        # If the client disconnected mid-loop, stop promptly.
        if not _is_websocket_connected(websocket):
            print(
                (
                    f"[WebSocket-Labels] {client_ip} ({username}): "
                    'Connection closed, stopping frame sending'
                ),
            )
            return sent
        # Prepare the lightweight header describing the frame.
        header: bytes = json.dumps(
            {
                'key': data.get('key', ''),
                'warnings': data.get('warnings', ''),
                'cone_polygons': data.get('cone_polygons', ''),
                'pole_polygons': data.get('pole_polygons', ''),
                'detection_items': data.get('detection_items', ''),
                'width': data.get('width', 0),
                'height': data.get('height', 0),
            },
        ).encode('utf-8')
        # Frame payload may be absent; skip such entries.
        frame_bytes_raw = data.get('frame_bytes', b'')
        frame_bytes: bytes = (
            cast(bytes, frame_bytes_raw) if frame_bytes_raw else b''
        )
        if not frame_bytes:
            continue
        # Track the last delivered ID per key to avoid resending the same
        # frame on the next fetch cycle.
        key = data.get('key', '')
        if key and 'id' in data:
            last_ids[key] = data['id']
        # Concatenate header and payload with the agreed delimiter.
        message_bytes: bytes = header + DELIMITER + frame_bytes
        # Guard again against a disconnect during assembly.
        if not _is_websocket_connected(websocket):
            print(
                (
                    f"[WebSocket-Labels] {client_ip} ({username}): "
                    'Connection closed during frame preparation'
                ),
            )
            return sent
        # Send the binary message; the safe helper returns False on failure
        # rather than raising.
        success = await _safe_websocket_send_bytes(
            websocket,
            message_bytes,
            f"{client_ip} ({username})",
        )
        if success:
            sent += 1
        else:
            print(
                (
                    f"[WebSocket-Labels] {client_ip} ({username}): "
                    'Failed to send frame, stopping'
                ),
            )
            return sent
    return sent


async def label_stream_loop(
    websocket: WebSocket,
    rds: redis.Redis,
    last_ids: dict[str, str],
    client_ip: str,
    username: str,
) -> int:
    """Main loop to stream frames for a label across multiple keys.

    Periodically fetches frames for all known keys using ``last_ids`` as the
    consumer cursors, and sends any updates to the client.

    Args:
        websocket: The active WebSocket connection.
        rds: Redis client used to read frames.
        last_ids: Mapping of ``key -> last_id`` to track progress.
        client_ip: Remote client IP for logging context.
        username: Authenticated username for logging context.

    Returns:
        The total number of frames sent before the loop terminates.
    """
    frame_count = 0
    session_start: float = start_session_timer()
    while True:
        if await check_and_maybe_close_on_timeout(
            websocket,
            session_start,
            f"[WebSocket-Labels] {client_ip} ({username})",
        ):
            break
        try:
            if not _is_websocket_connected(websocket):
                print(
                    (
                        f"[WebSocket-Labels] {client_ip} ({username}): "
                        'Connection lost, exiting loop'
                    ),
                )
                break

            # Fetch any frames newer than the recorded last IDs, with a
            # reasonable timeout to avoid blocking the loop.
            _aw_frames: Awaitable[list[FrameOutData]] = cast(
                Awaitable[list[FrameOutData]],
                fetch_latest_frames(rds, last_ids),
            )
            updated_data: list[FrameOutData] = await asyncio.wait_for(
                _aw_frames, timeout=10.0,
            )
            if updated_data:
                sent = await send_updated_frames(
                    websocket, updated_data, last_ids, client_ip, username,
                )
                frame_count += sent
            # Small sleep to avoid hot-looping when no updates are present.
            await asyncio.sleep(0.1)
        except asyncio.TimeoutError:
            print(
                (
                    f"[WebSocket-Labels] {client_ip} ({username}): "
                    'Redis fetch timeout'
                ),
            )
            continue
        except Exception as e:
            print(
                (
                    f"[WebSocket-Labels] {client_ip} ({username}): "
                    f"Error in streaming loop: {e}"
                ),
            )
            break
    return frame_count


async def process_stream_action(
    websocket: WebSocket,
    rds: redis.Redis,
    redis_key: str,
    data: dict[str, str | int],
    last_id: str,
    client_ip: str,
    username: str,
) -> tuple[bool, str]:
    """Handle a single action message for the per-stream WebSocket.

    Supports ``ping`` (responds with ``pong``) and ``pull`` (retrieve the next
    frame newer than ``last_id`` for ``redis_key``). Any other action returns
    an ``unknown action`` error message.

    Args:
        websocket: The active WebSocket connection.
        rds: Redis client used to fetch frame data.
        redis_key: Fully-qualified Redis stream key for the specific stream.
        data: Parsed action payload, typically including an ``action`` field.
        last_id: The last delivered Redis stream ID for this key.
        client_ip: Remote client IP for logging context.
        username: Authenticated username for logging context.

    Returns:
        A tuple ``(continue, new_last_id)``. If ``continue`` is ``False``, the
        caller should terminate the loop.
    """
    if data.get('action') == 'ping':
        # Lightweight liveness check.
        await _safe_websocket_send_text(
            websocket,
            json.dumps({'action': 'pong'}),
            f"{client_ip} ({username})",
        )
        return True, last_id

    if data.get('action') == 'pull':
        try:
            # Request the next available frame (if any) newer than last_id.
            _aw_frame: Awaitable[FrameOutData | None] = cast(
                Awaitable[FrameOutData | None],
                fetch_latest_frame_for_key(rds, redis_key, last_id),
            )
            frame_data: FrameOutData | None = await asyncio.wait_for(
                _aw_frame,
                timeout=10.0,
            )
            if frame_data:
                last_id = frame_data['id']
                if not _is_websocket_connected(websocket):
                    print(
                        (
                            f"[WebSocket-Stream] {client_ip} ({username}): "
                            'Connection closed, stopping'
                        ),
                    )
                    return False, last_id
                # Compose metadata header for the client.
                header: bytes = json.dumps(
                    {
                        'id': frame_data['id'],
                        'warnings': frame_data['warnings'],
                        'cone_polygons': frame_data['cone_polygons'],
                        'pole_polygons': frame_data['pole_polygons'],
                        'detection_items': frame_data['detection_items'],
                        'width': frame_data['width'],
                        'height': frame_data['height'],
                    },
                ).encode('utf-8')
                # Retrieve raw bytes; if absent, treat as empty and skip send.
                raw_bytes_data = frame_data.get('frame_bytes') or b''
                raw_bytes: bytes = (
                    cast(bytes, raw_bytes_data) if raw_bytes_data else b''
                )

                if not _is_websocket_connected(websocket):
                    print(
                        (
                            f"[WebSocket-Stream] {client_ip} ({username}): "
                            'Connection closed during frame preparation'
                        ),
                    )
                    return False, last_id
                # Send the combined header + payload as one binary message.
                success = await _safe_websocket_send_bytes(
                    websocket,
                    header + DELIMITER + raw_bytes,
                    f"{client_ip} ({username})",
                )
                if not success:
                    print(
                        (
                            f"[WebSocket-Stream] {client_ip} ({username}): "
                            'Failed to send frame'
                        ),
                    )
                    return False, last_id
        except asyncio.TimeoutError:
            print(
                (
                    f"[WebSocket-Stream] {client_ip} ({username}): "
                    'Frame fetch timeout'
                ),
            )
            await _safe_websocket_send_text(
                websocket,
                json.dumps({'error': 'Frame fetch timeout'}),
                f"{client_ip} ({username})",
            )
        return True, last_id

    await _safe_websocket_send_text(
        websocket,
        json.dumps({'error': 'unknown action'}),
        f"{client_ip} ({username})",
    )
    return True, last_id


async def store_frame_from_bytes(
    websocket: WebSocket,
    rds: redis.Redis,
    data: bytes,
    client_ip: str,
    username: str,
) -> None:
    """Persist a frame into Redis extracted from a combined binary message.

    The inbound binary message is expected to be the concatenation of a JSON
    header (UTF-8), a delimiter, and the raw frame bytes.

    Args:
        websocket: The active WebSocket connection.
        rds: Redis client used to store the frame.
        data: The raw bytes payload received from the WebSocket.
        client_ip: Remote client IP for logging context.
        username: Authenticated username for logging context.

    Returns:
        None. Sends an acknowledgement JSON on success.

    Raises:
        ValueError: If the input data does not contain the delimiter.
    """
    # Split header and payload; ValueError if delimiter is absent, which the
    # caller handles and reports gracefully.
    header_bytes, frame_bytes = data.split(DELIMITER, 1)
    header: FrameInHeader = json.loads(header_bytes.decode('utf-8'))

    label: str = str(header.get('label', '') or '')
    key: str = str(header.get('key', '') or '')
    warnings_json: str = str(header.get('warnings_json', '') or '')
    cone_polygons_json: str = str(
        header.get('cone_polygons_json', '') or '',
    )
    pole_polygons_json: str = str(
        header.get('pole_polygons_json', '') or '',
    )
    detection_items_json: str = str(
        header.get('detection_items_json', '') or '',
    )
    width: int = int(header.get('width', 0) or 0)
    height: int = int(header.get('height', 0) or 0)

    # Persist the frame and associated metadata to Redis.
    await store_to_redis(
        rds,
        label,
        key,
        frame_bytes,
        warnings_json,
        cone_polygons_json,
        pole_polygons_json,
        detection_items_json,
        width,
        height,
    )
    # Confirm success to the client.
    await _safe_websocket_send_json(
        websocket,
        {'status': 'ok', 'message': 'Frame stored successfully.'},
        f"{client_ip} ({username})",
    )


async def receive_text_with_timeout(
    websocket: WebSocket,
    client_ip: str,
    username: str,
    timeout: float = 60.0,
) -> str | None:
    """Receive a text message with a timeout and gentle error handling.

    Args:
        websocket: The active WebSocket connection.
        client_ip: Remote client IP for logging context.
        username: Authenticated username for logging context.
        timeout: Maximum time in seconds to wait for a message.

    Returns:
        The received text message, or ``None`` if the connection appears
        closed or a timeout occurs (in which case the socket is closed).
    """
    try:
        msg: str | None = await asyncio.wait_for(
            _safe_websocket_receive_text(
                websocket, f"{client_ip} ({username})",
            ),
            timeout=timeout,
        )
        if msg is None:
            print(
                (
                    f"[WebSocket-Stream] {client_ip} ({username}): "
                    'Failed to receive message, connection may be closed'
                ),
            )
        return msg
    except asyncio.TimeoutError:
        print(
            (
                f"[WebSocket-Stream] {client_ip} ({username}): "
                'Receive timeout after 60s'
            ),
        )
        await websocket.close(code=1000, reason='Receive timeout')
        return None


async def parse_and_process_action(
    msg: str,
    websocket: WebSocket,
    rds: redis.Redis,
    redis_key: str,
    last_id: str,
    client_ip: str,
    username: str,
) -> tuple[bool, str]:
    """Parse the raw message as JSON and delegate to action handling.

    Args:
        msg: The raw text received from the client.
        websocket: The active WebSocket connection.
        rds: Redis client used to fulfil actions.
        redis_key: The specific stream key for frame pulls.
        last_id: The last delivered stream ID for the key.
        client_ip: Remote client IP for logging context.
        username: Authenticated username for logging context.

    Returns:
        The result from :func:`process_stream_action`, or a continue signal on
        JSON errors after notifying the client.
    """
    try:
        data: dict[str, str | int] = json.loads(msg)
        return await process_stream_action(
            websocket,
            rds,
            redis_key,
            data,
            last_id,
            client_ip,
            username,
        )
    except json.JSONDecodeError as e:
        print(
            (
                f"[WebSocket-Stream] {client_ip} ({username}): "
                f"Invalid JSON: {e}"
            ),
        )
        await _safe_websocket_send_text(
            websocket,
            json.dumps({'error': 'Invalid JSON format'}),
            f"{client_ip} ({username})",
        )
        return True, last_id
    except Exception as e:
        print(
            (
                f"[WebSocket-Stream] {client_ip} ({username}): "
                f"Error processing action: {e}"
            ),
        )
        await _safe_websocket_send_text(
            websocket,
            json.dumps({'error': f'Action processing error: {str(e)}'}),
            f"{client_ip} ({username})",
        )
        return True, last_id


async def handle_label_stream_ws(
        websocket: WebSocket,
        label: str,
        rds: redis.Redis,
        settings: Settings,
) -> None:
    """WebSocket endpoint: stream frames for all keys under a label.

    Authenticates the client, verifies that streams exist for the specified
    ``label``, then enters a loop to push frames across all relevant keys.

    Args:
        websocket: The incoming WebSocket.
        label: The label grouping one or more underlying streams.
        rds: Redis client instance.
        settings: Application settings used for authentication.
    """
    client_ip = websocket.client.host if websocket.client else 'unknown'
    print(
        (
            f"[WebSocket-Labels] New connection from {client_ip} for label: "
            f"{label}"
        ),
    )
    await websocket.accept()
    username, _ = await authenticate_ws_or_none(
        websocket,
        rds,
        settings,
        auto_register_jti=AUTO_REGISTER_JTI,
        client_tag=f"[WebSocket-Labels] {client_ip}",
    )
    if not username:
        return
    print(
        f"[WebSocket-Labels] {client_ip}: Authenticated as {username}",
    )

    frame_count = 0
    try:
        # Resolve keys for the label and initialise per-key cursors.
        last_ids = await prepare_label_keys(
            websocket, rds, label, client_ip, username,
        )
        if last_ids is None:
            return
        # Enter the main streaming loop.
        frame_count = await label_stream_loop(
            websocket, rds, last_ids, client_ip, username,
        )
    except WebSocketDisconnect:
        print(
            (
                f"[WebSocket-Labels] {client_ip} ({username}): Client "
                f"disconnected after {frame_count} frames"
            ),
        )
    except Exception as e:
        print(
            (
                f"[WebSocket-Labels] {client_ip} ({username}): "
                f"Unexpected error: {e}"
            ),
        )
        try:
            await websocket.close(code=1011, reason='Internal server error')
        except Exception:
            pass
    finally:
        print(
            (
                f"[WebSocket-Labels] {client_ip} ({username}): Connection "
                f"closed, total frames: {frame_count}"
            ),
        )


async def handle_stream_ws(
        websocket: WebSocket,
        label: str,
        key: str,
        rds: redis.Redis,
        settings: Settings,
) -> None:
    """WebSocket endpoint: per-stream action-based communication.

    After authentication, the server listens for JSON messages from the client
    to perform actions such as ``ping`` and ``pull`` (retrieve a frame newer
    than the last delivered one) for the specified ``label``/``key`` stream.

    Args:
        websocket: The incoming WebSocket.
        label: The stream label (namespace or group).
        key: The specific stream key within the label.
        rds: Redis client instance.
        settings: Application settings used for authentication.
    """
    client_ip = websocket.client.host if websocket.client else 'unknown'
    print(
        (
            f"[WebSocket-Stream] New connection from {client_ip} for "
            f"{label}/{key}"
        ),
    )
    await websocket.accept()
    username, _ = await authenticate_ws_or_none(
        websocket,
        rds,
        settings,
        auto_register_jti=AUTO_REGISTER_JTI,
        client_tag=f"[WebSocket-Stream] {client_ip}",
    )
    if not username:
        return
    print(
        f"[WebSocket-Stream] {client_ip}: Authenticated as {username}",
    )
    redis_key: str = (
        f"stream_frame:{Utils.encode(label)}|{Utils.encode(key)}"
    )

    action_count = 0
    last_id: str = '0'
    session_start: float = start_session_timer()
    try:
        while True:
            if await check_and_maybe_close_on_timeout(
                    websocket,
                    session_start,
                    f"[WebSocket-Stream] {client_ip} ({username})",
                    use_text=True,
            ):
                break
            # Await the next client instruction; close if it times out.
            msg = await receive_text_with_timeout(
                websocket, client_ip, username, timeout=60.0,
            )
            if msg is None:
                break
            action_count += 1
            # Process the action and update our last seen ID if needed.
            cont, last_id = await parse_and_process_action(
                msg,
                websocket,
                rds,
                redis_key,
                last_id,
                client_ip,
                username,
            )
            if not cont:
                break
            # Periodic progress logging without being noisy.
            log_every_n(
                f"[WebSocket-Stream] {client_ip} ({username})",
                action_count,
                unit='actions',
                n=100,
            )
    finally:
        print(
            (
                f"[WebSocket-Stream] {client_ip} ({username}): "
                f'Connection closed, total actions: {action_count}'
            ),
        )


async def handle_frames_ws(
        websocket: WebSocket,
        rds: redis.Redis,
        settings: Settings,
) -> None:
    """WebSocket endpoint: accept and store inbound frames from clients.

    Clients send binary messages composed of a JSON header followed by a
    delimiter and the raw image bytes. The server persists these to Redis and
    acknowledges success with a small JSON response.

    Args:
        websocket: The incoming WebSocket.
        rds: Redis client used to persist frames.
        settings: Application settings used for authentication.
    """
    client_ip = websocket.client.host if websocket.client else 'unknown'
    print(f"[WebSocket] New connection from {client_ip}")
    await websocket.accept()
    username, _ = await authenticate_ws_or_none(
        websocket,
        rds,
        settings,
        auto_register_jti=AUTO_REGISTER_JTI,
        client_tag=f"[WebSocket] {client_ip}",
    )
    if not username:
        return
    print(f"[WebSocket] {client_ip}: Authenticated as {username}")
    session_start: float = start_session_timer()

    frame_count = 0
    try:
        while True:
            try:
                if await check_and_maybe_close_on_timeout(
                        websocket,
                        session_start,
                        f"[WebSocket] {client_ip} ({username})",
                ):
                    break
                # Receive a combined header+payload binary package.
                data: bytes | None = await asyncio.wait_for(
                    _safe_websocket_receive_bytes(
                        websocket, f"{client_ip} ({username})",
                    ),
                    timeout=60.0,
                )
                if data is None:
                    print(
                        (
                            f"[WebSocket] {client_ip} ({username}): "
                            'Failed to receive frame data, '
                            'connection may be closed'
                        ),
                    )
                    break
                # Persist the frame; errors are handled below.
                await store_frame_from_bytes(
                    websocket, rds, data, client_ip, username,
                )
                frame_count += 1
                log_every_n(
                    f"[WebSocket] {client_ip} ({username})",
                    frame_count,
                    unit='frames',
                    n=100,
                )
            except asyncio.TimeoutError:
                print(
                    (
                        f"[WebSocket] {client_ip} ({username}): "
                        f"Receive timeout after 60s"
                    ),
                )
                await websocket.close(code=1000, reason='Receive timeout')
                break
            except ValueError as e:
                print(
                    (
                        f"[WebSocket] {client_ip} ({username}): "
                        f"Data format error: {e}"
                    ),
                )
                await _safe_websocket_send_json(
                    websocket,
                    {
                        'status': 'error',
                        'message': f'Invalid data format: {str(e)}',
                    },
                    f"{client_ip} ({username})",
                )
            except Exception as e:
                print(
                    (
                        f"[WebSocket] {client_ip} ({username}): "
                        f"Frame processing error: {e}"
                    ),
                )
                await _safe_websocket_send_json(
                    websocket,
                    {
                        'status': 'error',
                        'message': f'Failed to store frame: {str(e)}',
                    },
                    f"{client_ip} ({username})",
                )
    finally:
        print(
            (
                f"[WebSocket] {client_ip} ({username}): "
                f'Connection closed, total frames: {frame_count}'
            ),
        )
