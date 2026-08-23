from __future__ import annotations

import logging
from collections.abc import Awaitable
from collections.abc import Callable
from typing import TypeVar

from fastapi import WebSocket
from fastapi import WebSocketDisconnect


_WebSocketResult = TypeVar('_WebSocketResult')
logger = logging.getLogger(__name__)


def _is_expected_websocket_close_error(exc: Exception) -> bool:
    """Return True for routine disconnect races during WebSocket I/O."""
    if isinstance(exc, WebSocketDisconnect):
        return True

    message = str(exc)
    return (
        'close message has been sent' in message
        or 'Cannot call "send" once a close message has been sent' in message
        or (
            'Cannot call "receive" once a disconnect message '
            'has been received' in message
        )
        or 'Unexpected ASGI message' in message
    )


def _is_websocket_connected(websocket: WebSocket) -> bool:
    """Check whether the WebSocket connection still appears valid.

    Args:
        websocket: The WebSocket instance to check.

    Returns:
        ``True`` when the connection looks healthy, ``False`` otherwise.
    """
    try:
        # Directly access attributes, handle AttributeError if missing
        if websocket.client_state.value != 1:
            return False
        if not websocket.client:
            return False
        return True
    except Exception:
        return False


async def _safe_websocket_send_json(
    websocket: WebSocket,
    data: dict | list | str | int | float | bool | None,
    client_info: str = '',
) -> bool:
    """Safely send JSON data if the WebSocket is connected.

    Args:
        websocket: The WebSocket instance to send data through.
        data: JSON-serialisable data to send.
        client_info: Optional tag included in log messages.

    Returns:
        ``True`` if the data was sent successfully; ``False`` otherwise.
    """
    sent, _ = await _safe_websocket_operation(
        websocket,
        lambda: websocket.send_json(data),
        client_info=client_info,
        disconnected_message='Connection closed, skipping JSON send',
        operation_name='send JSON',
    )
    return sent


async def _safe_websocket_send_text(
    websocket: WebSocket,
    text: str,
    client_info: str = '',
) -> bool:
    """Safely send text data if the WebSocket is connected.

    Args:
        websocket: The WebSocket instance to send data through.
        text: The text payload to send.
        client_info: Optional tag included in log messages.

    Returns:
        ``True`` if the data was sent successfully; ``False`` otherwise.
    """
    sent, _ = await _safe_websocket_operation(
        websocket,
        lambda: websocket.send_text(text),
        client_info=client_info,
        disconnected_message='Connection closed, skipping text send',
        operation_name='send text',
    )
    return sent


async def _safe_websocket_send_bytes(
    websocket: WebSocket,
    data: bytes,
    client_info: str = '',
) -> bool:
    """Safely send binary data if the WebSocket is connected.

    Args:
        websocket: The WebSocket instance to send data through.
        data: The binary payload to send.
        client_info: Optional tag included in log messages.

    Returns:
        ``True`` if the data was sent successfully; ``False`` otherwise.
    """
    sent, _ = await _safe_websocket_operation(
        websocket,
        lambda: websocket.send_bytes(data),
        client_info=client_info,
        disconnected_message='Connection closed, skipping bytes send',
        operation_name='send bytes',
    )
    return sent


async def _safe_websocket_receive_text(
    websocket: WebSocket,
    client_info: str = '',
) -> str | None:
    """Safely receive text data if the WebSocket is connected.

    Args:
        websocket: The WebSocket instance to receive data from.
        client_info: Optional tag included in log messages.

    Returns:
        The received text data, or ``None`` if the operation failed.
    """
    _, text = await _safe_websocket_operation(
        websocket,
        websocket.receive_text,
        client_info=client_info,
        disconnected_message='Connection closed, cannot receive text',
        operation_name='receive text',
    )
    return text


async def _safe_websocket_receive_bytes(
    websocket: WebSocket,
    client_info: str = '',
) -> bytes | None:
    """Safely receive binary data if the WebSocket is connected.

    Args:
        websocket: The WebSocket instance to receive data from.
        client_info: Optional tag included in log messages.

    Returns:
        The received binary data, or ``None`` if the operation failed.
    """
    _, data = await _safe_websocket_operation(
        websocket,
        websocket.receive_bytes,
        client_info=client_info,
        disconnected_message='Connection closed, cannot receive bytes',
        operation_name='receive bytes',
    )
    return data


async def _safe_websocket_operation(
    websocket: WebSocket,
    operation: Callable[[], Awaitable[_WebSocketResult]],
    *,
    client_info: str,
    disconnected_message: str,
    operation_name: str,
) -> tuple[bool, _WebSocketResult | None]:
    """Run one WebSocket operation with consistent close-race handling."""
    if not _is_websocket_connected(websocket):
        if client_info:
            logger.debug(
                'WebSocket disconnected client=%s reason=%s',
                client_info,
                disconnected_message,
            )
        return False, None
    try:
        return True, await operation()
    except Exception as exc:
        if client_info and not _is_expected_websocket_close_error(exc):
            logger.warning(
                'WebSocket operation failed client=%s operation=%s '
                'error_type=%s',
                client_info,
                operation_name,
                type(exc).__name__,
            )
        return False, None
