from __future__ import annotations

from fastapi import WebSocket


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
    if not _is_websocket_connected(websocket):
        if client_info:
            msg = (
                f"[WebSocket] {client_info}: Connection closed, "
                'skipping JSON send'
            )
            print(msg)
        return False
    try:
        await websocket.send_json(data)
        return True
    except Exception as e:
        if client_info:
            print(f"[WebSocket] {client_info}: Failed to send JSON: {e}")
        return False


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
    if not _is_websocket_connected(websocket):
        if client_info:
            msg = (
                f"[WebSocket] {client_info}: Connection closed, "
                'skipping text send'
            )
            print(msg)
        return False
    try:
        await websocket.send_text(text)
        return True
    except Exception as e:
        if client_info:
            print(f"[WebSocket] {client_info}: Failed to send text: {e}")
        return False


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
    if not _is_websocket_connected(websocket):
        if client_info:
            msg = (
                f"[WebSocket] {client_info}: Connection closed, "
                'skipping bytes send'
            )
            print(msg)
        return False
    try:
        await websocket.send_bytes(data)
        return True
    except Exception as e:
        if client_info:
            print(f"[WebSocket] {client_info}: Failed to send bytes: {e}")
        return False


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
    if not _is_websocket_connected(websocket):
        if client_info:
            msg = (
                f"[WebSocket] {client_info}: Connection closed, "
                'cannot receive text'
            )
            print(msg)
        return None
    try:
        return await websocket.receive_text()
    except Exception as e:
        if client_info:
            print(f"[WebSocket] {client_info}: Failed to receive text: {e}")
        return None


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
    if not _is_websocket_connected(websocket):
        if client_info:
            msg = (
                f"[WebSocket] {client_info}: Connection closed, "
                'cannot receive bytes'
            )
            print(msg)
        return None
    try:
        return await websocket.receive_bytes()
    except Exception as e:
        if client_info:
            print(f"[WebSocket] {client_info}: Failed to receive bytes: {e}")
        return None
