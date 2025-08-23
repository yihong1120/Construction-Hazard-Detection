from __future__ import annotations

from fastapi import WebSocket


def _is_websocket_connected(websocket: WebSocket) -> bool:
    """
    Check whether the WebSocket connection is still valid.

    Args:
        websocket (WebSocket): The WebSocket instance to check.

    Returns:
        bool: True if the WebSocket connection is valid,
            otherwise False.
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
    """
    Safely send JSON data over WebSocket if connected.

    Args:
        websocket (WebSocket): The WebSocket instance to send data through.
        data (dict | list | str | int | float | bool | None): The
            JSON-serialisable data to send.
        client_info (str, optional): Additional information about the client.
            Defaults to ''.

    Returns:
        bool: True if the data was sent successfully,
            otherwise False.
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
            print(
                f"[WebSocket] {client_info}: Failed to send "
                f"JSON: {e}",
            )
        return False


async def _safe_websocket_send_text(
    websocket: WebSocket,
    text: str,
    client_info: str = '',
) -> bool:
    """
    Safely send text data over WebSocket if connected.

    Args:
        websocket (WebSocket): The WebSocket instance to send data through.
        text (str): The text data to send.
        client_info (str, optional): Additional information about the client.
            Defaults to ''.

    Returns:
        bool: True if the data was sent successfully,
            otherwise False.
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
            print(
                f"[WebSocket] {client_info}: Failed to send "
                f"text: {e}",
            )
        return False


async def _safe_websocket_send_bytes(
    websocket: WebSocket,
    data: bytes,
    client_info: str = '',
) -> bool:
    """
    Safely send binary data over WebSocket if connected.

    Args:
        websocket (WebSocket): The WebSocket instance to send data through.
        data (bytes): The binary data to send.
        client_info (str, optional): Additional information about the client.
            Defaults to ''.

    Returns:
        bool: True if the data was sent successfully,
            otherwise False.
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
            print(
                f"[WebSocket] {client_info}: Failed to send "
                f"bytes: {e}",
            )
        return False


async def _safe_websocket_receive_text(
    websocket: WebSocket,
    client_info: str = '',
) -> str | None:
    """
    Safely receive text data from WebSocket if connected.

    Args:
        websocket (WebSocket): The WebSocket instance to receive data from.
        client_info (str, optional): Additional information about the client.
            Defaults to ''.

    Returns:
        str | None: The received text data, or None if the operation
            failed.
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
            print(
                f"[WebSocket] {client_info}: Failed to receive "
                f"text: {e}",
            )
        return None


async def _safe_websocket_receive_bytes(
    websocket: WebSocket,
    client_info: str = '',
) -> bytes | None:
    """
    Safely receive binary data from WebSocket if connected.

    Args:
        websocket (WebSocket): The WebSocket instance to receive data from.
        client_info (str, optional): Additional information about the client.
            Defaults to ''.

    Returns:
        bytes | None: The received binary data, or None if the operation
            failed.
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
            print(
                f"[WebSocket] {client_info}: Failed to receive ",
                f"bytes: {e}",
            )
        return None
