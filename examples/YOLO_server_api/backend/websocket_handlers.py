from __future__ import annotations

import asyncio
import json
from typing import cast

import redis
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from examples.auth.config import Settings
from examples.shared.ws_helpers import authenticate_ws_or_none
from examples.shared.ws_helpers import check_and_maybe_close_on_timeout
from examples.shared.ws_helpers import get_auto_register_jti
from examples.shared.ws_helpers import log_every_n
from examples.shared.ws_helpers import start_session_timer
from examples.shared.ws_utils import _safe_websocket_receive_bytes
from examples.shared.ws_utils import _safe_websocket_send_json
from examples.YOLO_server_api.backend.detection import run_detection_from_bytes
from examples.YOLO_server_api.backend.models import DetectionModelManager

# Limit WebSocket concurrency (consistent with original routers.py)
WS_INFERENCE_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(8)

# Control whether to automatically register JTI
# (inherited from shared settings)
AUTO_REGISTER_JTI: bool = get_auto_register_jti()


async def _get_model_key_from_ws(
        websocket: WebSocket,
        client_ip: str,
        username: str,
) -> str | None:
    """從 Header/Query/第一個訊息解析 model_key。"""
    model_key: str | None = websocket.headers.get('x-model-key')
    if model_key:
        print(
            (
                f"[YOLO-WebSocket] {client_ip} ({username}): "
                'Model key from header'
            ),
        )
        return model_key

    query_params = dict(websocket.query_params)
    model_key = query_params.get('model')
    if model_key:
        print(
            (
                f"[YOLO-WebSocket] {client_ip} ({username}): "
                'Model key from query parameter (deprecated)'
            ),
        )
        return model_key

    print(
        (
            f"[YOLO-WebSocket] {client_ip} ({username}): "
            'Waiting for model key in first message'
        ),
    )
    try:
        first_message: str = await websocket.receive_text()
        config_data: dict[str, object] = json.loads(first_message)
        model_key = cast(str | None, config_data.get('model_key'))
        if not model_key:
            print(
                (
                    f"[YOLO-WebSocket] {client_ip} ({username}): "
                    'No model_key found in first message'
                ),
            )
            await websocket.close(
                code=1008,
                reason='Missing model_key in configuration',
            )
            return None
        print(
            (
                f"[YOLO-WebSocket] {client_ip} ({username}): "
                'Model key from first message'
            ),
        )
        return model_key
    except Exception as e:
        print(
            (
                f"[YOLO-WebSocket] {client_ip} ({username}): "
                f"Failed to parse first message: {e}"
            ),
        )
        await websocket.close(
            code=1008,
            reason='Invalid configuration message',
        )
        return None


async def _send_ready_config(
        websocket: WebSocket,
        model_key: str,
        client_ip: str,
        username: str,
) -> bool:
    config_response: dict[str, str] = {
        'status': 'ready',
        'model': model_key,
        'message': 'Model loaded successfully, ready to process images',
    }
    success: bool = await _safe_websocket_send_json(
        websocket,
        config_response,
        f"{client_ip} ({username})",
    )
    if not success:
        print(
            (
                f"[YOLO-WebSocket] {client_ip} ({username}): "
                'Failed to send configuration response'
            ),
        )
    return success


async def _process_frame_and_respond(
        websocket: WebSocket,
        img_bytes: bytes,
        model_instance,
        client_ip: str,
        username: str,
) -> bool:
    datas, _ = await run_detection_from_bytes(
        img_bytes, model_instance, semaphore=WS_INFERENCE_SEMAPHORE,
    )
    success = await _safe_websocket_send_json(
        websocket, datas, f"{client_ip} ({username})",
    )
    if not success:
        print(
            (
                f"[YOLO-WebSocket] {client_ip} ({username}): "
                'Failed to send results, stopping'
            ),
        )
    return success


async def _prepare_model_and_notify(
    websocket: WebSocket,
    client_ip: str,
    username: str,
    model_loader: DetectionModelManager,
) -> object | None:
    model_key: str | None = await _get_model_key_from_ws(
        websocket,
        client_ip,
        username,
    )
    if not model_key:
        return None
    model_instance = model_loader.get_model(model_key)
    if model_instance is None:
        print(
            (
                f"[YOLO-WebSocket] {client_ip} ({username}): "
                f"Model {model_key} not found"
            ),
        )
        await websocket.close(code=1003, reason='Model not found')
        return None
    print(
        (
            f"[YOLO-WebSocket] {client_ip} ({username}): "
            f"Using model {model_key}"
        ),
    )
    if not await _send_ready_config(websocket, model_key, client_ip, username):
        return None
    return model_instance


async def _detect_loop(
        websocket: WebSocket,
        session_start: float,
        model_instance,
        client_ip: str,
        username: str,
) -> int:
    frame_count: int = 0
    while True:
        if await check_and_maybe_close_on_timeout(
                websocket,
                session_start,
                f"[YOLO-WebSocket] {client_ip} ({username})",
        ):
            break
        img_bytes: bytes | None = await _safe_websocket_receive_bytes(
            websocket,
            f"{client_ip} ({username})",
        )
        if img_bytes is None:
            print(
                (
                    f"[YOLO-WebSocket] {client_ip} ({username}): "
                    'Failed to receive image data, connection may be closed'
                ),
            )
            break
        frame_count += 1
        success = await _process_frame_and_respond(
            websocket,
            img_bytes,
            model_instance,
            client_ip,
            username,
        )
        if not success:
            break
        log_every_n(
            f"[YOLO-WebSocket] {client_ip} ({username})",
            frame_count,
            unit='frames',
            n=100,
        )
    return frame_count


async def handle_websocket_detect(
        websocket: WebSocket,
        rds: redis.Redis,
        settings: Settings,
        model_loader: DetectionModelManager,
) -> None:
    client_ip: str = websocket.client.host if websocket.client else 'unknown'
    print(f"[YOLO-WebSocket] New connection from {client_ip}")

    await websocket.accept()

    username, _ = await authenticate_ws_or_none(
        websocket,
        rds,
        settings,
        auto_register_jti=AUTO_REGISTER_JTI,
        client_tag=f"[YOLO-WebSocket] {client_ip}",
    )
    if not username:
        return
    print(f"[YOLO-WebSocket] {client_ip}: Authenticated as {username}")

    session_start: float = start_session_timer()

    model_instance = await _prepare_model_and_notify(
        websocket, client_ip, username, model_loader,
    )
    if model_instance is None:
        return

    try:
        await _detect_loop(
            websocket, session_start, model_instance, client_ip, username,
        )
    except WebSocketDisconnect:
        print(
            f"[YOLO-WebSocket] {client_ip} ({username}): Client disconnected",
        )
    except Exception as e:
        print(
            (
                f"[YOLO-WebSocket] {client_ip} ({username}): "
                f"Unexpected error: {e}"
            ),
        )
        try:
            await websocket.close(
                code=1011,
                reason='Internal server error',
            )
        except Exception:
            pass
    finally:
        print(f"[YOLO-WebSocket] {client_ip} ({username}): Connection closed")
