from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
from typing import cast

import redis.asyncio as redis
from fastapi import HTTPException
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from examples.auth.config import Settings
from examples.shared.ws_auth import WS_MAX_SESSION_SECONDS
from examples.shared.ws_helpers import authenticate_ws_or_none
from examples.shared.ws_helpers import check_and_maybe_close_on_timeout
from examples.shared.ws_helpers import start_session_timer
from examples.shared.ws_utils import _safe_websocket_receive_bytes
from examples.shared.ws_utils import _safe_websocket_send_json
from examples.YOLO_server_api.inference_runtime import InferenceRuntime
from examples.YOLO_server_api.inference_runtime import ModelSelection
from examples.YOLO_server_api.model_registry import select_local_model_async
from examples.YOLO_server_api.model_registry import select_model_async
from src.settings.inference import WebSocketSettings

# Limit WebSocket concurrency (consistent with original routers.py)

logger = logging.getLogger(__name__)


def _should_bypass_local_auth(client_ip: str) -> bool:
    """Allow trusted same-host pipeline calls to skip JWT auth."""
    if not WebSocketSettings().allow_localhost_bypass:
        return False
    try:
        return ipaddress.ip_address(client_ip).is_loopback
    except ValueError:
        return client_ip in {'localhost', 'unknown'}


async def _get_model_key_from_ws(
    websocket: WebSocket,
    client_ip: str,
    username: str,
) -> str | None:
    """Resolve the model key from headers, query parameters or the first
    message."""
    model_key: str | None = websocket.headers.get('x-model-key')
    if model_key:
        logger.info(
            (
                f'[YOLO-WebSocket] {client_ip} ({username}): '
                'Model key from header'
            ),
        )
        return model_key

    logger.info(
        (
            f'[YOLO-WebSocket] {client_ip} ({username}): '
            'Waiting for model key in first message'
        ),
    )
    try:
        first_message: str = await websocket.receive_text()
        config_data: dict[str, object] = json.loads(first_message)
        model_key = cast(str | None, config_data.get('model_key'))
        if not model_key:
            logger.info(
                (
                    f'[YOLO-WebSocket] {client_ip} ({username}): '
                    'No model_key found in first message'
                ),
            )
            await websocket.close(
                code=1008,
                reason='Missing model_key in configuration',
            )
            return None
        logger.info(
            (
                f'[YOLO-WebSocket] {client_ip} ({username}): '
                'Model key from first message'
            ),
        )
        return model_key
    except Exception as e:
        logger.info(
            (
                f'[YOLO-WebSocket] {client_ip} ({username}): '
                f'Failed to parse first message: {e}'
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
    """Send the ready message after a model has been selected."""
    config_response: dict[str, str] = {
        'status': 'ready',
        'model': model_key,
        'message': 'Model loaded successfully, ready to process images',
    }
    success: bool = await _safe_websocket_send_json(
        websocket,
        config_response,
        f'{client_ip} ({username})',
    )
    if not success:
        logger.info(
            (
                f'[YOLO-WebSocket] {client_ip} ({username}): '
                'Failed to send configuration response'
            ),
        )
    return success


async def _process_frame_and_respond(
    websocket: WebSocket,
    img_bytes: bytes,
    model_instance: ModelSelection,
    client_ip: str,
    username: str,
) -> bool:
    """Run detection for one frame and send the result."""
    try:
        result = await model_instance.detect(img_bytes)
        datas = result.rows
    except HTTPException as error:
        detail = error.detail if isinstance(error.detail, dict) else {}
        await _safe_websocket_send_json(
            websocket,
            {
                'status': 'error',
                'code': detail.get('code', 'DETECTION_FAILED'),
                'http_status': error.status_code,
            },
            f'{client_ip} ({username})',
        )
        await websocket.close(
            code={400: 1003, 413: 1009, 422: 1003}.get(
                error.status_code,
                1013,
            ),
            reason=str(detail.get('code', 'Detection unavailable')),
        )
        return False
    success = await _safe_websocket_send_json(
        websocket,
        datas,
        f'{client_ip} ({username})',
    )
    if not success:
        logger.info(
            (
                f'[YOLO-WebSocket] {client_ip} ({username}): '
                'Failed to send results, stopping'
            ),
        )
    return success


async def _load_session_model(websocket, model_key, client_ip, runtime):
    subject = websocket.scope.get('_catalog_subject')
    if isinstance(subject, dict):
        entry = await select_model_async(
            model_key,
            websocket.headers.get('x-model-version'),
            str(subject.get('tenant_id', '')),
            subject.get('role', ''),
            'image',
        )
    elif _should_bypass_local_auth(client_ip):
        entry = await select_local_model_async(
            model_key,
            websocket.headers.get('x-model-version'),
        )
    else:
        raise HTTPException(403, detail='Missing verified model scope')
    return await runtime.prepare(entry)


async def _prepare_model_and_notify(
    websocket: WebSocket,
    client_ip: str,
    username: str,
    runtime: InferenceRuntime,
) -> ModelSelection | None:
    """Load the requested model and notify the websocket client."""
    model_key: str | None = await _get_model_key_from_ws(
        websocket,
        client_ip,
        username,
    )
    if not model_key:
        return None
    try:
        model_instance = await _load_session_model(
            websocket,
            model_key,
            client_ip,
            runtime,
        )
    except HTTPException as exc:
        await websocket.close(
            code=1011 if exc.status_code >= 500 else 1008,
            reason='Model unavailable or access denied',
        )
        return None
    if model_instance is None:
        logger.info(
            (
                f'[YOLO-WebSocket] {client_ip} ({username}): '
                f'Model {model_key} not found'
            ),
        )
        await websocket.close(code=1003, reason='Model not found')
        return None
    logger.info(
        (
            f'[YOLO-WebSocket] {client_ip} ({username}): '
            f'Using model {model_key}'
        ),
    )
    if not await _send_ready_config(websocket, model_key, client_ip, username):
        return None
    return model_instance


async def _detect_loop(
    websocket: WebSocket,
    session_start: float,
    model_instance: ModelSelection,
    client_ip: str,
    username: str,
) -> int:
    """Receive websocket frames until timeout, close, or send failure."""
    frame_count: int = 0
    while True:
        if await check_and_maybe_close_on_timeout(
            websocket,
            session_start,
            f'[YOLO-WebSocket] {client_ip} ({username})',
        ):
            break
        img_bytes: bytes | None = await _safe_websocket_receive_bytes(
            websocket,
            f'{client_ip} ({username})',
        )
        if img_bytes is None:
            logger.info(
                (
                    f'[YOLO-WebSocket] {client_ip} ({username}): '
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
    return frame_count


async def handle_websocket_detect(
    websocket: WebSocket,
    rds: redis.Redis,
    settings: Settings,
    runtime: InferenceRuntime,
) -> None:
    """Handle one YOLO websocket detection session."""
    client_ip: str = websocket.client.host if websocket.client else 'unknown'
    logger.info(f'[YOLO-WebSocket] New connection from {client_ip}')

    session_start = start_session_timer()
    username = 'unauthenticated'
    try:
        async with asyncio.timeout(WS_MAX_SESSION_SECONDS):
            await websocket.accept()

            if _should_bypass_local_auth(client_ip):
                username = WebSocketSettings().local_username
                logger.info(
                    (
                        f'[YOLO-WebSocket] {client_ip}: '
                        'Localhost auth bypass enabled'
                    ),
                )
            else:
                authenticated_username, payload = (
                    await authenticate_ws_or_none(
                        websocket,
                        rds,
                        client_tag=f'[YOLO-WebSocket] {client_ip}',
                    )
                )
                if not authenticated_username:
                    return
                username = authenticated_username
                websocket.scope['_catalog_subject'] = (payload or {}).get(
                    'subject',
                )
                logger.info(
                    f'[YOLO-WebSocket] {client_ip}: '
                    f'Authenticated as {username}',
                )
            model_instance = await _prepare_model_and_notify(
                websocket,
                client_ip,
                username,
                runtime,
            )
            if model_instance is None:
                return
            await _detect_loop(
                websocket,
                session_start,
                model_instance,
                client_ip,
                username,
            )
    except TimeoutError:
        try:
            await websocket.close(code=1000, reason='Session timeout')
        except (WebSocketDisconnect, RuntimeError):
            pass
    except WebSocketDisconnect:
        logger.info(
            f'[YOLO-WebSocket] {client_ip} ({username}): Client disconnected',
        )
    except Exception as e:
        logger.info(
            (
                f'[YOLO-WebSocket] {client_ip} ({username}): '
                f'Unexpected error: {e}'
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
        logger.info(
            f'[YOLO-WebSocket] {client_ip} ({username}): Connection closed',
        )
