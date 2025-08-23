from __future__ import annotations

import asyncio
import json
from typing import cast

import jwt
import redis
from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi_jwt import JwtAuthorizationCredentials
from fastapi_limiter.depends import RateLimiter
from jwt import InvalidTokenError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from examples.auth.cache import get_user_data
from examples.auth.config import Settings
from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.models import User
from examples.auth.redis_pool import get_redis_pool
from examples.auth.redis_pool import get_redis_pool_ws
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
from examples.streaming_web.backend.redis_service import scan_for_labels
from examples.streaming_web.backend.redis_service import store_to_redis
from examples.streaming_web.backend.schemas import FramePostResponse
from examples.streaming_web.backend.schemas import LabelListResponse
from examples.streaming_web.backend.utils import Utils

# Global application settings instance
settings: Settings = Settings()

# Create a FastAPI router instance to encapsulate endpoints
router: APIRouter = APIRouter()

# Rate limiters to control endpoint access frequency
rate_limiter_index: RateLimiter = RateLimiter(times=60, seconds=60)
rate_limiter_label: RateLimiter = RateLimiter(times=600, seconds=60)


@router.get(
    '/labels',
    response_model=LabelListResponse,
    dependencies=[Depends(rate_limiter_index)],
)
async def get_labels_route(
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> LabelListResponse:
    """
    Fetch a list of labels accessible to the authenticated user.

    Args:
        credentials: JWT authorisation credentials containing user information.
        db: Asynchronous database session for user queries.
        rds: Redis connection pool for label retrieval.

    Returns:
        LabelListResponse containing filtered list of accessible labels.

    Raises:
        HTTPException:
        When token is invalid, user not found, or Redis operation fails.
    """
    try:
        # Extract username from JWT subject
        username: str | None = credentials.subject.get('username')
        if not username:
            raise HTTPException(
                status_code=401, detail='Invalid token: no subject',
            )

        # Query database for user with preloaded site relationships
        stmt = (
            select(User)
            .where(User.username == username)
            # Eagerly load site relationships
            .options(selectinload(User.sites))
        )
        result = await db.execute(stmt)
        user: User | None = result.scalars().first()
        if not user:
            raise HTTPException(status_code=401, detail='Invalid user')

        # Extract user role for permission checking
        user_role: str = user.role

        # Build list of site names the user has access to
        user_site_names: list[str] = [site.name for site in user.sites]
        print(f"User {username} has access to sites: {user_site_names}")

        # Fetch all available labels from Redis
        all_labels: list[str] = await scan_for_labels(rds)
        print(f"All labels in Redis: {all_labels}")

        # Filter labels based on user permissions
        if user_role == 'admin':
            # Admins can access all labels
            filtered_labels: list[str] = all_labels
        else:
            # Regular users only get labels for their accessible sites
            filtered_labels = [
                lbl for lbl in all_labels if lbl in user_site_names
            ]

        return LabelListResponse(labels=filtered_labels)

    except Exception as e:
        print(f"Failed to fetch labels: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch labels: {str(e)}",
        )


@router.post(
    '/frames',
    response_model=FramePostResponse,
    dependencies=[Depends(jwt_access)],
)
async def post_frame(
    label: str = Form(...),
    key: str = Form(...),
    file: UploadFile = File(...),
    warnings_json: str = Form(''),
    cone_polygons_json: str = Form(''),
    pole_polygons_json: str = Form(''),
    detection_items_json: str = Form(''),
    width: int = Form(0),
    height: int = Form(0),
    rds: redis.Redis = Depends(get_redis_pool),
) -> FramePostResponse:
    """
    Store a video frame with associated metadata in Redis.

    Args:
        label: Site or camera label to group frames under.
        key: Unique identifier for the specific video feed or stream.
        file: Uploaded image file containing the video frame.
        warnings_json: JSON string containing warning messages (optional).
        cone_polygons_json:
            JSON string containing traffic cone polygon data (optional).
        pole_polygons_json:
            JSON string containing utility pole polygon data (optional).
        detection_items_json:
            JSON string containing detected object data (optional).
        width: Frame width in pixels (defaults to 0).
        height: Frame height in pixels (defaults to 0).
        rds: Redis connection pool for frame storage.

    Returns:
        FramePostResponse indicating storage success or failure.

    Raises:
        HTTPException: When frame storage in Redis fails.
    """
    try:
        # Read uploaded file content asynchronously
        frame_bytes: bytes = await file.read()

        # Store frame and associated metadata in Redis
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

        return FramePostResponse(
            status='ok',
            message='Frame stored successfully.',
        )
    except Exception as e:
        print(f"Failed to store frame: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store frame: {str(e)}",
        )


@router.websocket('/ws/labels/{label}')
async def websocket_label_stream(
    websocket: WebSocket,
    label: str,
    rds: redis.Redis = Depends(get_redis_pool_ws),
) -> None:
    client_ip = websocket.client.host if websocket.client else 'unknown'
    print(
        (
            f"[WebSocket-Labels] New connection from {client_ip} "
            f"for label: {label}"
        ),
    )

    # Accept incoming WebSocket connection
    await websocket.accept()

    # Extract JWT token from Authorization header or query parameter
    token: str | None = None

    # First try to get token from Authorization header (移動端/桌面端)
    auth: str | None = websocket.headers.get('authorization')
    if auth and auth.lower().startswith('bearer '):
        token = auth.split(' ', 1)[1]
        print(
            f"[WebSocket-Labels] {client_ip}: Token from Authorization header",
        )

    # If no token from header, try query parameter (Web 平台)
    if not token:
        query_params = dict(websocket.query_params)
        token = query_params.get('token')
        if token:
            print(
                f"[WebSocket-Labels] {client_ip}: Token from query parameter",
            )

    # If still no token found, close connection
    if not token:
        print(
            (
                f"[WebSocket-Labels] {client_ip}: No token found in header "
                'or query parameter'
            ),
        )
        await websocket.close(code=1008, reason='Missing authentication token')
        return

    # Verify JWT token using authentication module configuration
    try:
        payload: dict = jwt.decode(
            token, settings.authjwt_secret_key,
            algorithms=[settings.ALGORITHM],
        )
    except InvalidTokenError as e:
        print(f"[WebSocket-Labels] {client_ip}: Invalid JWT token: {e}")
        await websocket.close(code=1008, reason='Invalid token')
        return

    # Ensure JWT payload contains required data
    if not payload:
        print(f"[WebSocket-Labels] {client_ip}: Empty JWT payload")
        await websocket.close(code=1008, reason='Empty token payload')
        return

    # Extract username and JTI (JWT ID) from token payload
    subject_data = payload.get('subject', {})
    username: str | None = (
        subject_data.get('username') if subject_data else None
    ) or payload.get('username')
    jti: str | None = (
        subject_data.get('jti') if subject_data else None
    ) or payload.get('jti')

    # Validate that both username and JTI are present
    if not username or not jti:
        print(
            (
                f"[WebSocket-Labels] {client_ip}: Missing username or "
                'JTI in token'
            ),
        )
        await websocket.close(code=1008, reason='Invalid token data')
        return

    # Verify JTI against cached user data in Redis
    user_data: dict[str, str | list[str]] | None = await get_user_data(
        cast(redis.asyncio.Redis, rds), username,
    )

    # Ensure user data exists and contains the JTI list
    if (
        not user_data
        or 'jti_list' not in user_data
        or jti not in user_data['jti_list']
    ):
        print(
            (
                f"[WebSocket-Labels] {client_ip}: JTI not found in user "
                f"active tokens for {username}"
            ),
        )
        await websocket.close(code=1008, reason='Token not active')
        return

    print(f"[WebSocket-Labels] {client_ip}: Authenticated as {username}")
    frame_count = 0

    try:
        # Fetch all stream keys associated with the provided label
        keys: list[str] = await get_keys_for_label(rds, label)
        if not keys:
            # Notify client that no streams exist for this label
            print(
                (
                    f"[WebSocket-Labels] {client_ip} ({username}): No keys "
                    f"found for label '{label}'"
                ),
            )
            await _safe_websocket_send_json(
                websocket, {
                    'error': f"No keys found for label '{label}'",
                }, f"{client_ip} ({username})",
            )
            await websocket.close(code=1000, reason='No streams available')
            return

        # Track last processed message ID for each stream to prevent duplicates
        last_ids: dict[str, str] = {k: '0' for k in keys}
        print(
            (
                f"[WebSocket-Labels] {client_ip} ({username}): Found "
                f"{len(keys)} streams for label '{label}'"
            ),
        )

        # Main streaming loop with timeout protection
        while True:
            try:
                # Check if connection is still open before processing
                if not _is_websocket_connected(websocket):
                    print(
                        (
                            f"[WebSocket-Labels] {client_ip} ({username}): "
                            'Connection lost, exiting loop'
                        ),
                    )
                    break

                # Poll Redis for new frames across all streams with timeout
                updated_data: list[dict[str, str]] = await asyncio.wait_for(
                    fetch_latest_frames(rds, last_ids),
                    timeout=10.0,
                )

                # If new frames are available, send them to the client
                if updated_data:
                    for data in updated_data:
                        try:
                            # Check if connection is still open before sending
                            if not _is_websocket_connected(websocket):
                                print(
                                    (
                                        f"[WebSocket-Labels] {client_ip} "
                                        f"({username}): Connection closed, "
                                        'stopping frame sending'
                                    ),
                                )
                                return

                            # Construct JSON metadata header for frame
                            header: bytes = json.dumps({
                                'key': data.get('key', ''),
                                'warnings': data.get('warnings', ''),
                                'cone_polygons': data.get(
                                    'cone_polygons', '',
                                ),
                                'pole_polygons': data.get(
                                    'pole_polygons', '',
                                ),
                                'detection_items': data.get(
                                    'detection_items', '',
                                ),
                                'width': data.get('width', 0),
                                'height': data.get('height', 0),
                            }).encode('utf-8')

                            # Extract frame bytes from Redis data
                            frame_bytes_raw = data.get('frame_bytes', b'')
                            frame_bytes: bytes = cast(
                                bytes, frame_bytes_raw,
                            ) if frame_bytes_raw else b''

                            if frame_bytes:
                                # Update last_id to track processed frames
                                key = data.get('key', '')
                                if key and 'id' in data:
                                    last_ids[key] = data['id']

                                # Combine header and frame data with delimiter
                                message_bytes: bytes = (
                                    header + DELIMITER + frame_bytes
                                )

                                # Double-check connection state before sending
                                if not _is_websocket_connected(websocket):
                                    print(
                                        (
                                            f"[WebSocket-Labels] {client_ip} "
                                            f"({username}): Connection closed "
                                            'during frame preparation'
                                        ),
                                    )
                                    return

                                # Send binary frame data with timeout
                                success = await _safe_websocket_send_bytes(
                                    websocket,
                                    message_bytes,
                                    f"{client_ip} ({username})",
                                )
                                if success:
                                    frame_count += 1
                                else:
                                    print(
                                        (
                                            f"[WebSocket-Labels] {client_ip} "
                                            f"({username}): Failed to send "
                                            'frame, stopping'
                                        ),
                                    )
                                    return

                        except asyncio.TimeoutError:
                            print(
                                (
                                    f"[WebSocket-Labels] {client_ip} "
                                    f"({username}): Send timeout"
                                ),
                            )
                            break
                        except Exception as e:
                            print(
                                (
                                    f"[WebSocket-Labels] {client_ip} "
                                    f"({username}): Error sending frame: {e}"
                                ),
                            )
                            break

                # Brief pause to prevent excessive Redis polling
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
            pass  # Connection might already be closed
    finally:
        print(
            (
                f"[WebSocket-Labels] {client_ip} ({username}): Connection "
                f"closed, total frames: {frame_count}"
            ),
        )


@router.websocket('/ws/stream/{label}/{key}')
async def websocket_stream(
    websocket: WebSocket,
    label: str,
    key: str,
    rds: redis.Redis = Depends(get_redis_pool_ws),
) -> None:
    client_ip = websocket.client.host if websocket.client else 'unknown'
    print(
        (
            f"[WebSocket-Stream] New connection from {client_ip} for "
            f"{label}/{key}"
        ),
    )

    # Accept incoming WebSocket connection
    await websocket.accept()

    # Extract JWT token from Authorization header or query parameter
    token: str | None = None

    # First try to get token from Authorization header (移動端/桌面端)
    auth: str | None = websocket.headers.get('authorization')
    if auth and auth.lower().startswith('bearer '):
        token = auth.split(' ', 1)[1]
        print(
            f"[WebSocket-Stream] {client_ip}: Token from Authorization header",
        )

    # If no token from header, try query parameter (Web 平台)
    if not token:
        query_params = dict(websocket.query_params)
        token = query_params.get('token')
        if token:
            print(
                f"[WebSocket-Stream] {client_ip}: Token from query parameter",
            )

    # If still no token found, close connection
    if not token:
        print(
            (
                f"[WebSocket-Stream] {client_ip}: No token found in header "
                'or query parameter'
            ),
        )
        await websocket.close(code=1008, reason='Missing authentication token')
        return

    # Verify JWT token using authentication module configuration
    try:
        payload: dict = jwt.decode(
            token, settings.authjwt_secret_key,
            algorithms=[settings.ALGORITHM],
        )
    except InvalidTokenError as e:
        print(f"[WebSocket-Stream] {client_ip}: Invalid JWT token: {e}")
        await websocket.close(code=1008, reason='Invalid token')
        return

    # Extract username and JTI (JWT ID) from token payload
    subject_data = payload.get('subject', {})
    username: str | None = (
        subject_data.get('username') if subject_data else None
    ) or payload.get('username')
    jti: str | None = (
        subject_data.get('jti') if subject_data else None
    ) or payload.get('jti')

    # Validate that both username and JTI are present
    if not username or not jti:
        print(
            (
                f"[WebSocket-Stream] {client_ip}: Missing username or "
                'JTI in token'
            ),
        )
        await websocket.close(code=1008, reason='Invalid token data')
        return

    # Verify JTI against cached user data in Redis
    user_data: dict[str, str | list[str]] | None = await get_user_data(
        cast(redis.asyncio.Redis, rds), username,
    )

    # Ensure user data exists and contains the JTI list
    if (
        not user_data
        or 'jti_list' not in user_data
        or jti not in user_data['jti_list']
    ):
        print(
            (
                f"[WebSocket-Stream] {client_ip}: JTI not found in user "
                f"active tokens for {username}"
            ),
        )
        await websocket.close(code=1008, reason='Token not active')
        return

    print(f"[WebSocket-Stream] {client_ip}: Authenticated as {username}")

    # Construct Redis key for frame storage using encoded label and key
    redis_key: str = f"stream_frame:{Utils.encode(label)}|{Utils.encode(key)}"

    # Track the last retrieved frame ID to fetch only new frames
    last_id: str = '0'
    action_count = 0

    try:
        # Main client interaction loop
        while True:
            try:
                # Receive client message in JSON text format with timeout
                msg: str | None = await asyncio.wait_for(
                    _safe_websocket_receive_text(
                        websocket, f"{client_ip} ({username})",
                    ),
                    timeout=60.0,  # 60秒超時
                )

                # Check if receive failed due to connection issues
                if msg is None:
                    print(
                        (
                            f"[WebSocket-Stream] {client_ip} ({username}): "
                            'Failed to receive message, connection may be '
                            'closed'
                        ),
                    )
                    break

                data: dict[str, str | int] = json.loads(msg)
                action_count += 1

                # Handle different client actions
                if data.get('action') == 'ping':
                    # Respond to keepalive ping with pong
                    await _safe_websocket_send_text(
                        websocket,
                        json.dumps({'action': 'pong'}),
                        f"{client_ip} ({username})",
                    )

                elif data.get('action') == 'pull':
                    try:
                        # Fetch latest frame data from Redis for this stream
                        frame_data: dict[str, str] | None = (
                            await asyncio.wait_for(
                                fetch_latest_frame_for_key(
                                    rds, redis_key, last_id,
                                ),
                                timeout=10.0,
                            )
                        )

                        if frame_data:
                            # Update last_id to avoid re-fetching same frame
                            last_id = frame_data['id']

                            # Check if connection is still open before sending
                            if not _is_websocket_connected(websocket):
                                print(
                                    (
                                        f"[WebSocket-Stream] {client_ip} "
                                        f"({username}): Connection closed, "
                                        'stopping'
                                    ),
                                )
                                return

                            # Build JSON header containing frame metadata
                            header: bytes = json.dumps({
                                'id': frame_data['id'],
                                'warnings': frame_data['warnings'],
                                'cone_polygons': frame_data['cone_polygons'],
                                'pole_polygons': frame_data['pole_polygons'],
                                'detection_items': frame_data[
                                    'detection_items'
                                ],
                                'width': frame_data['width'],
                                'height': frame_data['height'],
                            }).encode('utf-8')

                            # Extract raw frame bytes from Redis data
                            raw_bytes_data = (
                                frame_data.get('frame_bytes') or b''
                            )
                            raw_bytes: bytes = cast(
                                bytes, raw_bytes_data,
                            ) if raw_bytes_data else b''

                            # Double-check connection state before sending
                            if not _is_websocket_connected(websocket):
                                print(
                                    (
                                        f"[WebSocket-Stream] {client_ip} "
                                        f"({username}): Connection closed "
                                        'during frame preparation'
                                    ),
                                )
                                return

                            # Send combined header and frame data to client
                            success = await _safe_websocket_send_bytes(
                                websocket,
                                header + DELIMITER + raw_bytes,
                                f"{client_ip} ({username})",
                            )
                            if not success:
                                print(
                                    (
                                        f"[WebSocket-Stream] {client_ip} "
                                        f"({username}): Failed to send frame"
                                    ),
                                )
                                return

                    except asyncio.TimeoutError:
                        print(
                            (
                                f"[WebSocket-Stream] {client_ip} "
                                f"({username}): Frame fetch timeout"
                            ),
                        )
                        await _safe_websocket_send_text(
                            websocket,
                            json.dumps({'error': 'Frame fetch timeout'}),
                            f"{client_ip} ({username})",
                        )

                else:
                    # Handle unknown actions with error response
                    await _safe_websocket_send_text(
                        websocket,
                        json.dumps({'error': 'unknown action'}),
                        f"{client_ip} ({username})",
                    )

                # 每處理100個動作打印一次統計信息
                if action_count % 100 == 0:
                    print(
                        (
                            f"[WebSocket-Stream] {client_ip} ({username}): "
                            f"Processed {action_count} actions"
                        ),
                    )

            except asyncio.TimeoutError:
                print(
                    (
                        f"[WebSocket-Stream] {client_ip} ({username}): "
                        'Receive timeout after 60s'
                    ),
                )
                await websocket.close(code=1000, reason='Receive timeout')
                break

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

            except Exception as e:
                print(
                    (
                        f"[WebSocket-Stream] {client_ip} ({username}): "
                        f"Error processing action: {e}"
                    ),
                )
                await _safe_websocket_send_text(
                    websocket,
                    json.dumps(
                        {'error': f'Action processing error: {str(e)}'},
                    ),
                    f"{client_ip} ({username})",
                )

    except WebSocketDisconnect:
        print(
            (
                f"[WebSocket-Stream] {client_ip} ({username}): Client "
                f"disconnected after {action_count} actions"
            ),
        )
    except Exception as e:
        print(
            (
                f"[WebSocket-Stream] {client_ip} ({username}): "
                f"Unexpected error: {e}"
            ),
        )
        try:
            await websocket.close(code=1011, reason='Internal server error')
        except Exception:
            pass  # Connection might already be closed
    finally:
        print(
            (
                f"[WebSocket-Stream] {client_ip} ({username}): Connection "
                f"closed, total actions: {action_count}"
            ),
        )


@router.websocket('/ws/frames')
async def websocket_frames(
    websocket: WebSocket,
    rds: redis.Redis = Depends(get_redis_pool_ws),
) -> None:
    client_ip = websocket.client.host if websocket.client else 'unknown'
    print(f"[WebSocket] New connection from {client_ip}")

    # Accept incoming WebSocket connection
    await websocket.accept()

    # Extract JWT token from Authorization header or query parameter
    token: str | None = None

    # First try to get token from Authorization header (移動端/桌面端)
    auth: str | None = websocket.headers.get('authorization')
    if auth and auth.lower().startswith('bearer '):
        token = auth.split(' ', 1)[1]
        print(f"[WebSocket] {client_ip}: Token from Authorization header")

    # If no token from header, try query parameter (Web 平台)
    if not token:
        query_params = dict(websocket.query_params)
        token = query_params.get('token')
        if token:
            print(f"[WebSocket] {client_ip}: Token from query parameter")

    # If still no token found, close connection
    if not token:
        print(
            (
                f"[WebSocket] {client_ip}: No token found in header "
                'or query parameter'
            ),
        )
        await websocket.close(code=1008, reason='Missing authentication token')
        return

    # Verify JWT token using authentication module configuration
    try:
        payload: dict = jwt.decode(
            token, settings.authjwt_secret_key,
            algorithms=[settings.ALGORITHM],
        )
    except InvalidTokenError as e:
        print(f"[WebSocket] {client_ip}: Invalid JWT token: {e}")
        await websocket.close(code=1008, reason='Invalid token')
        return

    # Ensure JWT payload contains required data
    if not payload:
        print(f"[WebSocket] {client_ip}: Empty JWT payload")
        await websocket.close(code=1008, reason='Empty token payload')
        return

    # Extract username and JTI (JWT ID) from token payload
    subject_data = payload.get('subject', {})
    username: str | None = (
        subject_data.get('username') if subject_data else None
    ) or payload.get('username')
    jti: str | None = (
        subject_data.get('jti') if subject_data else None
    ) or payload.get('jti')

    # Validate that both username and JTI are present
    if not username or not jti:
        print(f"[WebSocket] {client_ip}: Missing username or JTI in token")
        await websocket.close(code=1008, reason='Invalid token data')
        return

    # Verify JTI against cached user data in Redis
    user_data: dict[str, str | list[str]] | None = await get_user_data(
        cast(redis.asyncio.Redis, rds), username,
    )

    # Ensure user data exists and contains the JTI list
    # If JTI is not found in user's active token list, close connection
    if (
        not user_data
        or 'jti_list' not in user_data
        or jti not in user_data['jti_list']
    ):
        # Close connection if JTI is not found in user's active token list
        print(
            (
                f"[WebSocket] {client_ip}: JTI not found in user "
                f"active tokens for {username}"
            ),
        )
        await websocket.close(code=1008, reason='Token not active')
        return

    print(f"[WebSocket] {client_ip}: Authenticated as {username}")
    frame_count = 0

    try:
        # Main frame processing loop
        while True:
            try:
                # Receive frame data in binary format with timeout
                # JSON header + DELIMITER + image bytes
                data: bytes | None = await asyncio.wait_for(
                    _safe_websocket_receive_bytes(
                        websocket, f"{client_ip} ({username})",
                    ),
                    timeout=60.0,  # 60秒超時
                )

                # Check if receive failed due to connection issues
                if data is None:
                    print(
                        (
                            f"[WebSocket] {client_ip} ({username}): "
                            'Failed to receive frame data, connection may be '
                            'closed'
                        ),
                    )
                    break

                # Split received data into header and frame components
                header_bytes: bytes
                frame_bytes: bytes
                header_bytes, frame_bytes = data.split(DELIMITER, 1)

                # Parse JSON header containing frame metadata
                header: dict[str, str | int] = json.loads(
                    header_bytes.decode('utf-8'),
                )

                # Extract frame metadata from parsed header
                label_value = header.get('label', '')
                label: str = str(label_value) if label_value else ''
                key_value = header.get('key', '')
                key: str = str(key_value) if key_value else ''
                warnings_json_value = header.get('warnings_json', '')
                warnings_json: str = (
                    str(warnings_json_value) if warnings_json_value else ''
                )
                cone_polygons_json_value = header.get('cone_polygons_json', '')
                cone_polygons_json: str = (
                    str(cone_polygons_json_value)
                    if cone_polygons_json_value else ''
                )
                pole_polygons_json_value = header.get('pole_polygons_json', '')
                pole_polygons_json: str = (
                    str(pole_polygons_json_value)
                    if pole_polygons_json_value else ''
                )
                detection_items_json_value = header.get(
                    'detection_items_json', '',
                )
                detection_items_json: str = (
                    str(detection_items_json_value)
                    if detection_items_json_value else ''
                )
                width_value = header.get('width', 0)
                width: int = int(width_value) if width_value else 0
                height_value = header.get('height', 0)
                height: int = int(height_value) if height_value else 0

                # Store frame and metadata in Redis for later retrieval
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

                frame_count += 1

                # Send success confirmation to client
                await _safe_websocket_send_json(
                    websocket,
                    {'status': 'ok', 'message': 'Frame stored successfully.'},
                    f"{client_ip} ({username})",
                )

                # 每處理100幀打印一次統計信息
                if frame_count % 100 == 0:
                    print(
                        (
                            f"[WebSocket] {client_ip} ({username}): Processed "
                            f"{frame_count} frames"
                        ),
                    )

            except asyncio.TimeoutError:
                print(
                    (
                        f"[WebSocket] {client_ip} ({username}): "
                        'Receive timeout after 60s'
                    ),
                )
                await websocket.close(code=1000, reason='Receive timeout')
                break

            except ValueError as e:
                # JSON parsing or data format error
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
                # Send error response if frame processing fails
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

    except WebSocketDisconnect:
        # Log WebSocket disconnection event
        print(
            (
                f"[WebSocket] {client_ip} ({username}): Client disconnected "
                f"after {frame_count} frames"
            ),
        )
    except Exception as e:
        # Handle general exceptions and close WebSocket gracefully
        print(
            (
                f"[WebSocket] {client_ip} ({username}): Unexpected error: "
                f"{e}"
            ),
        )
        try:
            await websocket.close(code=1011, reason='Internal server error')
        except Exception:
            pass  # Connection might already be closed
    finally:
        # Ensure proper cleanup when WebSocket connection terminates
        print(
            (
                f"[WebSocket] {client_ip} ({username}): Connection closed, "
                f"total frames: {frame_count}"
            ),
        )
