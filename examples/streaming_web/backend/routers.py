from __future__ import annotations

import redis
from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import Security
from fastapi import UploadFile
from fastapi import WebSocket
from fastapi_jwt import JwtAuthorizationCredentials
from fastapi_limiter.depends import RateLimiter
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.config import Settings
from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.redis_pool import get_redis_pool
from examples.auth.redis_pool import get_redis_pool_ws
from examples.auth.user_service import get_user_and_sites
from examples.shared.ws_helpers import get_auto_register_jti
from examples.streaming_web.backend.redis_service import scan_for_labels
from examples.streaming_web.backend.redis_service import store_to_redis
from examples.streaming_web.backend.schemas import FramePostResponse
from examples.streaming_web.backend.schemas import LabelListResponse
from examples.streaming_web.backend.utils import Utils
from examples.streaming_web.backend.ws_handlers import handle_frames_ws
from examples.streaming_web.backend.ws_handlers import handle_label_stream_ws
from examples.streaming_web.backend.ws_handlers import handle_stream_ws


# Settings and router
settings: Settings = Settings()
router: APIRouter = APIRouter()

# Rate limiters
rate_limiter_index: RateLimiter = RateLimiter(times=60, seconds=60)
rate_limiter_label: RateLimiter = RateLimiter(times=600, seconds=60)

# Env-configured behaviour
AUTO_REGISTER_JTI: bool = get_auto_register_jti()


@router.get(
    '/labels',
    response_model=LabelListResponse,
    dependencies=[Depends(rate_limiter_index)],
)
async def get_labels_route(
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> LabelListResponse:
    try:
        username: str | None = credentials.subject.get('username')
        if not username:
            raise HTTPException(
                status_code=401, detail='Invalid token: no subject',
            )

        _, user_site_names, user_role = await get_user_and_sites(db, username)
        print(
            f"User {username} has access to sites: {user_site_names}",
        )
        all_labels: list[str] = await scan_for_labels(rds)
        print(f"All labels in Redis: {all_labels}")
        filtered_labels = Utils.filter_labels(
            all_labels, user_role, user_site_names,
        )
        return LabelListResponse(labels=filtered_labels)
    except Exception as e:
        print(f"Failed to fetch labels: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch labels: {str(e)}",
        )


@router.post(
    '/frames',
    response_model=FramePostResponse,
)
async def post_frame(
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
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
    try:
        frame_bytes: bytes = await file.read()
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
            status='ok', message='Frame stored successfully.',
        )
    except Exception as e:
        print(f"Failed to store frame: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to store frame: {str(e)}",
        )


@router.websocket('/ws/labels/{label}')
async def websocket_label_stream(
    websocket: WebSocket,
    label: str,
    rds: redis.Redis = Depends(get_redis_pool_ws),
) -> None:
    await handle_label_stream_ws(
        websocket=websocket,
        label=label,
        rds=rds,
        settings=settings,
    )


@router.websocket('/ws/stream/{label}/{key}')
async def websocket_stream(
    websocket: WebSocket,
    label: str,
    key: str,
    rds: redis.Redis = Depends(get_redis_pool_ws),
) -> None:
    await handle_stream_ws(
        websocket=websocket,
        label=label,
        key=key,
        rds=rds,
        settings=settings,
    )


@router.websocket('/ws/frames')
async def websocket_frames(
    websocket: WebSocket,
    rds: redis.Redis = Depends(get_redis_pool_ws),
) -> None:
    await handle_frames_ws(websocket=websocket, rds=rds, settings=settings)
