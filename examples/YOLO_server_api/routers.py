from __future__ import annotations

import asyncio
import datetime
import logging
import time
from pathlib import Path
from typing import Final
from typing import Literal

import aiofiles  # type: ignore[import-untyped]
import redis.asyncio as redis
from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import Form
from fastapi import HTTPException
from fastapi import Request
from fastapi import Security
from fastapi import UploadFile
from fastapi import WebSocket
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.responses import Response

from examples.auth.cache import rate_limiter_service
from examples.auth.config import Settings
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.redis_pool import get_redis_pool_ws
from examples.YOLO_server_api.inference_runtime import InferenceRuntime
from examples.YOLO_server_api.model_registry import load_registry_async
from examples.YOLO_server_api.model_registry import select_model_async
from examples.YOLO_server_api.model_registry import verified_artifact
from examples.YOLO_server_api.overlay_runtime import OverlayRuntime
from examples.YOLO_server_api.overlay_runtime import OverlayWork
from examples.YOLO_server_api.schemas import DetectionModelInfo
from examples.YOLO_server_api.schemas import DetectionModelList
from examples.YOLO_server_api.schemas import DetectionOverlayResponse
from examples.YOLO_server_api.schemas import DetectionRequest
from examples.YOLO_server_api.schemas import ModelFileUpdate
from examples.YOLO_server_api.schemas import UpdateModelRequest
from examples.YOLO_server_api.websocket_handlers import (
    handle_websocket_detect,
)
from src.settings.inference import ModelCatalogSettings

# Router instances for API endpoints
detection_router: APIRouter = APIRouter()
model_management_router: APIRouter = APIRouter()

# Global model manager instance for handling YOLO models
_inference_runtime: InferenceRuntime | None = None
_overlay_runtime: OverlayRuntime | None = None


def get_overlay_runtime() -> OverlayRuntime:
    global _overlay_runtime
    if _overlay_runtime is None or _overlay_runtime._closed:
        _overlay_runtime = OverlayRuntime()
    return _overlay_runtime


def get_inference_runtime() -> InferenceRuntime:
    global _inference_runtime
    if _inference_runtime is None or _inference_runtime._closed:
        _inference_runtime = InferenceRuntime()
    return _inference_runtime


# Application settings configuration
settings: Settings = Settings()
logger = logging.getLogger(__name__)

_upload_chunk_size: Final[int] = 1024 * 1024
_model_upload_max_bytes: Final[int] = ModelCatalogSettings().upload_max_bytes


async def _read_limited_upload(
    upload_file: UploadFile,
    *,
    max_bytes: int,
    chunk_size: int = _upload_chunk_size,
) -> bytes:
    """Read an upload with a strict allocation limit."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload_file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f'Upload exceeds the {max_bytes}-byte limit',
            )
        chunks.append(chunk)
    if not chunks:
        raise HTTPException(status_code=400, detail='Empty upload file')
    return b''.join(chunks)


async def _stream_upload_to_path(
    upload_file: UploadFile,
    destination: Path,
    *,
    max_bytes: int = _model_upload_max_bytes,
    chunk_size: int = _upload_chunk_size,
) -> None:
    """Stream an uploaded file to disk without buffering it in memory."""
    wrote_any = False
    total = 0
    async with aiofiles.open(destination, 'wb') as f:
        while True:
            chunk = await upload_file.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f'Upload exceeds the {max_bytes}-byte limit',
                )
            wrote_any = True
            await f.write(chunk)
    if not wrote_any:
        raise ValueError('Empty upload file')


@detection_router.get(
    '/models',
    response_model=DetectionModelList,
    response_model_exclude_unset=True,
)
async def list_detection_models(
    response: Response,
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> DetectionModelList:
    """List models using the same JWT access policy as image detection."""
    registry = await load_registry_async()
    entries = [
        entry
        for entry in registry.models
        if entry.allowed(
            str(credentials.subject.get('tenant_id', '')),
            credentials.subject['role'],
            'image',
        )
    ]
    default = (
        registry.default_id
        if registry.default_id in {e.id for e in entries}
        else None
    )
    response.headers['Cache-Control'] = 'private, no-store'
    return DetectionModelList(
        default_model_id=default,
        models=[
            DetectionModelInfo(
                id=e.id,
                display_name=e.display_name,
                version=e.version,
                classes=e.classes,
            )
            for e in entries
        ],
    )


@detection_router.post('/detect', response_class=JSONResponse)
async def detect(
    detection_request: DetectionRequest = Depends(DetectionRequest.as_form),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    remaining_requests: int = Depends(rate_limiter_service),
) -> list[list[float | int]]:
    """Process object detection on uploaded images using YOLO models.

    This endpoint accepts image uploads and performs object detection using
    the specified YOLO model. It includes authentication, rate limiting,
    and comprehensive timing metrics for performance monitoring.

    Args:
        detection_request: Form data containing image file and model selection.
        credentials: JWT authentication credentials for user validation.
        remaining_requests: Number of remaining API requests for rate limiting.

    Returns:
        A list of detection results where each detection is represented as
        a list of numerical values [x1, y1, x2, y2, confidence, class_id].

    Raises:
        HTTPException: 404 if the specified model is not found.

    Note:
        The shared inference runtime bounds queued and running work, including
        model loading. Capacity exhaustion returns HTTP 503.
    """
    datas, _, _ = await _run_detection_request(
        detection_request,
        credentials,
        remaining_requests,
    )
    return datas


async def _run_detection_request(
    detection_request,
    credentials,
    remaining_requests,
):
    start_time = time.monotonic()
    runtime = get_inference_runtime()
    async with runtime.admission() as work:
        img_bytes = await _read_limited_upload(
            detection_request.image,
            max_bytes=runtime.settings.max_upload_bytes,
        )
        io_time = time.monotonic() - start_time
        entry = await select_model_async(
            detection_request.model,
            detection_request.model_version,
            str(credentials.subject.get('tenant_id', '')),
            credentials.subject['role'],
            'image',
        )
        result = await work.run(entry, img_bytes)
        datas, timing = result.rows, result.timing
    inference_time = timing['inference']
    post_time = timing['post']

    # Log comprehensive timing information for performance analysis
    total_time: float = time.monotonic() - start_time
    logger.info(
        'Detection completed model=%s io_seconds=%.3f '
        'inference_seconds=%.3f post_seconds=%.3f total_seconds=%.3f',
        detection_request.model,
        io_time,
        inference_time,
        post_time,
        total_time,
    )

    return datas, entry, result.image_size


@detection_router.websocket('/ws/detect')
async def websocket_detect(
    websocket: WebSocket,
    rds: redis.Redis = Depends(get_redis_pool_ws),
) -> None:
    """"""
    await handle_websocket_detect(
        websocket=websocket,
        rds=rds,
        settings=settings,
        runtime=get_inference_runtime(),
    )


@model_management_router.post('/model_file_update')
async def model_file_update(
    data: ModelFileUpdate = Depends(ModelFileUpdate.as_form),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> dict[str, str]:
    """Update a YOLO model file with administrative privileges.

    This endpoint allows administrators and model managers to upload and
    update YOLO model files. It includes role-based access control,
    secure file handling, and comprehensive error management.

    Args:
        data: Form data containing the model file and model identifier.
        credentials: JWT authentication credentials for authorisation.

    Returns:
        A dictionary containing a success message upon completion.

    Raises:
        HTTPException:
            - 403 if user lacks required permissions
            - 400 for validation errors
            - 500 for I/O errors

    Note:
        This endpoint requires 'admin' or 'model_manage' roles and
        automatically cleans up temporary files after processing.
    """
    # Verify user has the required role for model management
    role: str = credentials.subject.get('role', '')
    if role not in ['admin', 'model_manage']:
        raise HTTPException(
            status_code=403,
            detail="Permission denied. Need 'admin' or 'model_manage' role.",
        )

    raise HTTPException(
        409,
        detail={
            'code': 'IMMUTABLE_MODEL_REGISTRY',
            'message': (
                'Publish to Hugging Face and synchronize the catalog key.'
            ),
        },
    )


@model_management_router.post('/get_new_model')
async def get_new_model(
    request: Request,
    update_request: UpdateModelRequest = Body(...),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> Response:
    """Stream a newer authenticated model artefact when one is available.

    This endpoint checks if a newer version of the requested model is
    available on the server compared to the client's last update time.
    It supports incremental updates to minimise bandwidth usage.

    Args:
        update_request:
            Request body containing model name and last update timestamp.
        credentials:
            JWT authentication credentials for authorisation.

    Returns:
        A dictionary containing either:
        - Binary model response with ETag if a newer version exists
        - ``204 No Content`` if the local copy is already current

    Raises:
        HTTPException:
            - 403 if user has 'guest' role (insufficient permissions)
            - 400 for invalid timestamp format or validation errors
            - 500 for unexpected server errors

    Note:
        This endpoint excludes 'guest' users but allows other authenticated
        roles to check for and download model updates.
    """
    # Verify user has sufficient privileges (exclude guest users)
    role: str = credentials.subject.get('role', '')
    if role == 'guest':
        raise HTTPException(
            status_code=403,
            detail="Permission denied. Need 'admin' or 'model_manage' role.",
        )

    try:
        # Parse the client's last update timestamp
        datetime.datetime.fromisoformat(
            update_request.last_update_time,
        )

        entry = await select_model_async(
            update_request.model,
            None,
            str(credentials.subject.get('tenant_id', '')),
            role,
            'image',
        )
        artifact = await asyncio.to_thread(verified_artifact, entry)
        etag = artifact.etag
        if request.headers.get('if-none-match') == etag:
            return Response(status_code=304, headers={'ETag': etag})

        logger.info('Streaming updated model model=%s', update_request.model)
        return FileResponse(
            artifact.path,
            media_type='application/octet-stream',
            filename=artifact.path.name,
            headers={
                'Cache-Control': 'private, no-store',
                'ETag': etag,
                'X-Model-SHA256': artifact.sha256,
                'X-Model-Version': artifact.version,
                'X-Model-Id': artifact.model_id,
            },
        )

    except HTTPException:
        raise
    except ValueError as e:
        # Handle invalid timestamp format or validation errors
        logger.warning('Model fetch validation error: %s', e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        # Handle unexpected server errors
        logger.exception('Error retrieving model')
        raise HTTPException(
            status_code=500,
            detail='Failed to retrieve model.',
        )


@detection_router.post('/v2/detect', response_model=DetectionOverlayResponse)
async def detect_v2(
    request: Request,
    detection_request: DetectionRequest = Depends(DetectionRequest.as_form),
    compute_regions: bool = Form(True),
    view: Literal['objects', 'regions'] = Form('regions'),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    remaining_requests: int = Depends(rate_limiter_service),
) -> DetectionOverlayResponse:
    # This API defaults to region computation, independently of HTML checkbox
    # handling that interprets omitted boolean form fields as false.
    if 'compute_regions' not in await request.form():
        compute_regions = True
    if view == 'regions' and not compute_regions:
        raise HTTPException(
            422,
            detail='Region view requires compute_regions=true',
        )
    datas, entry, image_size = await _run_detection_request(
        detection_request,
        credentials,
        remaining_requests,
    )
    return await get_overlay_runtime().render(
        OverlayWork(datas, entry, image_size, compute_regions, view),
    )
