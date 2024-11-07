from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import UploadFile
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi_limiter.depends import RateLimiter
from starlette.templating import Jinja2Templates

from .utils import redis_manager
from .utils import Utils

# Create an API router for defining routes
router = APIRouter()
templates = Jinja2Templates(
    directory='examples/streaming_web/templates',
)


def register_routes(app: Any) -> None:
    """
    Registers the API router with the FastAPI application.

    Args:
        app (Any): The FastAPI application instance.
    """
    app.include_router(router)


@router.get('/', dependencies=[Depends(RateLimiter(times=60, seconds=60))])
async def index(request: Request) -> Jinja2Templates.TemplateResponse:
    """
    Renders the index page with available labels from Redis.

    Args:
        request (Request): The HTTP request object.

    Returns:
        TemplateResponse: The rendered HTML template for the index page,
            containing available labels.
    """
    try:
        # Retrieve available labels from Redis
        labels = await redis_manager.get_labels()
    except Exception as e:
        # Raise HTTP 500 error if labels cannot be fetched
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch labels: {str(e)}",
        )
    # Render and return the index template with the fetched labels
    return templates.TemplateResponse(
        'index.html',
        {
            'request': request,
            'labels': labels,
        },
    )


@router.get(
    '/label/{label}',
    dependencies=[Depends(RateLimiter(times=6000, seconds=6000))],
)
async def label_page(
    request: Request,
    label: str,
) -> Jinja2Templates.TemplateResponse:
    """
    Renders the page for a specific label with available labels from Redis.

    Args:
        request (Request): The HTTP request object.
        label (str): The label identifier to display on the page.

    Returns:
        TemplateResponse: The rendered HTML template for the label page
            with available labels.
    """
    try:
        # Retrieve available labels from Redis
        labels = await redis_manager.get_labels()
        # Check if the requested label is present in the available labels
        if label not in labels:
            raise HTTPException(
                status_code=404, detail=f"Label '{label}' not found",
            )
    except Exception as e:
        # Raise HTTP 500 error if labels cannot be fetched
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch labels: {str(e)}",
        )
    # Render and return the label template with the fetched labels
    return templates.TemplateResponse(
        'label.html',
        {
            'request': request,
            'label': label,
            'labels': labels,
        },
    )


@router.websocket('/ws/label/{label}')
async def websocket_label_stream(websocket: WebSocket, label: str) -> None:
    """
    Establishes a WebSocket connection to stream updated frames
    for a specific label.

    Args:
        websocket (WebSocket): The WebSocket connection object.
        label (str): The label identifier for the frame stream.
    """
    await websocket.accept()
    try:
        # Fetch Redis keys associated with the label
        keys = await redis_manager.get_keys_for_label(label)
        # Check if there are any keys associated with the label
        if not keys:
            await websocket.send_json({
                'error': f"No keys found for label '{label}'",
            })
            await websocket.close()
            return

        # Initialise last message IDs for each key in the stream
        last_ids: dict[str, str] = {key: '0' for key in keys}

        while True:
            # Fetch the latest frames for each key in the specified label
            updated_data = await redis_manager.fetch_latest_frames(last_ids)
            if updated_data:
                # Send the updated frames to the WebSocket client
                await Utils.send_frames(websocket, label, updated_data)
    except WebSocketDisconnect:
        # Handle WebSocket disconnection gracefully
        print('WebSocket disconnected')
    except Exception as e:
        # Log unexpected errors
        print(f"Unexpected error: {e}")
    finally:
        # Final message on WebSocket closure
        print('WebSocket connection closed')


@router.post('/webhook')
async def webhook(request: Request) -> JSONResponse:
    """
    Processes incoming webhook requests by logging the request body.

    Args:
        request (Request): The HTTP request containing the webhook payload.

    Returns:
        JSONResponse: A JSON response indicating the status of the request.
    """
    # Retrieve and log the webhook request body
    body = await request.json()
    print(body)
    # Respond with a JSON status message
    return JSONResponse(content={'status': 'ok'})

# Uncomment and use the following endpoint for file uploads if needed


@router.post('/upload')
async def upload_file(file: UploadFile) -> JSONResponse:
    """
    Saves an uploaded file to the designated upload folder
    and returns its accessible URL.

    Args:
        file (UploadFile): The file to upload.

    Returns:
        JSONResponse: A JSON response containing the URL of the uploaded file.
    """
    UPLOAD_FOLDER = Path('static/uploads')
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    # 检查 filename 是否为 None
    if not file.filename:
        raise HTTPException(status_code=400, detail='Filename is missing')

    file_path = UPLOAD_FOLDER / file.filename
    try:
        with open(file_path, 'wb') as buffer:
            buffer.write(await file.read())
    except PermissionError as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save file: {str(e)}",
        )

    url = f"https://yihong-server.mooo.com/static/uploads/{file.filename}"
    return JSONResponse(content={'url': url})
