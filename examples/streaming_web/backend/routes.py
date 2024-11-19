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

from .utils import RedisManager
from .utils import Utils

redis_manager = RedisManager()

# Create an API router for defining routes
router = APIRouter()


def register_routes(app: Any) -> None:
    """
    Registers the API router with the FastAPI application.

    Args:
        app (Any): The FastAPI application instance.
    """
    app.include_router(router)


# Create rate limiters for the API routes
rate_limiter_index = RateLimiter(times=60, seconds=60)
rate_limiter_label = RateLimiter(times=6000, seconds=6000)


@router.get('/api/labels', dependencies=[Depends(rate_limiter_index)])
async def get_labels() -> JSONResponse:
    """
    Renders the page for a specific label with available labels from Redis.

    Args:
        None

    Returns:
        JSONResponse: A JSON response containing the labels.
    """
    try:
        # Retrieve available labels from Redis
        labels = await redis_manager.get_labels()

    except ValueError as ve:
        print(f"ValueError while fetching labels: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data encountered: {str(ve)}",
        )
    except KeyError as ke:
        print(f"KeyError while fetching labels: {str(ke)}")
        raise HTTPException(
            status_code=404,
            detail=f"Missing key encountered: {str(ke)}",
        )
    except ConnectionError as ce:
        print(f"ConnectionError while fetching labels: {str(ce)}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to the database: {str(ce)}",
        )
    except TimeoutError as te:
        print(f"TimeoutError while fetching labels: {str(te)}")
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out: {str(te)}",
        )
    except Exception as e:
        print(f"Unexpected error while fetching labels: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch labels: {str(e)}",
        )

    return JSONResponse(content={'labels': labels})


@router.websocket('/api/ws/labels/{label}')
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


@router.post('/api/webhook')
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


@router.post('/api/upload')
async def upload_file(file: UploadFile) -> JSONResponse:
    """
    Saves an uploaded file to the designated upload folder
    and returns its accessible URL.

    Args:
        file (UploadFile): The file to upload.

    Returns:
        JSONResponse: A JSON response containing the URL of the uploaded file.
    """
    UPLOAD_FOLDER = Path('uploads')
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    # Check if the file has a filename
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

    url = f"/uploads/{file.filename}"
    return JSONResponse(content={'url': url})
