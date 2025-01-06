from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi_limiter.depends import RateLimiter
from linebot import LineBotApi

from .utils import RedisManager
from .utils import Utils
from .utils import WebhookHandler
# from pathlib import Path
# from fastapi import UploadFile

redis_manager = RedisManager()

# Create an API router for defining routes
router = APIRouter()
line_bot_api = LineBotApi(
    os.getenv('LINE_CHANNEL_ACCESS_TOKEN'),
)
webhook_handler = WebhookHandler(line_bot_api)

CONFIG_PATH = 'config/configuration.json'  # Path to the configuration file


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


@router.websocket('/api/ws/stream/{label}/{key}')
async def websocket_stream(websocket: WebSocket, label: str, key: str) -> None:
    """
    Establishes a WebSocket connection to stream data for a single camera.

    Args:
        websocket (WebSocket): The WebSocket connection object.
        label (str): The label associated with the stream.
        key (str): The key identifying the specific camera stream.
    """
    await websocket.accept()
    try:
        # Encode the label and key for Redis lookup
        encoded_label = Utils.encode(label)
        encoded_key = Utils.encode(key)
        redis_key = f"stream_frame:{encoded_label}|{encoded_key}"

        # Initialize last message ID for the stream
        last_id = '0'

        while True:
            # Fetch the latest frame for the specific stream
            message = await redis_manager.fetch_latest_frame_for_key(
                redis_key, last_id,
            )
            if message:
                # Update the last ID
                last_id = message['id']
                # Send the latest frame and warnings to the client
                await websocket.send_json(message)
            else:
                # If no new data is available, close the connection
                await websocket.send_json({'error': 'No new data available'})
                await websocket.close()
                break

    except WebSocketDisconnect:
        print('WebSocket disconnected')
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Close the WebSocket connection on error
        await websocket.close()
    finally:
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
    try:
        # Retrieve the request body
        body = await request.json()
        responses = await webhook_handler.process_webhook_events(body)

        # If all events were skipped, return a success response
        if all(resp.get('status') == 'skipped' for resp in responses):
            return JSONResponse(
                content={
                    'status': 'skipped',
                    'message': 'All events skipped.',
                },
                status_code=200,
            )

        # If any event failed, return a partial error response
        if any(resp['status'] == 'error' for resp in responses):
            return JSONResponse(
                content={'status': 'partial_error', 'responses': responses},
                status_code=207,  # 207 Multi-Status
            )

        return JSONResponse(content={'status': 'ok', 'responses': responses})

    except Exception as e:
        print(f"Unexpected error while processing webhook: {e}")
        raise HTTPException(
            status_code=500, detail='Webhook processing failed',
        )


@router.get('/api/config')
async def get_config(request: Request) -> JSONResponse:
    """
    Retrieve the current configuration.

    Args:
        request (Request): The HTTP request.

    Returns:
        JSONResponse: A JSON response containing the current configuration.
    """
    Utils.verify_localhost(request)
    config = Utils.load_configuration(CONFIG_PATH)
    return JSONResponse(content={'config': config})


@router.post('/api/config')
async def update_config(request: Request) -> JSONResponse:
    """
    Update the configuration with the provided data.

    Args:
        request (Request): The HTTP request containing the new configuration.

    Returns:
        JSONResponse: A JSON response indicating the status of the request.
    """
    Utils.verify_localhost(request)

    try:
        # Retrieve the new configuration data
        data = await request.json()
        new_config = data.get('config', [])

        # Update the configuration file
        updated_config = Utils.update_configuration(CONFIG_PATH, new_config)

        return JSONResponse(
            content={
                'status': 'Configuration updated successfully.',
                'config': updated_config,
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update configuration: {e}",
        )


# Uncomment and use the following endpoint for file uploads if needed
# @router.post('/api/upload')
# async def upload_file(file: UploadFile) -> JSONResponse:
#     """
#     Saves an uploaded file to the designated upload folder
#     and returns its accessible URL.

#     Args:
#         file (UploadFile): The file to upload.

#     Returns:
#         JSONResponse: A JSON response containing the URL of
#             the uploaded file.
#     """
#     UPLOAD_FOLDER = Path('uploads')
#     UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

#     # Check if the file has a filename
#     if not file.filename:
#         raise HTTPException(status_code=400, detail='Filename is missing')

#     file_path = UPLOAD_FOLDER / file.filename
#     try:
#         with open(file_path, 'wb') as buffer:
#             buffer.write(await file.read())
#     except PermissionError as e:
#         raise HTTPException(
#             status_code=500, detail=f"Failed to save file: {str(e)}",
#         )

#     url = f"/uploads/{file.filename}"
#     return JSONResponse(content={'url': url})
