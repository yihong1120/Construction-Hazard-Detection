from __future__ import annotations

import datetime
from pathlib import Path

import httpx
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi_limiter.depends import RateLimiter
from werkzeug.utils import secure_filename

models_router = APIRouter()
MODELS_DIRECTORY = Path('models/pt/')
ALLOWED_MODELS = {'best_yolo11l.pt', 'best_yolo11x.pt'}


@models_router.get(
    '/models/pt/{model_name}',
    dependencies=[Depends(RateLimiter(times=10, seconds=60))],
    response_model=None,  # Disable automatic response model generation
)
async def download_model(model_name: str):
    """
    Downloads the specified model if it is allowed and up-to-date.

    Args:
        model_name (str): The name of the model to download.

    Returns:
        FileResponse or dict: FileResponse if the model is found
        and up-to-date, otherwise an error message.

    Raises:
        HTTPException: If the model is not found, or if there is an error
        fetching model information.
    """
    if model_name not in ALLOWED_MODELS:
        raise HTTPException(status_code=404, detail='Model not found')

    # Ensure the model name is sanitized
    sanitized_model_name = secure_filename(model_name)
    if sanitized_model_name != model_name:
        raise HTTPException(status_code=400, detail='Invalid model name')

    try:
        MODEL_URL = (
            f"http://changdar-server.mooo.com:28000/"
            f"models/{sanitized_model_name}"
        )

        # Asynchronously request headers information
        async with httpx.AsyncClient() as client:
            response = await client.head(MODEL_URL)

        if response.status_code == 200 and 'Last-Modified' in response.headers:
            server_last_modified = datetime.datetime.strptime(
                response.headers['Last-Modified'],
                '%a, %d %b %Y %H:%M:%S GMT',
            )
            local_file_path = MODELS_DIRECTORY / sanitized_model_name
            try:
                # Resolve the local file path and ensure it is within the
                # models directory
                resolved_path = local_file_path.resolve()
                if not resolved_path.parent == MODELS_DIRECTORY.resolve():
                    raise HTTPException(
                        status_code=400, detail='Invalid model name',
                    )
            except ValueError:
                raise HTTPException(
                    status_code=400, detail='Invalid model name',
                )

            # Check if the local file exists and its modification time
            if local_file_path.exists():
                local_last_modified = datetime.datetime.fromtimestamp(
                    local_file_path.stat().st_mtime,
                )
                if local_last_modified >= server_last_modified:
                    return {'message': 'Local model is up-to-date'}, 304

        # Return file response
        return FileResponse(
            local_file_path,
            filename=sanitized_model_name,
            headers={
                'Content-Disposition': (
                    f"attachment; filename={sanitized_model_name}"
                ),
            },
        )

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail='Model not found')

    except httpx.RequestError:
        raise HTTPException(
            status_code=500, detail='Failed to fetch model information',
        )
