from __future__ import annotations

import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from examples.YOLO_server_api.inference_runtime import InferenceRuntime
from examples.YOLO_server_api.inference_runtime import ModelSelection
from examples.YOLO_server_api.routers import _read_limited_upload
from src.settings.inference import ApiInferenceSettings


async def check_size_contract():
    models = Mock()
    runtime = InferenceRuntime(
        models, ApiInferenceSettings(max_upload_bytes=1),
    )
    try:
        selection = ModelSelection(SimpleNamespace(), runtime)
        with pytest.raises(HTTPException) as caught:
            await selection.detect(b'xx')
        assert caught.value.status_code == 413
        assert caught.value.detail['code'] == 'IMAGE_UPLOAD_LIMIT_EXCEEDED'
        assert runtime._admitted == 0 and not runtime._requests
        models.lease.assert_not_called()
        with pytest.raises(HTTPException) as http:
            await _read_limited_upload(UploadFile(BytesIO(b'xx')), max_bytes=1)
        assert http.value.status_code == caught.value.status_code
    finally:
        await runtime.close()


def test_websocket_and_http_limits_match():
    asyncio.run(check_size_contract())
