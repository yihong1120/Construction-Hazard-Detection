from __future__ import annotations

import base64
import unittest
from functools import partial
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from starlette.datastructures import UploadFile

from examples.mcp_server.tools.record import RecordTools
from examples.mcp_server.tools.record import ViolationUpload
from examples.violation_records import violation_upload_service as module
from examples.violation_records.evidence import ViolationEvidence
from tests.image_fixtures_test import image_bytes


async def _invalid_metadata_returns_422_before_endpoint(evidence, /):
    return await module.upload_violation(
        evidence,
        None,
        SimpleNamespace(subject={'username': 'operator'}),
    )


class MetadataBoundaryTest(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_metadata_returns_422_before_catalog_or_storage(
        self,
    ):
        for changes in [
            {'warnings_json': 'not json'},
            {'warnings_json': '{"warning_no_hardhat":{"count":"invalid"}}'},
            {'detections_json': '[[0,0,1,1,NaN,0,1]]'},
            {'cone_polygon_json': '[[[0,0],[1,1]]]'},
        ]:
            evidence = ViolationEvidence(
                site='site',
                site_id=1,
                stream_name='cam',
                model_id='model',
                model_version='v1',
                image_file=UploadFile(BytesIO(image_bytes())),
                **changes,
            )
            app = FastAPI()

            app.post('/upload')(
                partial(
                    _invalid_metadata_returns_422_before_endpoint,
                    evidence,
                ),
            )

            with patch.object(
                module,
                'select_model_async',
                AsyncMock(),
            ) as catalog, patch.object(
                module.violation_manager,
                'save_violation',
                AsyncMock(),
            ) as storage:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url='http://test',
                ) as client:
                    response = await client.post('/upload')
                self.assertEqual(response.status_code, 422)
                catalog.assert_not_awaited()
                storage.assert_not_awaited()

    async def test_mcp_rejects_invalid_warning_before_creating_transport(self):
        upload = ViolationUpload(
            image_base64=base64.b64encode(image_bytes()).decode(),
            site='site',
            stream_name='cam',
            model_id='model',
            model_version='v1',
            detections=[],
            warnings={'warning_no_hardhat': {'count': 'invalid'}},
        )
        with patch(
            'examples.mcp_server.tools.record.ViolationSender',
        ) as sender:
            result = await RecordTools().send_violation(upload)
        self.assertFalse(result['success'])
        sender.assert_not_called()
