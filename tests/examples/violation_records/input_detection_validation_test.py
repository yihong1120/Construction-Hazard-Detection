from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.violation_records import violation_upload_service as module
from examples.violation_records.evidence import ViolationEvidence


class InputDetectionValidationTest(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_tracking_values_return_422_before_evidence_write(
        self,
    ):
        entry = SimpleNamespace(
            id='model',
            version='v1',
            classes=[SimpleNamespace(id=0)],
        )
        db = SimpleNamespace(
            get=AsyncMock(
                return_value=SimpleNamespace(tenant_id='tenant', role='user'),
            ),
        )
        credentials = SimpleNamespace(
            subject={
                'username': 'alice',
                'user_id': 1,
                'tenant_id': 'tenant',
                'role': 'user',
            },
        )
        with patch.object(
            module,
            'require_site',
            new=AsyncMock(),
        ), patch.object(
            module,
            'select_model_async',
            new=AsyncMock(return_value=entry),
        ), patch.object(
            module,
            'resolve_evidence_version',
            new=AsyncMock(return_value=entry),
        ), patch.object(
            module.violation_manager,
            'save_violation',
            new=AsyncMock(),
        ) as write:
            with self.assertRaises(HTTPException) as caught:
                await module.upload_violation(
                    ViolationEvidence(
                        site='Site',
                        site_id=1,
                        stream_name='Cam',
                        image_file=None,
                        model_id='model',
                        model_version='v1',
                        detections_json='[[1,2,3,4,0.9,0,NaN]]',
                    ),
                    db,
                    credentials,
                )
        self.assertEqual(caught.exception.status_code, 422)
        write.assert_not_awaited()
