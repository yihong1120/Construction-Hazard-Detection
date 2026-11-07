from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.YOLO_server_api import model_registry as catalog
from examples.YOLO_server_api import routers
from examples.YOLO_server_api.schemas import UpdateModelRequest


class VerifiedArtifactTest(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_conditional_downloads_reuse_one_verified_checksum(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pt'
            path.write_bytes(b'trusted-weights')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entry = SimpleNamespace(
                artifact=str(path),
                sha256=digest,
                id='custom',
                version='v1',
            )
            request = SimpleNamespace(headers={'if-none-match': f'"{digest}"'})
            payload = UpdateModelRequest(
                model='custom',
                last_update_time='2026-01-01T00:00:00',
            )
            credentials = SimpleNamespace(
                subject={'role': 'admin', 'tenant_id': 't1'},
            )
            catalog._artifact_digest.cache_clear()
            with patch.object(
                routers,
                'select_model_async',
                AsyncMock(return_value=entry),
            ), patch.object(
                catalog.hashlib,
                'file_digest',
                wraps=hashlib.file_digest,
            ) as checksum:
                for _ in range(2):
                    response = await routers.get_new_model(
                        request,
                        payload,
                        credentials,
                    )
                    self.assertEqual(response.status_code, 304)
                    self.assertEqual(response.headers['etag'], f'"{digest}"')
                checksum.assert_called_once()
                path.write_bytes(b'replaced-with-other-weights')
                with self.assertRaises(HTTPException) as error:
                    await routers.get_new_model(request, payload, credentials)
                self.assertEqual(error.exception.status_code, 409)
                self.assertEqual(checksum.call_count, 2)

    async def test_permission_checks_precede_file_validation(self):
        with patch.object(
            routers,
            'select_model_async',
            AsyncMock(side_effect=HTTPException(403, 'denied')),
        ), patch.object(routers, 'verified_artifact') as artifact:
            with self.assertRaises(HTTPException) as error:
                await routers.get_new_model(
                    SimpleNamespace(headers={}),
                    UpdateModelRequest(
                        model='hidden',
                        last_update_time='2026-01-01T00:00:00',
                    ),
                    SimpleNamespace(
                        subject={'role': 'user', 'tenant_id': 't1'},
                    ),
                )
            self.assertEqual(error.exception.status_code, 403)
            artifact.assert_not_called()
