from __future__ import annotations

import os
from unittest import TestCase
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.testclient import TestClient

from examples.violation_records.response_debug import ViolationResponseDebug


class ResponseDebugTest(TestCase):
    def test_logs_serialized_json_only_for_selected_paths(self):
        app = FastAPI()
        app.add_middleware(ViolationResponseDebug)

        @app.get('/violations/{id}')
        @app.get('/violations/{id}/feedback-options')
        @app.get('/violations/{id}/review-options')
        def detail(id: int):
            return {'id': id, 'display_name': '人員', 'bbox': {'x': 0.2}}

        @app.get('/get_violation_image')
        def media():
            return Response(b'image', media_type='image/png')

        with (
            patch.dict(os.environ, VIOLATION_DEBUG_RESPONSE_IDS='155157'),
            patch(
                'examples.violation_records.response_debug.logger.info',
            ) as log,
            TestClient(app) as client,
        ):
            for suffix in ('', '/feedback-options', '/review-options'):
                response = client.get(
                    '/violations/155157' + suffix,
                    headers={'Authorization': 'Bearer secret'},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(log.call_args.args[-1], response.text)
                self.assertNotIn('secret', str(log.call_args))
            self.assertEqual(log.call_count, 3)
            client.get('/violations/123')
            client.get('/get_violation_image')
            self.assertEqual(log.call_count, 3)
            with patch.dict(os.environ, VIOLATION_DEBUG_RESPONSE_IDS=''):
                client.get('/violations/155157')
            self.assertEqual(log.call_count, 3)

            with patch.dict(os.environ, VIOLATION_DEBUG_RESPONSE_IDS='*'):
                for record_id in (155157, 155158, 999999):
                    for suffix in ('', '/feedback-options', '/review-options'):
                        response = client.get(
                            f'/violations/{record_id}{suffix}',
                        )
                        self.assertEqual(log.call_args.args[-1], response.text)
                self.assertEqual(log.call_count, 12)
                client.get('/get_violation_image')
                self.assertEqual(log.call_count, 12)
