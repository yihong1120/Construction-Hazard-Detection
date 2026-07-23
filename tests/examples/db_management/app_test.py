from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from examples.bff.router import router as bff_router
from examples.db_management import app as app_module


class AppIntegrationTest(unittest.TestCase):
    """
    Integration tests for the FastAPI app in db_management.
    """
    app: FastAPI
    client: TestClient
    _patchers: list = []

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the test class by patching global dependencies
        and initialising the app and client.
        """
        # Store patchers so they can be stopped later
        cls._patchers = [
            patch('examples.auth.database.get_db', new=lambda *a, **kw: None),
            patch(
                'examples.auth.redis_pool.get_redis_pool',
                new=lambda *a, **kw: None,
            ),
            patch(
                'examples.db_management.deps.require_admin',
                new=lambda *a, **kw: None,
            ),
            patch(
                'examples.db_management.deps.require_super_admin',
                new=lambda *a, **kw: None,
            ),
            patch(
                'examples.db_management.deps.get_current_user',
                new=lambda *a, **kw: None,
            ),
        ]
        for p in cls._patchers:
            p.start()
        # Use the FastAPI app instance from the imported module
        cls.app = app_module.app
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up test fixtures."""
        for p in getattr(cls, '_patchers', []):
            p.stop()

    # ---------- Tests ----------

    def test_openapi_available(self) -> None:
        """
        The OpenAPI schema endpoint should return 200
        and contain title and paths.
        """
        resp = self.client.get('/openapi.json')
        self.assertEqual(resp.status_code, 200)
        data: dict = resp.json()
        self.assertIn('paths', data)
        # Roughly check that several classic router paths exist
        expected_paths: list[str] = [
            '/list_features',
            '/list_groups',
            '/list_pending_users',
            '/list_sites',
            '/list_stream_configs',
            '/list_users',
            '/sites/{site_id}/stream-config',
            '/approve_user_signup',
            '/signup',
            '/password/forgot',
            '/password/reset',
            '/auth/google',
            '/auth/apple',
            '/legal/documents',
        ]
        for p in expected_paths:
            self.assertIn(p, data['paths'])

    def test_docs_ui_accessible(self) -> None:
        """
        Swagger UI (/docs) should return 200 and HTML content.

        This test checks that the Swagger UI documentation endpoint is
        accessible and returns HTML content.
        """
        resp = self.client.get('/docs')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('text/html', resp.headers['content-type'])

    def test_cors_preflight_allows_web_credentials(self) -> None:
        """Browser preflight should be handled before auth routes."""
        resp = self.client.options(
            '/login',
            headers={
                'Origin': 'https://changdar-server.mooo.com',
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'content-type',
            },
        )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.headers.get('access-control-allow-origin'),
            'https://changdar-server.mooo.com',
        )
        self.assertEqual(
            resp.headers.get('access-control-allow-credentials'),
            'true',
        )

    def test_router_tags_registered(self) -> None:
        """
        Confirm all expected router tags are registered in the app.

        This test verifies that all expected router tags are present in
        the application's route definitions.
        """
        paths = self.app.openapi()['paths']
        tags = {
            tag
            for operations in paths.values()
            for operation in operations.values()
            if isinstance(operation, dict)
            for tag in operation.get('tags', [])
        }
        expected: set[str] = {
            'auth',
            'user-mgmt',
            'site-mgmt',
            'feature-mgmt',
            'group-mgmt',
            'stream-config',
            'legal',
            'playback',
        }
        self.assertTrue(
            expected.issubset(tags),
            msg=f"Missing router tag(s): {expected - tags}",
        )

    def test_bff_module_and_playback_routes_are_registered(self) -> None:
        paths = self.app.openapi()['paths']
        bff_paths = [route.path for route in bff_router.routes]

        self.assertIn('/bff/auth/session', paths)
        self.assertIn('/api/playback/walls', paths)
        self.assertNotIn('/api/media/sessions/batch', paths)
        self.assertNotIn('/bff/media/sessions/batch', paths)
        self.assertIn('/bff/{service}/{path:path}', bff_paths)

    def test_main_calls_uvicorn_run(self) -> None:
        """
        Test that the main() function calls uvicorn.run.

        This test patches uvicorn.run and checks that it is called when
        the app's main() function is invoked.
        """
        with patch('examples.db_management.app.uvicorn.run') as mock_run:
            app_module.main()
            mock_run.assert_called_once()


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.db_management.app\
    --cov-report=term-missing\
        tests/examples/db_management/app_test.py
'''
