from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware

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
        for p in getattr(cls, '_patchers', []):
            p.stop()

    # ---------- Tests ----------

    def test_cors_middleware_present(self) -> None:
        """
        The application should load CORSMiddleware and allow all origins.

        This test checks that CORSMiddleware is present in the app's middleware
        stack and that CORS headers are set correctly for OPTIONS requests.
        """
        cors = [
            m for m in self.app.user_middleware if m.cls is CORSMiddleware
        ]
        self.assertTrue(cors, msg='CORSMiddleware not found on app')

        # Simple check for CORS headers
        resp = self.client.options(
            '/openapi.json',
            headers={
                'Origin': 'http://example.com',
                'Access-Control-Request-Method': 'GET',
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.headers.get(
                'access-control-allow-origin',
            ), 'http://example.com',
        )

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
            '/list_sites',
            '/list_stream_configs',
            '/list_users',
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

    def test_router_tags_registered(self) -> None:
        """
        Confirm all expected router tags are registered in the app.

        This test verifies that all expected router tags are present in
        the application's route definitions.
        """
        tags: set[str] = {
            r.tags[0]
            for r in self.app.routes if getattr(r, 'tags', None)
        }
        expected: set[str] = {
            'auth',
            'user-mgmt',
            'site-mgmt',
            'feature-mgmt',
            'group-mgmt',
            'stream-config',
        }
        self.assertTrue(
            expected.issubset(tags),
            msg=f"Missing router tag(s): {expected - tags}",
        )

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
