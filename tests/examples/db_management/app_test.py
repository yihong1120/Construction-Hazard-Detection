# tests/examples/db_management/app_test.py
from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware


class AppIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # ---- 預先 mock 全域依賴，避免真 DB / Redis 連線 ----
        patch('examples.auth.database.get_db', new=lambda: None).start()
        patch(
            'examples.auth.redis_pool.get_redis_pool',
            new=lambda: None,
        ).start()

        # 以下權限相關依賴統一回傳 None，避免 422
        patch(
            'examples.db_management.deps.require_admin',
            new=lambda: None,
        ).start()
        patch(
            'examples.db_management.deps.require_super_admin',
            new=lambda: None,
        ).start()
        patch(
            'examples.db_management.deps.get_current_user',
            new=lambda: None,
        ).start()

        # 匯入 app（在所有依賴都被 patch 之後）
        from examples.db_management.app import app  # noqa: WPS433

        cls.app = app
        cls.client = TestClient(app)

    # ---------- Tests ----------

    def test_cors_middleware_present(self):
        """應用程式應加載 CORSMiddleware，且允許所有 Origin。"""
        cors = [
            m for m in self.app.user_middleware if m.cls is CORSMiddleware
        ]
        self.assertTrue(cors, msg='CORSMiddleware not found on app')

        # 簡單驗證 CORS headers
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

    def test_openapi_available(self):
        """OpenAPI schema endpoint 應該能回 200 並包含 title 與 paths。"""
        resp = self.client.get('/openapi.json')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('paths', data)
        # 大致檢查幾個 router 的經典路徑是否存在
        expected_paths = [
            '/list_features',
            '/list_groups',
            '/list_sites',
            '/list_stream_configs',
            '/list_users',
        ]
        for p in expected_paths:
            self.assertIn(p, data['paths'])

    def test_docs_ui_accessible(self):
        """Swagger UI (/docs) 應能正常回 200。"""
        resp = self.client.get('/docs')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('text/html', resp.headers['content-type'])

    def test_router_tags_registered(self):
        """確認 app 中已註冊所有預期的 tag。"""
        tags = {r.tags[0] for r in self.app.routes if getattr(r, 'tags', None)}
        expected = {
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


if __name__ == '__main__':
    unittest.main()
