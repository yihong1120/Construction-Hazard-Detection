from __future__ import annotations

import time
import unittest
from datetime import datetime
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi_jwt import JwtAuthorizationCredentials
from werkzeug.utils import secure_filename

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.user_service import _cache_ttl
from examples.auth.user_service import _user_sites_cache
from examples.violation_records.routers import get_user_sites_cached
from examples.violation_records.routers import router


class TestViolationRouters(unittest.IsolatedAsyncioTestCase):
    """
    A test suite for violation-related endpoints.
    """
    # fake_db is created in setUpClass as a SimpleNamespace with AsyncMocks
    client: ClassVar[TestClient]
    fake_db: ClassVar[SimpleNamespace]  # 明確定義 fake_db 作為類別屬性

    @classmethod
    def setUpClass(cls) -> None:
        """Set up once before running all tests in this suite."""
        super().setUpClass()

        # Create a FastAPI app, include the router
        app = FastAPI()
        app.include_router(router, prefix='/api')

        # Create a single fake DB instance with AsyncMock methods
        cls.fake_db = SimpleNamespace(
            execute=AsyncMock(),
            scalars=AsyncMock(),
        )

        async def override_get_db():
            return cls.fake_db

        # Override get_db
        app.dependency_overrides[get_db] = override_get_db

        # Override jwt_access dependency
        def override_jwt():
            return JwtAuthorizationCredentials(
                subject={'username': 'test_user'},
            )

        app.dependency_overrides[jwt_access] = override_jwt

        # Create a test client
        cls.client = TestClient(app)  # <== typed as ClassVar[TestClient]

    async def asyncSetUp(self) -> None:
        """
        Reset the fake DB's mocks before each test.
        """
        self.fake_db.execute.reset_mock()
        self.fake_db.execute.side_effect = None
        self.fake_db.scalars.reset_mock()
        self.fake_db.scalars.side_effect = None

    ###################################################
    # Helper methods for simulating DB results
    ###################################################
    def simulate_user_query(
        self,
        user_obj: object | None,
    ) -> None:
        """
        Simulate the DB returning a user object.

        Args:
            user_obj (MockUser | None): The mock user object to return.
        """
        self.fake_db.execute.side_effect = None
        self.fake_db.execute.return_value = self._exec_scalar(user_obj)

    def append_site_query(
        self,
        site_obj: object | None,
    ) -> None:
        """
        Append a site query result to the side_effect queue.

        Args:
            site_obj (MockSite | None): The mock site object to return.
        """
        cur = list(
            self.fake_db.execute.side_effect,
        ) if self.fake_db.execute.side_effect else []
        cur.append(self._exec_scalar(site_obj))
        self.fake_db.execute.side_effect = cur

    def append_count_query(self, count_val: int) -> None:
        """
        Append a count query result to the side_effect queue.

        Args:
            count_val (int): The integer count to return from db.execute().
        """
        cur = list(
            self.fake_db.execute.side_effect,
        ) if self.fake_db.execute.side_effect else []
        cur.append(self._exec_scalar(count_val))
        self.fake_db.execute.side_effect = cur

    def simulate_scalars_list(self, items: list) -> None:
        """
        Simulate db.scalars(stmt).all() returning a list of items.

        Args:
            items (list): A list of mock items to return.
        """
        self.fake_db.scalars.return_value = self._scalars_list(items)

    ###################################################
    # Lightweight helpers replacing Fake* classes
    ###################################################
    def _exec_scalar(self, value):
        """Return an object with scalar() -> value."""
        return SimpleNamespace(scalar=lambda: value)

    def _scalars_list(self, items: list):
        """Return an object with all() -> items."""
        return SimpleNamespace(all=lambda: items)

    # Domain object creators (replace former Mock* classes)
    def make_site(self, site_id: int, name: str):
        return SimpleNamespace(
            id=site_id,
            name=name,
            created_at=datetime(2023, 1, 1),
            updated_at=datetime(2023, 1, 2),
        )

    def make_user(self, username: str, sites: list):
        return SimpleNamespace(username=username, sites=sites)

    def make_violation(
        self,
        violation_id: int,
        site: str,
        detection_time: datetime | None = None,
        stream_name: str = 'Cam1',
        image_path: str = 'some.jpg',
    ):
        return SimpleNamespace(
            id=violation_id,
            site=site,
            stream_name=stream_name,
            detection_time=detection_time or datetime.now(),
            image_path=image_path,
            created_at=datetime(2023, 1, 3),
            detections_json='some detection',
            warnings_json='some warning',
            cone_polygon_json='some cone polygons',
            pole_polygon_json='some pole polygons',
        )

    ###################################################
    # Cache function tests
    ###################################################
    async def test_get_user_sites_cached_user_not_found(self) -> None:
        """
        Test get_user_sites_cached function when user is not found.
        """
        from examples.violation_records.routers import get_user_sites_cached
        from fastapi import HTTPException

        # Mock database to return None for user
        self.fake_db.execute.return_value = self._exec_scalar(None)

        with self.assertRaises(HTTPException) as context:
            await get_user_sites_cached('nonexistent_user', self.fake_db)

        self.assertEqual(context.exception.status_code, 404)
        self.assertEqual(context.exception.detail, 'User not found')

    async def test_get_user_sites_cached_success(self) -> None:
        """
        Test get_user_sites_cached function with successful user retrieval.
        """
        # Clear cache first
        _user_sites_cache.clear()

        # Create mock user with sites
        siteA = self.make_site(1, 'SiteA')
        siteB = self.make_site(2, 'SiteB')
        user = self.make_user('test_user', [siteA, siteB])

        # Mock database to return user
        self.fake_db.execute.return_value = self._exec_scalar(user)

        # Call function
        result = await get_user_sites_cached('test_user', self.fake_db)

        # Verify result
        self.assertEqual(result, ['SiteA', 'SiteB'])

        # Verify cache was populated
        self.assertIn('test_user', _user_sites_cache)

    async def test_get_user_sites_cached_cache_hit(self) -> None:
        """
        Test get_user_sites_cached function returns cached result.
        """
        # Pre-populate cache
        current_time = time.time()
        _user_sites_cache['cached_user'] = (['CachedSite'], current_time)

        # This should return cached result without calling DB
        result = await get_user_sites_cached('cached_user', self.fake_db)

        # Verify cached result returned
        self.assertEqual(result, ['CachedSite'])

        # Verify DB was not called (execute should not have been called)
        self.fake_db.execute.assert_not_called()

    async def test_get_user_sites_cached_cache_expired(self) -> None:
        """
        Test get_user_sites_cached function refreshes expired cache.
        """
        # Pre-populate cache with expired entry
        old_time = time.time() - _cache_ttl - 10  # expired
        _user_sites_cache['expired_user'] = (['OldSite'], old_time)

        # Create new mock user with different sites
        siteA = self.make_site(1, 'NewSite')
        user = self.make_user('expired_user', [siteA])

        # Mock database to return updated user
        self.fake_db.execute.return_value = self._exec_scalar(user)

        # Call function
        result = await get_user_sites_cached('expired_user', self.fake_db)

        # Verify new result returned (not cached)
        self.assertEqual(result, ['NewSite'])

        # Verify cache was updated with new values
        self.assertEqual(_user_sites_cache['expired_user'][0], ['NewSite'])

    async def test_get_my_sites_integration_cache(self) -> None:
        """
        Integration test for get_my_sites endpoint.
        """
        # Create mock user with sites
        siteA = self.make_site(1, 'SiteA')
        user = self.make_user('test_user', [siteA])

        # Mock the database query
        self.simulate_user_query(user)
        resp1 = self.client.get('/api/my_sites')
        self.assertEqual(resp1.status_code, 200)

        # Verify response content
        data = resp1.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['name'], 'SiteA')

    ###################################################
    # /api/my_sites Tests
    ###################################################
    async def test_get_my_sites_user_not_found(self) -> None:
        """
        If the DB returns no user, the endpoint should return 404.
        """
        self.simulate_user_query(None)
        resp = self.client.get('/api/my_sites')
        self.assertEqual(resp.status_code, 404)

    async def test_get_my_sites_empty_sites(self) -> None:
        """
        If the user has no sites, the endpoint should return an empty list.
        """
        user = self.make_user('test_user', [])
        self.simulate_user_query(user)
        resp = self.client.get('/api/my_sites')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [])

    async def test_get_my_sites_success(self) -> None:
        """
        If the user has multiple sites, return their info as a list of dicts.
        """
        siteA = self.make_site(1, 'SiteA')
        siteB = self.make_site(2, 'SiteB')
        user = self.make_user('test_user', [siteA, siteB])
        self.simulate_user_query(user)
        resp = self.client.get('/api/my_sites')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['name'], 'SiteA')
        self.assertEqual(data[1]['name'], 'SiteB')

    async def test_get_my_sites_missing_username(self) -> None:
        """
        Provide a JWT token with no 'username' in the subject dict to
        exercise `if not username: ...`.
        """
        def override_jwt_no_username():
            return JwtAuthorizationCredentials(subject={})
        self.client.app.dependency_overrides[jwt_access] = (
            override_jwt_no_username
        )

        resp = self.client.get('/api/my_sites')
        self.assertEqual(
            resp.status_code, 401,
            'Expected 401 when username is missing',
        )

        # Restore
        self.client.app.dependency_overrides[jwt_access] = (
            lambda: JwtAuthorizationCredentials(
                subject={'username': 'test_user'},
            )
        )

    ###################################################
    # /api/get_violation_image Tests
    ###################################################
    @patch('examples.violation_records.routers.Path')
    def test_get_violation_image_dotdot(self, mock_path: MagicMock) -> None:
        """
        If the path contains '..', return 400 for 'Invalid path'.
        """
        path_mock = MagicMock()
        path_mock.parts = ('..', 'secret.jpg')
        mock_path.return_value = path_mock
        resp = self.client.get(
            '/api/get_violation_image?image_path=../secret.jpg',
        )
        self.assertEqual(resp.status_code, 400)

    @patch('examples.violation_records.routers.Path')
    def test_get_violation_image_not_relative(
        self,
        mock_path: MagicMock,
    ) -> None:
        """
        If the path is not relative to the 'static' dir, return 403.

        Args:
            mock_path (MagicMock): Mocked Path class.
        """
        path_mock = MagicMock()
        path_mock.resolve.return_value = path_mock
        path_mock.__truediv__.return_value = path_mock
        path_mock.parts = ('some', 'path')
        # Ensure it is treated as a relative path by the code under test
        path_mock.is_absolute.return_value = False
        path_mock.relative_to.side_effect = ValueError('Not relative')
        path_mock.exists.return_value = True
        path_mock.suffix.lower.return_value = '.jpg'
        path_mock.name = 'some.jpg'
        mock_path.return_value = path_mock

        resp = self.client.get('/api/get_violation_image?image_path=some.jpg')
        self.assertEqual(resp.status_code, 403)

    @patch('examples.violation_records.routers.Path')
    def test_get_violation_image_not_found(self, mock_path: MagicMock) -> None:
        """
        If the requested file does not exist, return 404.
        """
        path_mock = MagicMock()
        path_mock.resolve.return_value = path_mock
        path_mock.__truediv__.return_value = path_mock
        path_mock.parts = ('some', 'path')
        # Ensure it is treated as a relative path by the code under test
        path_mock.is_absolute.return_value = False
        path_mock.is_relative_to.return_value = True
        path_mock.exists.return_value = False
        path_mock.suffix.lower.return_value = '.jpg'
        path_mock.name = 'some.jpg'
        mock_path.return_value = path_mock

        resp = self.client.get('/api/get_violation_image?image_path=some.jpg')
        self.assertEqual(resp.status_code, 404)

    @patch('examples.violation_records.routers.Path')
    def test_get_violation_image_success_png(
        self,
        mock_path: MagicMock,
    ) -> None:
        """
        If the file exists,
        return 200 with the correct content-type for a '.png'.

        Args:
            mock_path (MagicMock): Mocked Path class.
        """
        path_mock = MagicMock()
        path_mock.resolve.return_value = path_mock
        path_mock.__truediv__.return_value = path_mock
        path_mock.parts = ('valid', 'path')
        # Ensure it is treated as a relative path by the code under test
        path_mock.is_absolute.return_value = False
        path_mock.is_relative_to.return_value = True
        path_mock.exists.return_value = True
        path_mock.suffix.lower.return_value = '.png'
        path_mock.name = 'image.png'
        mock_path.return_value = path_mock

        resp = self.client.get('/api/get_violation_image?image_path=image.png')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers['content-type'], 'image/png')

    @patch('examples.violation_records.routers.Path')
    def test_get_violation_image_success_jpeg_and_header_sanitised(
        self,
        mock_path: MagicMock,
    ) -> None:
        """
        If the file exists with a .jpg/.jpeg, return 200 and ensure
        content-type is image/jpeg and the Content-Disposition filename is
        sanitised via secure_filename.

        Args:
            mock_path (MagicMock): Mocked Path class.
        """
        path_mock = MagicMock()
        path_mock.resolve.return_value = path_mock
        path_mock.__truediv__.return_value = path_mock
        path_mock.parts = ('valid', 'path')
        path_mock.is_absolute.return_value = False
        path_mock.exists.return_value = True
        path_mock.suffix.lower.return_value = '.jpg'
        unsafe_name = 'my image(1).JPG'
        path_mock.name = unsafe_name
        mock_path.return_value = path_mock

        resp = self.client.get('/api/get_violation_image?image_path=image.jpg')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers['content-type'], 'image/jpeg')
        # Header should contain sanitized filename
        self.assertIn(
            f'filename="{secure_filename(unsafe_name)}"',
            resp.headers['content-disposition'],
        )

    @patch('examples.violation_records.routers.Path')
    def test_get_violation_image_unsupported_file_type(
        self,
        mock_path: MagicMock,
    ) -> None:
        """
        If the file exists but has an unsupported extension, return 400.
        """
        path_mock = MagicMock()
        # Simulate safe relative path resolution
        path_mock.resolve.return_value = path_mock
        path_mock.__truediv__.return_value = path_mock
        path_mock.parts = ('valid', 'path')
        path_mock.is_absolute.return_value = False
        # Keep it under base_dir by not raising from relative_to
        # and simulate that the file exists
        path_mock.exists.return_value = True
        # Unsupported extension
        path_mock.suffix.lower.return_value = '.gif'
        path_mock.name = 'image.gif'
        mock_path.return_value = path_mock

        resp = self.client.get('/api/get_violation_image?image_path=image.gif')
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()['detail'], 'Unsupported file type')

    def test_get_violation_image_invalid_path_segment(self) -> None:
        """
        A path segment that sanitises to empty (e.g. '***') should return 400
        with 'Invalid path segment'.
        """
        resp = self.client.get('/api/get_violation_image?image_path=***')
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()['detail'], 'Invalid path segment')

    @patch('examples.violation_records.routers.Path')
    def test_get_violation_image_leading_static_normalised(
        self,
        mock_path: MagicMock,
    ) -> None:
        """
        If image_path starts with 'static/', it should be normalised to avoid
        constructing 'static/static/...'. Expect success for valid PNG.
        """
        path_mock = MagicMock()
        path_mock.resolve.return_value = path_mock
        path_mock.__truediv__.return_value = path_mock
        # Simulate parts starting with 'static' followed by valid subpath
        path_mock.parts = ('static', '2025-01-01', 'img.png')
        path_mock.is_absolute.return_value = False
        # Keep under base_dir and existing file
        path_mock.exists.return_value = True
        path_mock.suffix.lower.return_value = '.png'
        path_mock.name = 'img.png'
        mock_path.return_value = path_mock

        resp = self.client.get(
            '/api/get_violation_image?image_path=static/2025-01-01/img.png',
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers['content-type'], 'image/png')

    @patch('examples.violation_records.routers.Path')
    def test_get_violation_image_dot_segment_invalid(
        self,
        mock_path: MagicMock,
    ) -> None:
        """
        If a path contains a '.' segment, the per-segment validation should
        raise 400 'Invalid path' (covers the branch on line ~384).
        """
        path_mock = MagicMock()
        # Force parts to include a '.' so it isn't normalised away
        path_mock.parts = ('valid', '.', 'image.jpg')
        path_mock.is_absolute.return_value = False
        # No further attributes needed; it should fail before resolving
        mock_path.return_value = path_mock

        resp = self.client.get(
            '/api/get_violation_image?image_path=valid/./image.jpg',
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()['detail'], 'Invalid path')

    ###################################################
    # /api/violations Tests
    ###################################################
    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_get_violations_user_not_found(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """
        If no user is found in the DB, return 404.
        """
        # Mock get_user_sites_cached to raise 404 when user not found
        from fastapi import HTTPException
        mock_get_user_sites.side_effect = HTTPException(
            status_code=404, detail='User not found',
        )
        resp = self.client.get('/api/violations')
        self.assertEqual(resp.status_code, 404)

    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_get_violations_no_sites(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """
        If the user has no sites, return total=0 and items=[].
        """
        mock_get_user_sites.return_value = []
        resp = self.client.get('/api/violations')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {'total': 0, 'items': []})

    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_get_violations_site_id_403(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """
        If the user tries to access a site ID that does not match their site,
        return 403.
        """
        mock_get_user_sites.return_value = ['SiteA']
        # Mock site query to return a different site
        self.fake_db.execute.return_value = self._exec_scalar(
            self.make_site(2, 'SiteB'),
        )
        resp = self.client.get('/api/violations?site_id=2')
        self.assertEqual(resp.status_code, 403)

    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_get_violations_with_filters(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """
        If keyword, start_time, end_time, limit, and offset are provided,
        verify the response returns the expected data.
        """
        mock_get_user_sites.return_value = ['SiteA']
        # Mock count query and violations query
        self.fake_db.execute.return_value = self._exec_scalar(2)
        v1 = self.make_violation(123, 'SiteA')
        v2 = self.make_violation(456, 'SiteA')
        self.fake_db.scalars.return_value = self._scalars_list([v1, v2])

        params = {
            'keyword': 'cam',
            'start_time': '2023-01-01T00:00:00',
            'end_time': '2023-12-31T23:59:59',
            'limit': 5,
            'offset': 0,
        }
        resp = self.client.get('/api/violations', params=params)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['total'], 2)
        self.assertEqual(len(data['items']), 2)

    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_get_violations_success(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """
        If the user and site are valid, and there's 1 violation, return it.
        """
        mock_get_user_sites.return_value = ['SiteA']
        # Mock site query and count query
        self.fake_db.execute.side_effect = [
            self._exec_scalar(self.make_site(1, 'SiteA')),
            self._exec_scalar(1),
        ]
        viol = self.make_violation(101, 'SiteA')
        self.fake_db.scalars.return_value = self._scalars_list([viol])

        resp = self.client.get('/api/violations?site_id=1')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['total'], 1)
        self.assertEqual(len(data['items']), 1)
        self.assertEqual(data['items'][0]['id'], 101)

    ###################################################
    # /api/violations/{violation_id} Tests
    ###################################################
    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_get_single_violation_user_not_found(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """
        If there's no user, return 404.
        """
        from fastapi import HTTPException
        mock_get_user_sites.side_effect = HTTPException(
            status_code=404, detail='User not found',
        )
        resp = self.client.get('/api/violations/9999')
        self.assertEqual(resp.status_code, 404)

    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_get_single_violation_forbidden_violation_none(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """
        If the DB returns None for the violation, respond with 403 because it's
        not accessible.
        """
        mock_get_user_sites.return_value = ['SiteA']
        self.fake_db.execute.return_value = self._exec_scalar(None)
        resp = self.client.get('/api/violations/1234')
        self.assertEqual(resp.status_code, 403)

    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_get_single_violation_forbidden_site_mismatch(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """
        If the violation's site doesn't match the user's site,
        respond with 403.
        """
        mock_get_user_sites.return_value = ['SiteA']
        viol = self.make_violation(88, 'SiteB')
        self.fake_db.execute.return_value = self._exec_scalar(viol)
        resp = self.client.get('/api/violations/88')
        self.assertEqual(resp.status_code, 403)

    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_get_single_violation_success(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """
        If the violation matches the user's site,
        return 200 with violation data.
        """
        mock_get_user_sites.return_value = ['SiteA']
        viol = self.make_violation(77, 'SiteA')
        self.fake_db.execute.return_value = self._exec_scalar(viol)
        resp = self.client.get('/api/violations/77')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['id'], 77)
        self.assertEqual(data['site_name'], 'SiteA')
        self.assertIn('detection_items', data)
        self.assertIn('warnings', data)
        self.assertIn('cone_polygons', data)
        self.assertIn('pole_polygons', data)

    async def test_get_violations_missing_username(self) -> None:
        """
        If the JWT token has no 'username', return 401 for invalid token.
        """
        def override_jwt_no_username():
            return JwtAuthorizationCredentials(subject={})
        self.client.app.dependency_overrides[jwt_access] = (
            override_jwt_no_username
        )

        resp = self.client.get('/api/violations')
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()['detail'], 'Invalid token')

        # Restore
        self.client.app.dependency_overrides[jwt_access] = (
            lambda: JwtAuthorizationCredentials(
                subject={'username': 'test_user'},
            )
        )

    async def test_get_single_violation_missing_username(self) -> None:
        """
        If the JWT token has no 'username', return 401 for invalid token.
        """
        def override_jwt_no_username():
            return JwtAuthorizationCredentials(subject={})
        self.client.app.dependency_overrides[jwt_access] = (
            override_jwt_no_username
        )

        resp = self.client.get('/api/violations/123')
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()['detail'], 'Invalid token')

        # Restore
        self.client.app.dependency_overrides[jwt_access] = (
            lambda: JwtAuthorizationCredentials(
                subject={'username': 'test_user'},
            )
        )

    @patch('examples.violation_records.routers.Path')
    def test_get_violation_image_missing_username(
        self,
        mock_path: MagicMock,
    ) -> None:
        """
        If the token has no username, /api/get_violation_image should 401.

        Args:
            mock_path (MagicMock): Mocked Path class.
        """
        def override_jwt_no_username():
            return JwtAuthorizationCredentials(subject={})
        self.client.app.dependency_overrides[jwt_access] = (
            override_jwt_no_username
        )

        resp = self.client.get('/api/get_violation_image?image_path=some.jpg')
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()['detail'], 'Invalid token')

        # Restore
        self.client.app.dependency_overrides[jwt_access] = (
            lambda: JwtAuthorizationCredentials(
                subject={'username': 'test_user'},
            )
        )

    ###################################################
    # /api/upload tests
    ###################################################
    @patch(
        'examples.violation_records.routers.violation_manager.save_violation',
        new_callable=AsyncMock,
    )
    @patch('examples.violation_records.routers.UploadFile')
    async def test_upload_violation_success(
        self,
        mock_file_cls: MagicMock,
        mock_save_violation: AsyncMock,
    ) -> None:
        """
        A successful upload should return 200 with the violation_id.

        Args:
            mock_file_cls (MagicMock): Mocked UploadFile class.
            mock_save_violation (AsyncMock): Mocked save_violation function.
        """
        # 1) User can access "SiteA"
        siteA = self.make_site(1, 'SiteA')
        user = self.make_user('test_user', [siteA])
        self.simulate_user_query(user)

        # 2) Simulate reading normal bytes from the image
        mock_file_obj = MagicMock()
        mock_file_obj.read = AsyncMock(return_value=b'some_image_bytes')
        mock_file_cls.return_value = mock_file_obj

        # 3) The manager returns a new violation ID
        mock_save_violation.return_value = 123

        payload = {
            'site': 'SiteA',
            'stream_name': 'Cam1',
        }
        files = {'image': ('test.png', b'some_image_bytes', 'image/png')}
        resp = self.client.post('/api/upload', data=payload, files=files)

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['violation_id'], 123)

    async def test_upload_violation_missing_username(self) -> None:
        """
        If the token lacks 'username', /api/upload should 401.
        """
        def override_jwt_no_username():
            return JwtAuthorizationCredentials(subject={})
        self.client.app.dependency_overrides[jwt_access] = (
            override_jwt_no_username
        )

        files = {'image': ('test.png', b'some bytes', 'image/png')}
        resp = self.client.post(
            '/api/upload',
            data={'site': 'SiteA', 'stream_name': 'Cam1'},
            files=files,
        )
        self.assertEqual(resp.status_code, 401)

        # Restore
        self.client.app.dependency_overrides[jwt_access] = (
            lambda: JwtAuthorizationCredentials(
                subject={'username': 'test_user'},
            )
        )

    @patch('examples.violation_records.routers.UploadFile')
    async def test_upload_violation_no_access_site(
        self,
        mock_file_cls: MagicMock,
    ) -> None:
        """
        If the user can only access "SiteA" but requests "SiteB", return 403.

        Args:
            mock_file_cls (MagicMock): Mocked UploadFile class.
        """
        siteA = self.make_site(1, 'SiteA')
        user = self.make_user('test_user', [siteA])
        self.simulate_user_query(user)

        # The second DB call for site check returns None => triggers 403
        self.append_site_query(None)

        # Provide non-empty bytes so we don't fail with "empty image file"
        mock_file_obj = MagicMock()
        mock_file_obj.read = AsyncMock(return_value=b'dummy_image')
        mock_file_cls.return_value = mock_file_obj

        resp = self.client.post(
            '/api/upload',
            data={'site': 'SiteB', 'stream_name': 'Cam1'},
            files={'image': ('test.png', b'dummy_image', 'image/png')},
        )
        self.assertEqual(resp.status_code, 403)

    @patch('examples.violation_records.routers.UploadFile')
    async def test_upload_violation_empty_image(
        self,
        mock_file_cls: MagicMock,
    ) -> None:
        """
        Simulate reading empty bytes from the file => triggers 400,
        with detail 'Failed to read image file' after re-raising.

        Args:
            mock_file_cls (MagicMock): Mocked UploadFile class.
        """
        siteA = self.make_site(1, 'SiteA')
        user = self.make_user('test_user', [siteA])
        self.simulate_user_query(user)

        mock_file_obj = MagicMock()
        mock_file_obj.read = AsyncMock(return_value=b'')
        mock_file_cls.return_value = mock_file_obj

        resp = self.client.post(
            '/api/upload',
            data={'site': 'SiteA', 'stream_name': 'Cam1'},
            files={'image': ('test.png', b'', 'image/png')},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn('Failed to read image file', resp.text)

    @patch('examples.violation_records.routers.UploadFile')
    async def test_upload_violation_read_error(
        self,
        mock_file_cls: MagicMock,
    ) -> None:
        """
        Simulate an exception when reading the file => 400 with
        'Failed to read image file'.

        Args:
            mock_file_cls (MagicMock): Mocked UploadFile class.
        """
        siteA = self.make_site(1, 'SiteA')
        user = self.make_user('test_user', [siteA])
        self.simulate_user_query(user)

        mock_file_obj = MagicMock()
        mock_file_obj.read = AsyncMock(
            side_effect=Exception('Some read error'),
        )
        mock_file_cls.return_value = mock_file_obj

        resp = self.client.post(
            '/api/upload',
            data={'site': 'SiteA', 'stream_name': 'Cam1'},
            files={'image': ('test.png', b'', 'image/png')},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn('Failed to read image file', resp.text)

    @patch(
        'examples.violation_records.routers.violation_manager.save_violation',
        new_callable=AsyncMock,
    )
    @patch('examples.violation_records.routers.UploadFile')
    async def test_upload_violation_save_fail(
        self,
        mock_file_cls: MagicMock,
        mock_save_violation: AsyncMock,
    ) -> None:
        """
        If save_violation returns None, /api/upload should return 500.
        """
        siteA = self.make_site(1, 'SiteA')
        user = self.make_user('test_user', [siteA])
        self.simulate_user_query(user)

        mock_file_obj = MagicMock()
        mock_file_obj.read = AsyncMock(return_value=b'some_image_bytes')
        mock_file_cls.return_value = mock_file_obj

        # Force the violation_manager to return None => 500
        mock_save_violation.return_value = None

        resp = self.client.post(
            '/api/upload',
            data={'site': 'SiteA', 'stream_name': 'Cam1'},
            files={'image': ('test.png', b'some_image_bytes', 'image/png')},
        )
        self.assertEqual(resp.status_code, 500)
        self.assertIn('Failed to create violation record', resp.text)


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.violation_records.routers \
    --cov-report=term-missing tests/examples/violation_records/routers_test.py
'''
