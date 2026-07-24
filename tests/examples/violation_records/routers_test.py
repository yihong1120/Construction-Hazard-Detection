from __future__ import annotations

import json
import tempfile
import time
import unittest
from datetime import datetime
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import ClassVar
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.user_service import _cache_ttl
from examples.auth.user_service import _user_sites_cache
from examples.shared.filename_utils import sanitize_filename
from examples.violation_records.routers import _validate_analytics_range
from examples.violation_records.routers import get_user_sites_cached
from examples.violation_records.routers import router
from examples.violation_records.routers import upload_violation
from examples.violation_records.violation_manager import EmptyViolationImageError
from examples.violation_records.violation_manager import ViolationImageReadError


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
            add=MagicMock(),
            commit=AsyncMock(),
            refresh=AsyncMock(),
            rollback=AsyncMock(),
        )

        async def override_get_db() -> Any:
            """Support override_get_db."""
            return cls.fake_db

        # Override get_db
        app.dependency_overrides[get_db] = override_get_db

        # Override jwt_access dependency
        def override_jwt() -> Any:
            """Support override_jwt."""
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
        _user_sites_cache.clear()
        self.fake_db.execute.reset_mock()
        self.fake_db.execute.side_effect = None
        self.fake_db.scalars.reset_mock()
        self.fake_db.scalars.side_effect = None
        self.fake_db.add.reset_mock()
        self.fake_db.commit.reset_mock()
        self.fake_db.commit.side_effect = None
        self.fake_db.refresh.reset_mock()
        self.fake_db.refresh.side_effect = None
        self.fake_db.rollback.reset_mock()
        self.fake_db.rollback.side_effect = None

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
        if user_obj is None:
            self.fake_db.execute.return_value = self._exec_scalar(None)
            return

        self.fake_db.execute.side_effect = [
            self._exec_scalar(user_obj),
            self._exec_scalars_all(getattr(user_obj, 'sites', [])),
        ]

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
    def _exec_scalar(self, value: Any) -> Any:
        """Return an object with scalar() -> value."""
        return SimpleNamespace(
            scalar=lambda: value,
            scalar_one_or_none=lambda: value,
            unique=lambda: SimpleNamespace(
                scalars=lambda: SimpleNamespace(one_or_none=lambda: value),
            ),
        )

    def _exec_scalars_all(self, values: list) -> Any:
        """Return an object with scalars().all() -> values."""
        _scalars_ns = SimpleNamespace(
            all=lambda: values,
            unique=lambda: SimpleNamespace(all=lambda: values),
        )
        return SimpleNamespace(
            scalars=lambda: _scalars_ns,
            unique=lambda: SimpleNamespace(
                scalars=lambda: _scalars_ns,
            ),
        )

    def _exec_all(self, values: list) -> Any:
        """Return an object with all() -> values."""
        return SimpleNamespace(all=lambda: values)

    def _exec_scalars_feedbacks(self, values: list) -> Any:
        """Return an object with scalars().all() -> feedback rows."""
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: values))

    def _exec_first(self, value: Any) -> Any:
        """Return an object with first() -> value."""
        return SimpleNamespace(first=lambda: value)

    def _violation_row(self, violation: Any) -> Any:
        """Return a selected-column row matching the violations query."""
        return (
            violation.id,
            violation.site,
            violation.stream_name,
            violation.detection_time,
            violation.image_path,
            violation.created_at,
            violation.detections_json,
            violation.warnings_json,
            violation.cone_polygon_json,
            violation.pole_polygon_json,
            violation.is_flagged,
            violation.flag_reason,
            violation.flagged_by,
            violation.flagged_at,
            violation.review_status,
            violation.review_note,
            violation.reviewed_by,
            violation.reviewed_at,
            violation.feedback_note,
        )

    def _violation_row_with_total(self, violation: Any, total: int) -> Any:
        """Return a selected-column row with the window total appended."""
        return (*self._violation_row(violation), total)

    def _scalars_list(self, items: list) -> Any:
        """Return an object with all() -> items."""
        return SimpleNamespace(all=lambda: items)

    # Domain object creators (replace former Mock* classes)
    def make_site(self, site_id: int, name: str) -> Any:
        """Support make_site."""
        return SimpleNamespace(
            id=site_id,
            name=name,
            created_at=datetime(2023, 1, 1),
            updated_at=datetime(2023, 1, 2),
        )

    def make_user(
        self,
        username: str,
        sites: list,
        user_id: int = 1,
        role: str = 'user',
        group_id: int | None = 1,
    ) -> Any:
        """Support make_user."""
        return SimpleNamespace(
            id=user_id,
            username=username,
            role=role,
            group_id=group_id,
            sites=sites,
        )

    def make_violation(
        self,
        violation_id: int,
        site: str,
        detection_time: datetime | None = None,
        stream_name: str = 'Cam1',
        image_path: str = 'some.jpg',
    ) -> Any:
        """Support make_violation."""
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
            is_flagged=False,
            flag_reason=None,
            flagged_by=None,
            flagged_at=None,
            review_status='pending',
            review_note=None,
            reviewed_by=None,
            reviewed_at=None,
            feedback_note=None,
        )

    def make_feedback(
        self,
        feedback_id: int,
        violation_id: int,
        note: str | None = None,
    ) -> Any:
        """Support make_feedback."""
        return SimpleNamespace(
            id=feedback_id,
            violation_id=violation_id,
            feedback_type='false_positive',
            note=note,
            target_detection_id='det_0',
            original_label='class-5',
            corrected_label=None,
            original_bbox=[40, 40, 180, 210],
            corrected_bbox=None,
            model_version='yolo-v1',
            confidence=0.93,
            status='pending',
            user_id=9,
            created_at=datetime(2026, 6, 26, 1, 0, 0),
        )

    def make_review_audit(
        self,
        audit_id: int,
        violation_id: int,
        *,
        old_status: str | None = 'pending',
        new_status: str = 'resolved',
    ) -> Any:
        """Support make_review_audit."""
        return SimpleNamespace(
            id=audit_id,
            violation_id=violation_id,
            action='review_status_changed',
            old_status=old_status,
            new_status=new_status,
            review_note='Confirmed violation',
            flagged_reason='false_positive',
            reviewed_by=7,
            reviewed_at=datetime(2026, 6, 27, 10, 0, 0),
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
        self.fake_db.execute.side_effect = [
            self._exec_scalar(user),
            self._exec_scalars_all(user.sites),
        ]

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
        self.fake_db.execute.side_effect = [
            self._exec_scalar(user),
            self._exec_scalars_all(user.sites),
        ]

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
        def override_jwt_no_username() -> Any:
            """Support override_jwt_no_username."""
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

    @patch(
        'examples.violation_records.routers.get_user_sites_cached',
        new_callable=AsyncMock,
        return_value=['SiteA'],
    )
    @patch('examples.violation_records.routers.Path')
    def test_get_violation_image_success_png(
        self,
        mock_path: MagicMock,
        mock_get_user_sites: AsyncMock,
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
        mock_get_user_sites.return_value = ['SiteA']
        self.fake_db.execute.return_value = self._exec_scalar(1)

        resp = self.client.get('/api/get_violation_image?image_path=image.png')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers['content-type'], 'image/png')

    @patch(
        'examples.violation_records.routers.get_user_sites_cached',
        new_callable=AsyncMock,
        return_value=['SiteA'],
    )
    @patch('examples.violation_records.routers.Path')
    def test_get_violation_image_success_jpeg_and_header_sanitised(
        self,
        mock_path: MagicMock,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """
        If the file exists with a .jpg/.jpeg, return 200 and ensure
        content-type is image/jpeg and the Content-Disposition filename is
        sanitised before being returned.

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
        mock_get_user_sites.return_value = ['SiteA']
        self.fake_db.execute.return_value = self._exec_scalar(1)

        resp = self.client.get('/api/get_violation_image?image_path=image.jpg')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers['content-type'], 'image/jpeg')
        # Header should contain sanitized filename
        self.assertIn(
            f'filename="{sanitize_filename(unsafe_name)}"',
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

    @patch(
        'examples.violation_records.routers.get_user_sites_cached',
        new_callable=AsyncMock,
        return_value=['SiteA'],
    )
    @patch('examples.violation_records.routers.Path')
    def test_get_violation_image_leading_static_normalised(
        self,
        mock_path: MagicMock,
        mock_get_user_sites: AsyncMock,
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
        mock_get_user_sites.return_value = ['SiteA']
        self.fake_db.execute.return_value = self._exec_scalar(1)

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

    @patch(
        'examples.violation_records.routers.get_user_sites_cached',
        new_callable=AsyncMock,
        return_value=['SiteA'],
    )
    def test_get_violation_image_forbidden_when_not_owned(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """Image endpoints require the path to belong to an accessible record."""
        mock_get_user_sites.return_value = ['SiteA']
        self.fake_db.execute.return_value = self._exec_scalar(None)
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            source = base_dir / '2026-06-28' / 'image.png'
            source.parent.mkdir(parents=True)
            Image.new('RGB', (24, 24), color='white').save(source)
            with patch('examples.violation_records.routers.STATIC_DIR', base_dir):
                resp = self.client.get(
                    '/api/get_violation_image'
                    '?image_path=2026-06-28/image.png',
                )

        self.assertEqual(resp.status_code, 403)

    @patch(
        'examples.violation_records.routers.get_user_sites_cached',
        new_callable=AsyncMock,
        return_value=['SiteA'],
    )
    def test_get_violation_thumbnail_generates_cached_jpeg(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """Thumbnail endpoint generates a small cached JPEG for list cards."""
        mock_get_user_sites.return_value = ['SiteA']
        self.fake_db.execute.return_value = self._exec_scalar(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            source = base_dir / '2026-06-28' / 'image.png'
            source.parent.mkdir(parents=True)
            Image.new('RGB', (800, 600), color='white').save(source)

            with patch('examples.violation_records.routers.STATIC_DIR', base_dir):
                resp = self.client.get(
                    '/api/get_violation_thumbnail'
                    '?image_path=2026-06-28/image.png',
                )
                thumbnail = (
                    base_dir
                    / '_thumbnails'
                    / '2026-06-28'
                    / 'image.jpg'
                )
                thumbnail_exists = thumbnail.exists()

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers['content-type'], 'image/jpeg')
        self.assertTrue(thumbnail_exists)

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
        self.assertEqual(
            resp.json(),
            {'total': 0, 'items': [], 'next_cursor': None},
        )

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
        v1 = self.make_violation(123, 'SiteA')
        v2 = self.make_violation(456, 'SiteA')
        self.fake_db.execute.return_value = self._exec_all([
            self._violation_row_with_total(v1, 2),
            self._violation_row_with_total(v2, 2),
        ])

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

    @patch('examples.violation_records.routers.load_user_with_effective_sites')
    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_get_violations_with_camera_and_type_filters(
        self,
        mock_get_user_sites: AsyncMock,
        mock_load_user: AsyncMock,
    ) -> None:
        """Camera IDs and canonical type codes can narrow the list together."""
        site_a = self.make_site(1, 'SiteA')
        admin = self.make_user('test_user', [site_a], role='admin')
        mock_get_user_sites.return_value = ['SiteA']
        mock_load_user.return_value = (admin, [site_a])
        violation = self.make_violation(101, 'SiteA')
        self.fake_db.execute.side_effect = [
            self._exec_first((10, 1, 'SiteA')),
            self._exec_all([self._violation_row_with_total(violation, 1)]),
        ]

        resp = self.client.get(
            '/api/violations',
            params={
                'stream_id': '10',
                'violation_type': 'near_vehicle',
            },
        )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['total'], 1)
        self.assertEqual(self.fake_db.execute.await_count, 2)

    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_get_violations_rejects_non_numeric_stream_id(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """The API accepts only stable numeric stream configuration IDs."""
        mock_get_user_sites.return_value = ['SiteA']
        site_a = self.make_site(1, 'SiteA')
        admin = self.make_user('test_user', [site_a], role='admin')
        with patch(
            'examples.violation_records.routers.load_user_with_effective_sites',
            new=AsyncMock(return_value=(admin, [site_a])),
        ):
            resp = self.client.get('/api/violations?stream_id=Cam1')

        self.assertEqual(resp.status_code, 422)
        self.assertEqual(
            resp.json()['detail'],
            'stream_id must be a positive stream configuration ID',
        )

    @patch('examples.violation_records.routers.load_user_with_effective_sites')
    async def test_get_violation_filter_options_success(
        self,
        mock_load_user: AsyncMock,
    ) -> None:
        """Filter options expose only cameras in the selected site scope."""
        site_a = self.make_site(1, 'SiteA')
        admin = self.make_user('test_user', [site_a], role='admin')
        mock_load_user.return_value = (admin, [site_a])
        self.fake_db.execute.return_value = self._exec_all([
            (10, 'Cam A'),
            (11, 'Cam B'),
        ])

        resp = self.client.get('/api/violations/filter-options?site_id=1')

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.json()['cameras'], [
                {'stream_id': '10', 'name': 'Cam A'},
                {'stream_id': '11', 'name': 'Cam B'},
            ],
        )
        self.assertEqual(
            resp.json()['violation_types'][0], {
                'code': 'no_safety_helmet',
                'label': '未戴安全帽',
            },
        )

    @patch('examples.violation_records.routers.load_user_with_effective_sites')
    async def test_get_violation_filter_options_rejects_other_group(
        self,
        mock_load_user: AsyncMock,
    ) -> None:
        """A non-super-admin cannot select another group's cameras."""
        site_a = self.make_site(1, 'SiteA')
        admin = self.make_user(
            'test_user',
            [site_a],
            role='admin',
            group_id=1,
        )
        mock_load_user.return_value = (admin, [site_a])

        resp = self.client.get(
            '/api/violations/filter-options?site_id=1&group_id=2',
        )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()['detail'], 'No access to group_id')

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
        viol = self.make_violation(101, 'SiteA')
        self.fake_db.execute.side_effect = [
            self._exec_scalar(self.make_site(1, 'SiteA')),
            self._exec_all([self._violation_row_with_total(viol, 1)]),
        ]

        resp = self.client.get('/api/violations?site_id=1')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['total'], 1)
        self.assertEqual(len(data['items']), 1)
        self.assertEqual(data['items'][0]['id'], 101)
        self.assertIn(
            '/api/get_violation_thumbnail',
            data['items'][0]['thumbnail_url'],
        )
        self.assertIn(
            '/api/get_violation_image',
            data['items'][0]['image_url'],
        )

    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_get_violations_cursor_pagination_returns_next_cursor(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """Cursor pagination returns one page plus a next cursor."""
        mock_get_user_sites.return_value = ['SiteA']
        first = self.make_violation(
            456,
            'SiteA',
            detection_time=datetime(2026, 6, 28, 12, 0, 0),
        )
        second = self.make_violation(
            123,
            'SiteA',
            detection_time=datetime(2026, 6, 28, 11, 0, 0),
        )
        self.fake_db.execute.return_value = self._exec_all([
            self._violation_row_with_total(first, 2),
            self._violation_row_with_total(second, 2),
        ])

        resp = self.client.get('/api/violations?limit=1')

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['total'], 2)
        self.assertEqual(len(data['items']), 1)
        self.assertEqual(data['items'][0]['id'], 456)
        self.assertIsInstance(data['next_cursor'], str)
        self.assertTrue(data['next_cursor'])

    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_get_violations_invalid_cursor(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """Invalid cursor payloads are rejected."""
        mock_get_user_sites.return_value = ['SiteA']

        resp = self.client.get('/api/violations?cursor=not-a-cursor')

        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()['detail'], 'Invalid cursor')

    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_get_violations_empty_tail_page_counts_total(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """If offset is beyond the last row, total remains accurate."""
        mock_get_user_sites.return_value = ['SiteA']
        self.fake_db.execute.side_effect = [
            self._exec_all([]),
            self._exec_scalar(3),
        ]

        resp = self.client.get('/api/violations?offset=100')

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.json(),
            {'total': 3, 'items': [], 'next_cursor': None},
        )
        self.assertEqual(self.fake_db.execute.await_count, 2)

    @patch('examples.violation_records.routers.load_user_with_effective_sites')
    async def test_get_flagged_violations_admin_success(
        self,
        mock_load_user: AsyncMock,
    ) -> None:
        """Admins can list flagged records inside their effective sites."""
        siteA = self.make_site(1, 'SiteA')
        user = self.make_user(
            'test_user',
            [siteA],
            user_id=7,
            role='admin',
            group_id=1,
        )
        mock_load_user.return_value = (user, [siteA])

        viol = self.make_violation(101, 'SiteA')
        viol.is_flagged = True
        viol.flag_reason = 'false_positive'
        viol.flagged_by = 9
        viol.flagged_at = datetime(2026, 6, 25, 10, 0, 0)
        viol.review_status = 'pending'
        self.fake_db.execute.return_value = self._exec_all([
            self._violation_row_with_total(viol, 1),
        ])

        resp = self.client.get(
            '/api/violations?flagged=true&review_status=pending',
        )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['total'], 1)
        self.assertTrue(data['items'][0]['is_flagged'])
        self.assertEqual(data['items'][0]['flag_reason'], 'false_positive')
        self.assertEqual(data['items'][0]['review_status'], 'pending')

    @patch('examples.violation_records.routers.load_user_with_effective_sites')
    async def test_get_flagged_violations_regular_user_forbidden(
        self,
        mock_load_user: AsyncMock,
    ) -> None:
        """Regular users cannot access the review queue."""
        siteA = self.make_site(1, 'SiteA')
        user = self.make_user('test_user', [siteA], role='user')
        mock_load_user.return_value = (user, [siteA])

        resp = self.client.get('/api/violations?flagged=true')

        self.assertEqual(resp.status_code, 403)
        self.fake_db.execute.assert_not_called()

    @patch('examples.violation_records.routers.load_user_with_effective_sites')
    async def test_get_next_review_violation_success(
        self,
        mock_load_user: AsyncMock,
    ) -> None:
        """Admins can fetch the next pending record inside their scope."""
        siteA = self.make_site(1, 'SiteA')
        reviewer = self.make_user(
            'test_user',
            [siteA],
            user_id=7,
            role='admin',
            group_id=1,
        )
        mock_load_user.return_value = (reviewer, [siteA])

        viol = self.make_violation(77, 'SiteA')
        viol.is_flagged = True
        viol.flag_reason = 'false_positive'
        viol.flagged_at = datetime(2026, 6, 26, 1, 0, 0)
        viol.feedback_note = '測試'
        feedback = self.make_feedback(1, 77, note='測試')
        self.fake_db.execute.side_effect = [
            self._exec_first(self._violation_row(viol)),
            self._exec_scalars_feedbacks([feedback]),
            self._exec_scalars_feedbacks([]),
        ]

        resp = self.client.get('/api/violations/next?review_status=pending')

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['id'], 77)
        self.assertEqual(data['feedback_note'], '測試')
        self.assertEqual(data['feedbacks'][0]['note'], '測試')
        self.assertEqual(data['review_audit_logs'], [])

    @patch('examples.violation_records.routers.load_user_with_effective_sites')
    async def test_get_next_review_violation_empty(
        self,
        mock_load_user: AsyncMock,
    ) -> None:
        """Next pending returns null when the review queue is empty."""
        siteA = self.make_site(1, 'SiteA')
        reviewer = self.make_user('test_user', [siteA], role='admin')
        mock_load_user.return_value = (reviewer, [siteA])
        self.fake_db.execute.return_value = self._exec_first(None)

        resp = self.client.get('/api/violations/next?review_status=pending')

        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.json())

    ###################################################
    # /api/violations/analytics Tests
    ###################################################
    @patch(
        'examples.violation_records.routers.load_user_with_effective_sites',
    )
    async def test_get_violation_analytics_success(
        self,
        mock_load_user: AsyncMock,
    ) -> None:
        """Analytics returns aggregate counts without row-level details."""
        site_a = self.make_site(1, 'SiteA')
        site_b = self.make_site(2, 'SiteB')
        admin = self.make_user(
            'test_user',
            [site_a, site_b],
            role='admin',
        )
        mock_load_user.return_value = (admin, [site_a, site_b])
        self.fake_db.execute.side_effect = [
            self._exec_scalar(128),
            self._exec_scalar(12),
            self._exec_all([
                (datetime(2026, 6, 20), 18),
                (datetime(2026, 6, 21), 22),
            ]),
            self._exec_all([
                (1, 'SiteA', 80),
                (2, 'SiteB', 48),
            ]),
            self._exec_all([(8, 12), (9, 21), (10, 9)]),
            self._exec_scalar(64),
            self._exec_scalar(30),
            self._exec_scalar(34),
            self._exec_scalar(0),
            self._exec_scalar(0),
            self._exec_scalar(0),
            self._exec_scalar(0),
        ]

        resp = self.client.get(
            '/api/violations/analytics',
            params={
                'start': '2026-06-01T00:00:00Z',
                'end': '2026-06-24T23:59:59Z',
            },
        )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['summary']['total'], 128)
        self.assertEqual(data['summary']['today'], 12)
        self.assertEqual(data['summary']['top_site']['site_id'], 1)
        self.assertEqual(
            data['summary']['top_type']['type'],
            'no_safety_helmet',
        )
        self.assertEqual(
            data['trend'][0], {
                'bucket': '2026-06-20',
                'count': 18,
            },
        )
        self.assertEqual(data['by_type'][0]['count'], 64)
        self.assertEqual(data['by_site'][1]['site_name'], 'SiteB')
        self.assertEqual(data['by_hour'][1], {'hour': 9, 'count': 21})
        self.assertNotIn('items', data)
        self.assertNotIn('image_path', str(data))

    @patch(
        'examples.violation_records.routers.load_user_with_effective_sites',
    )
    async def test_get_violation_analytics_applies_camera_and_type_everywhere(
        self,
        mock_load_user: AsyncMock,
    ) -> None:
        """Every analytics aggregate shares the requested camera/type scope."""
        site_a = self.make_site(1, 'SiteA')
        admin = self.make_user(
            'test_user',
            [site_a],
            role='admin',
            group_id=1,
        )
        mock_load_user.return_value = (admin, [site_a])
        self.fake_db.execute.side_effect = [
            self._exec_scalar('SiteA'),
            self._exec_first((10, 1, 'SiteA')),
            self._exec_scalar(7),
            self._exec_scalar(2),
            self._exec_all([(datetime(2026, 7, 22), 7)]),
            self._exec_all([(1, 'SiteA', 7)]),
            self._exec_all([(9, 7)]),
            self._exec_scalar(7),
        ]

        resp = self.client.get(
            '/api/violations/analytics',
            params={
                'site_id': 1,
                'stream_id': '10',
                'violation_type': 'near_vehicle',
                'start': '2026-07-01T00:00:00Z',
                'end': '2026-07-23T00:00:00Z',
                'bucket': 'day',
            },
        )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['summary']['total'], 7)
        self.assertEqual(data['summary']['today'], 2)
        self.assertEqual(
            data['by_type'], [{
                'type': 'near_vehicle',
                'label': '人員靠近車輛',
                'count': 7,
            }],
        )

        aggregate_statements = [
            str(execute_call.args[0])
            for execute_call in self.fake_db.execute.await_args_list[2:]
        ]
        for statement in aggregate_statements:
            self.assertIn('violations.stream_config_id', statement)
            self.assertIn('violations.violation_type_codes', statement)

    @patch(
        'examples.violation_records.routers.load_user_with_effective_sites',
    )
    async def test_get_violation_analytics_empty(
        self,
        mock_load_user: AsyncMock,
    ) -> None:
        """No matching rows returns 200 with empty aggregate arrays."""
        site_a = self.make_site(1, 'SiteA')
        admin = self.make_user('test_user', [site_a], role='admin')
        mock_load_user.return_value = (admin, [site_a])
        self.fake_db.execute.return_value = self._exec_scalar(0)

        resp = self.client.get(
            '/api/violations/analytics',
            params={
                'start': '2026-06-01T00:00:00Z',
                'end': '2026-06-24T23:59:59Z',
            },
        )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.json(),
            {
                'summary': {
                    'total': 0,
                    'today': 0,
                    'top_site': None,
                    'top_type': None,
                },
                'trend': [],
                'by_type': [],
                'by_site': [],
                'by_hour': [],
            },
        )

    @patch(
        'examples.violation_records.routers.load_user_with_effective_sites',
    )
    async def test_get_violation_analytics_prefix_stripped_alias(
        self,
        mock_load_user: AsyncMock,
    ) -> None:
        """The nginx /hazard/api/violations prefix maps to /analytics."""
        site_a = self.make_site(1, 'SiteA')
        super_admin = self.make_user(
            'test_user',
            [site_a],
            role='super_admin',
        )
        mock_load_user.return_value = (super_admin, [site_a])
        self.fake_db.execute.return_value = self._exec_scalar(0)

        resp = self.client.get(
            '/api/analytics',
            params={
                'start': '2026-06-01T00:00:00Z',
                'end': '2026-06-24T23:59:59Z',
            },
        )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['summary']['total'], 0)

    @patch(
        'examples.violation_records.routers.load_user_with_effective_sites',
    )
    async def test_get_violation_analytics_site_id_403(
        self,
        mock_load_user: AsyncMock,
    ) -> None:
        """Filtering by an inaccessible site_id is rejected."""
        site_a = self.make_site(1, 'SiteA')
        admin = self.make_user('test_user', [site_a], role='admin')
        mock_load_user.return_value = (admin, [site_a])
        self.fake_db.execute.return_value = self._exec_scalar('SiteB')

        resp = self.client.get(
            '/api/violations/analytics',
            params={
                'start': '2026-06-01T00:00:00Z',
                'end': '2026-06-24T23:59:59Z',
                'site_id': 2,
            },
        )

        self.assertEqual(resp.status_code, 403)

    @patch(
        'examples.violation_records.routers.load_user_with_effective_sites',
    )
    async def test_get_violation_analytics_range_too_large(
        self,
        mock_load_user: AsyncMock,
    ) -> None:
        """Analytics queries are capped to five calendar years."""
        site_a = self.make_site(1, 'SiteA')
        admin = self.make_user('test_user', [site_a], role='admin')
        mock_load_user.return_value = (admin, [site_a])

        resp = self.client.get(
            '/api/violations/analytics',
            params={
                'start': '2020-01-01T00:00:00Z',
                'end': '2026-06-24T23:59:59Z',
            },
        )

        self.assertEqual(resp.status_code, 422)
        self.fake_db.execute.assert_not_called()

    async def test_validate_analytics_range_allows_five_calendar_years(
        self,
    ) -> None:
        """An exact five-year range remains valid across a leap day."""
        start = datetime(2020, 2, 29, tzinfo=timezone.utc)
        end = datetime(2025, 2, 28, tzinfo=timezone.utc)

        start_utc, end_utc = _validate_analytics_range(start, end)

        self.assertEqual(start_utc, start)
        self.assertEqual(end_utc, end)

    @patch(
        'examples.violation_records.routers.load_user_with_effective_sites',
    )
    async def test_get_violation_analytics_regular_user_forbidden(
        self,
        mock_load_user: AsyncMock,
    ) -> None:
        """A normal user cannot retrieve aggregate data by calling the API."""
        site_a = self.make_site(1, 'SiteA')
        regular_user = self.make_user('test_user', [site_a], role='user')
        mock_load_user.return_value = (regular_user, [site_a])

        resp = self.client.get(
            '/api/violations/analytics',
            params={
                'start': '2026-06-01T00:00:00Z',
                'end': '2026-06-24T23:59:59Z',
            },
        )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(
            resp.json(), {
                'detail': 'violation_analytics_forbidden',
            },
        )
        self.fake_db.execute.assert_not_called()

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
        viol.feedback_note = '測試'
        viol.detections_json = json.dumps([
            [40, 40, 180, 210, 0.93, 5, 12],
        ])
        feedback = self.make_feedback(1, 77, note='測試')
        self.fake_db.execute.side_effect = [
            self._exec_scalar(viol),
            self._exec_scalars_feedbacks([feedback]),
        ]
        resp = self.client.get('/api/violations/77')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['id'], 77)
        self.assertEqual(data['site_name'], 'SiteA')
        self.assertIn('detection_items', data)
        self.assertIn('warnings', data)
        self.assertIn('cone_polygons', data)
        self.assertIn('pole_polygons', data)
        self.assertEqual(data['feedback_note'], '測試')
        self.assertIsNone(data['review_status'])
        self.assertEqual(data['detections'][0]['id'], 'det_0')
        self.assertEqual(data['feedbacks'][0]['note'], '測試')
        self.assertEqual(data['feedbacks'][0]['target_detection_id'], 'det_0')

    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_get_single_violation_includes_normalized_overlay_objects(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """Detail response exposes normalized overlay objects for painters."""
        mock_get_user_sites.return_value = ['SiteA']
        viol = self.make_violation(77, 'SiteA')
        viol.detections_json = json.dumps([
            [40, 40, 180, 210, 0.93, 5, 12],
        ])
        feedback = self.make_feedback(1, 77, note='測試')
        self.fake_db.execute.side_effect = [
            self._exec_scalar(viol),
            self._exec_scalars_feedbacks([feedback]),
        ]

        with patch(
            'examples.violation_records.routers._image_size_for_violation',
            return_value=(400, 400),
        ):
            resp = self.client.get('/api/violations/77')

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        overlay = data['overlay_objects'][0]
        self.assertEqual(overlay['object_id'], 'det_0')
        self.assertTrue(overlay['is_flagged'])
        self.assertEqual(overlay['flag_reason'], 'false_positive')
        self.assertEqual(overlay['flag_note'], '測試')
        self.assertEqual(overlay['bbox']['x'], 0.1)
        self.assertEqual(overlay['bbox']['y'], 0.1)
        self.assertEqual(overlay['bbox']['w'], 0.35)

    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_get_single_flagged_violation_includes_audit_logs(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """Flagged record details include review audit history."""
        mock_get_user_sites.return_value = ['SiteA']
        viol = self.make_violation(77, 'SiteA')
        viol.is_flagged = True
        viol.flag_reason = 'false_positive'
        feedback = self.make_feedback(1, 77, note='測試')
        audit = self.make_review_audit(5, 77)
        self.fake_db.execute.side_effect = [
            self._exec_scalar(viol),
            self._exec_scalars_feedbacks([feedback]),
            self._exec_scalars_all([audit]),
        ]

        resp = self.client.get('/api/violations/77')

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['review_audit_logs'][0]['id'], 5)
        self.assertEqual(
            data['review_audit_logs'][0]['actor_user_id'],
            7,
        )
        self.assertEqual(
            data['review_audit_logs'][0]['flagged_reason'],
            'false_positive',
        )

    async def test_get_violations_missing_username(self) -> None:
        """
        If the JWT token has no 'username', return 401 for invalid token.
        """
        def override_jwt_no_username() -> Any:
            """Support override_jwt_no_username."""
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
        def override_jwt_no_username() -> Any:
            """Support override_jwt_no_username."""
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

    ###################################################
    # /api/violations/{violation_id}/feedback Tests
    ###################################################
    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_submit_violation_feedback_success(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """A valid feedback request creates a pending feedback row."""
        mock_get_user_sites.return_value = ['SiteA']
        viol = self.make_violation(77, 'SiteA')
        viol.detections_json = json.dumps([
            [40, 40, 180, 210, 0.93, 5, 12],
        ])
        self.fake_db.execute.side_effect = [
            self._exec_scalar(viol),
            self._exec_scalar(9),
        ]

        def refresh_feedback(feedback: Any) -> None:
            feedback.id = 321

        self.fake_db.refresh.side_effect = refresh_feedback

        resp = self.client.post(
            '/api/violations/77/feedback',
            json={
                'type': 'false_positive',
                'target_detection_id': 'det_0',
                'original_label': 'class-5',
                'original_bbox': [40, 40, 180, 210],
                'confidence': 0.93,
                'model_version': 'yolo-v1',
                'note': 'shadow',
            },
        )

        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data['id'], 321)
        self.assertEqual(data['violation_id'], 77)
        self.assertEqual(data['type'], 'false_positive')
        self.assertEqual(data['status'], 'pending')

        feedback = self.fake_db.add.call_args.args[0]
        self.assertEqual(feedback.feedback_type, 'false_positive')
        self.assertEqual(feedback.user_id, 9)
        self.assertEqual(feedback.target_detection_id, 'det_0')
        self.assertEqual(feedback.original_bbox, [40.0, 40.0, 180.0, 210.0])
        self.assertEqual(feedback.note, 'shadow')
        self.fake_db.commit.assert_awaited_once()

    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_submit_violation_feedback_accepts_note_only(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """A false-positive feedback can be a record-level note."""
        mock_get_user_sites.return_value = ['SiteA']
        viol = self.make_violation(77, 'SiteA')
        self.fake_db.execute.side_effect = [
            self._exec_scalar(viol),
            self._exec_scalar(9),
        ]

        def refresh_feedback(feedback: Any) -> None:
            feedback.id = 322

        self.fake_db.refresh.side_effect = refresh_feedback

        resp = self.client.post(
            '/api/violations/77/feedback',
            json={
                'type': 'false_positive',
                'note': '測試',
            },
        )

        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data['note'], '測試')
        self.assertEqual(data['status'], 'pending')

        feedback = self.fake_db.add.call_args.args[0]
        self.assertEqual(feedback.feedback_type, 'false_positive')
        self.assertEqual(feedback.note, '測試')
        self.assertIsNone(feedback.target_detection_id)
        self.assertTrue(viol.is_flagged)
        self.assertEqual(viol.review_status, 'pending')

    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_submit_violation_feedback_forbidden_record(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """Feedback cannot be submitted for an inaccessible record."""
        mock_get_user_sites.return_value = ['SiteA']
        self.fake_db.execute.return_value = self._exec_scalar(None)

        resp = self.client.post(
            '/api/violations/77/feedback',
            json={
                'type': 'false_positive',
                'original_bbox': [40, 40, 180, 210],
            },
        )

        self.assertEqual(resp.status_code, 403)
        self.fake_db.add.assert_not_called()

    @patch('examples.violation_records.routers.get_user_sites_cached')
    async def test_submit_violation_feedback_rejects_unknown_detection(
        self,
        mock_get_user_sites: AsyncMock,
    ) -> None:
        """A target_detection_id must belong to parsed record detections."""
        mock_get_user_sites.return_value = ['SiteA']
        viol = self.make_violation(77, 'SiteA')
        viol.detections_json = json.dumps([
            [40, 40, 180, 210, 0.93, 5, 12],
        ])
        self.fake_db.execute.return_value = self._exec_scalar(viol)

        resp = self.client.post(
            '/api/violations/77/feedback',
            json={
                'type': 'false_positive',
                'target_detection_id': 'det_99',
            },
        )

        self.assertEqual(resp.status_code, 422)
        self.assertEqual(
            resp.json()['detail'],
            'target_detection_id does not belong to this violation',
        )
        self.fake_db.add.assert_not_called()

    async def test_submit_violation_feedback_missing_username(self) -> None:
        """Feedback submission requires an authenticated username."""
        def override_jwt_no_username() -> Any:
            """Support override_jwt_no_username."""
            return JwtAuthorizationCredentials(subject={})
        self.client.app.dependency_overrides[jwt_access] = (
            override_jwt_no_username
        )

        resp = self.client.post(
            '/api/violations/77/feedback',
            json={
                'type': 'false_positive',
                'original_bbox': [40, 40, 180, 210],
            },
        )
        self.assertEqual(resp.status_code, 401)

        self.client.app.dependency_overrides[jwt_access] = (
            lambda: JwtAuthorizationCredentials(
                subject={'username': 'test_user'},
            )
        )

    ###################################################
    # /api/violations/{violation_id}/audit-log Tests
    ###################################################
    @patch('examples.violation_records.routers.load_user_with_effective_sites')
    async def test_get_violation_audit_log_admin_success(
        self,
        mock_load_user: AsyncMock,
    ) -> None:
        """Admins can read audit logs for records in scope."""
        siteA = self.make_site(1, 'SiteA')
        reviewer = self.make_user(
            'test_user',
            [siteA],
            user_id=7,
            role='admin',
            group_id=1,
        )
        mock_load_user.return_value = (reviewer, [siteA])
        audit = self.make_review_audit(5, 77)
        self.fake_db.execute.side_effect = [
            self._exec_scalar(77),
            self._exec_scalars_all([audit]),
        ]

        resp = self.client.get('/api/violations/77/audit-log')

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data[0]['action'], 'review_status_changed')
        self.assertEqual(data[0]['actor_user_id'], 7)
        self.assertEqual(data[0]['note'], 'Confirmed violation')

    @patch('examples.violation_records.routers.load_user_with_effective_sites')
    async def test_get_violation_audit_log_regular_user_forbidden(
        self,
        mock_load_user: AsyncMock,
    ) -> None:
        """Regular users cannot read review audit logs."""
        siteA = self.make_site(1, 'SiteA')
        user = self.make_user('test_user', [siteA], role='user')
        mock_load_user.return_value = (user, [siteA])

        resp = self.client.get('/api/violations/77/audit-log')

        self.assertEqual(resp.status_code, 403)
        self.fake_db.execute.assert_not_called()

    ###################################################
    # /api/violations/{violation_id}/review Tests
    ###################################################
    @patch('examples.violation_records.routers.load_user_with_effective_sites')
    async def test_review_violation_admin_success(
        self,
        mock_load_user: AsyncMock,
    ) -> None:
        """Admins can update review status for records in scope."""
        siteA = self.make_site(1, 'SiteA')
        reviewer = self.make_user(
            'test_user',
            [siteA],
            user_id=7,
            role='admin',
            group_id=1,
        )
        mock_load_user.return_value = (reviewer, [siteA])

        viol = self.make_violation(77, 'SiteA')
        viol.is_flagged = True
        viol.flag_reason = 'false_positive'
        viol.review_status = 'pending'
        self.fake_db.execute.side_effect = [
            self._exec_scalar(viol),
            self._exec_scalar('測試'),
        ]

        resp = self.client.patch(
            '/api/violations/77/review',
            json={
                'review_status': 'resolved',
                'review_note': 'Confirmed violation',
            },
        )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['review_status'], 'resolved')
        self.assertEqual(data['review_note'], 'Confirmed violation')
        self.assertEqual(data['reviewed_by'], 7)
        self.assertEqual(data['feedback_note'], '測試')

        audit_log = self.fake_db.add.call_args.args[0]
        self.assertEqual(audit_log.violation_id, 77)
        self.assertEqual(audit_log.action, 'review_status_changed')
        self.assertEqual(audit_log.old_status, 'pending')
        self.assertEqual(audit_log.new_status, 'resolved')
        self.assertEqual(audit_log.flagged_reason, 'false_positive')
        self.fake_db.commit.assert_awaited_once()

    @patch('examples.violation_records.routers.load_user_with_effective_sites')
    async def test_review_violation_regular_user_forbidden(
        self,
        mock_load_user: AsyncMock,
    ) -> None:
        """Regular users cannot update review status."""
        siteA = self.make_site(1, 'SiteA')
        user = self.make_user('test_user', [siteA], role='user')
        mock_load_user.return_value = (user, [siteA])

        resp = self.client.patch(
            '/api/violations/77/review',
            json={'review_status': 'dismissed'},
        )

        self.assertEqual(resp.status_code, 403)
        self.fake_db.add.assert_not_called()

    @patch('examples.violation_records.routers.load_user_with_effective_sites')
    async def test_review_violation_forbidden_record(
        self,
        mock_load_user: AsyncMock,
    ) -> None:
        """Admins cannot review records outside their scope."""
        siteA = self.make_site(1, 'SiteA')
        reviewer = self.make_user('test_user', [siteA], role='admin')
        mock_load_user.return_value = (reviewer, [siteA])
        self.fake_db.execute.return_value = self._exec_scalar(None)

        resp = self.client.patch(
            '/api/violations/77/review',
            json={'review_status': 'resolved'},
        )

        self.assertEqual(resp.status_code, 403)
        self.fake_db.add.assert_not_called()

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
        def override_jwt_no_username() -> Any:
            """Support override_jwt_no_username."""
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
    async def test_upload_violation_success(
        self,
        mock_save_violation: AsyncMock,
    ) -> None:
        """A successful upload returns the violation ID from the manager."""
        # 1) User can access "SiteA"
        siteA = self.make_site(1, 'SiteA')
        user = self.make_user('test_user', [siteA])
        self.simulate_user_query(user)

        # 2) Simulate reading normal bytes from the image
        mock_file_obj = MagicMock()
        mock_file_obj.read = AsyncMock(return_value=b'some_image_bytes')

        # 3) The manager returns a new violation ID
        mock_save_violation.return_value = 123

        response = await upload_violation(
            site='SiteA',
            stream_name='Cam1',
            detection_time=None,
            warnings_json=None,
            detections_json=None,
            cone_polygon_json=None,
            pole_polygon_json=None,
            image=mock_file_obj,
            db=self.fake_db,
            credentials=JwtAuthorizationCredentials(
                subject={'username': 'test_user'},
            ),
        )

        self.assertEqual(response.violation_id, 123)
        mock_save_violation.assert_awaited_once()

    async def test_upload_violation_missing_username(self) -> None:
        """
        If the token lacks 'username', /api/upload should 401.
        """
        def override_jwt_no_username() -> Any:
            """Support override_jwt_no_username."""
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

    async def test_upload_violation_no_access_site(
        self,
    ) -> None:
        """A user cannot upload a violation for a site outside their scope."""
        siteA = self.make_site(1, 'SiteA')
        user = self.make_user('test_user', [siteA])
        self.simulate_user_query(user)

        mock_file_obj = MagicMock()
        with self.assertRaisesRegex(
                HTTPException,
                'No access to this site',
        ) as raised:
            await upload_violation(
                site='SiteB',
                stream_name='Cam1',
                detection_time=None,
                warnings_json=None,
                detections_json=None,
                cone_polygon_json=None,
                pole_polygon_json=None,
                image=mock_file_obj,
                db=self.fake_db,
                credentials=JwtAuthorizationCredentials(
                    subject={'username': 'test_user'},
                ),
            )
        self.assertEqual(raised.exception.status_code, 403)

    @patch(
        'examples.violation_records.routers.violation_manager.save_violation',
        new_callable=AsyncMock,
    )
    async def test_upload_violation_empty_image(
        self,
        mock_save_violation: AsyncMock,
    ) -> None:
        """An empty image is reported as HTTP 400."""
        siteA = self.make_site(1, 'SiteA')
        user = self.make_user('test_user', [siteA])
        self.simulate_user_query(user)

        mock_file_obj = MagicMock()
        mock_save_violation.side_effect = EmptyViolationImageError(
            'Empty image file',
        )

        with self.assertRaisesRegex(
                HTTPException,
                'Failed to read image file',
        ) as raised:
            await upload_violation(
                site='SiteA',
                stream_name='Cam1',
                detection_time=None,
                warnings_json=None,
                detections_json=None,
                cone_polygon_json=None,
                pole_polygon_json=None,
                image=mock_file_obj,
                db=self.fake_db,
                credentials=JwtAuthorizationCredentials(
                    subject={'username': 'test_user'},
                ),
            )
        self.assertEqual(raised.exception.status_code, 400)

    @patch(
        'examples.violation_records.routers.violation_manager.save_violation',
        new_callable=AsyncMock,
    )
    async def test_upload_violation_read_error(
        self,
        mock_save_violation: AsyncMock,
    ) -> None:
        """A manager image-read error is reported as HTTP 400."""
        siteA = self.make_site(1, 'SiteA')
        user = self.make_user('test_user', [siteA])
        self.simulate_user_query(user)

        mock_file_obj = MagicMock()
        mock_save_violation.side_effect = ViolationImageReadError(
            'Failed to read image file',
        )

        with self.assertRaisesRegex(
                HTTPException,
                'Failed to read image file',
        ) as raised:
            await upload_violation(
                site='SiteA',
                stream_name='Cam1',
                detection_time=None,
                warnings_json=None,
                detections_json=None,
                cone_polygon_json=None,
                pole_polygon_json=None,
                image=mock_file_obj,
                db=self.fake_db,
                credentials=JwtAuthorizationCredentials(
                    subject={'username': 'test_user'},
                ),
            )
        self.assertEqual(raised.exception.status_code, 400)

    @patch(
        'examples.violation_records.routers.violation_manager.save_violation',
        new_callable=AsyncMock,
    )
    async def test_upload_violation_save_fail(
        self,
        mock_save_violation: AsyncMock,
    ) -> None:
        """A missing violation ID from the manager is reported as HTTP 500."""
        siteA = self.make_site(1, 'SiteA')
        user = self.make_user('test_user', [siteA])
        self.simulate_user_query(user)

        mock_file_obj = MagicMock()
        mock_file_obj.read = AsyncMock(return_value=b'some_image_bytes')

        # Force the violation_manager to return None => 500
        mock_save_violation.return_value = None

        with self.assertRaisesRegex(
                HTTPException,
                'Failed to create violation record',
        ) as raised:
            await upload_violation(
                site='SiteA',
                stream_name='Cam1',
                detection_time=None,
                warnings_json=None,
                detections_json=None,
                cone_polygon_json=None,
                pole_polygon_json=None,
                image=mock_file_obj,
                db=self.fake_db,
                credentials=JwtAuthorizationCredentials(
                    subject={'username': 'test_user'},
                ),
            )
        self.assertEqual(raised.exception.status_code, 500)


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.violation_records.routers \
    --cov-report=term-missing tests/examples/violation_records/routers_test.py
'''
