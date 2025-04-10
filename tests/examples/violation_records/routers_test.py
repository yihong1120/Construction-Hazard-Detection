from __future__ import annotations

import unittest
from datetime import datetime
from typing import ClassVar
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi_jwt import JwtAuthorizationCredentials

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.violation_records.routers import router


#######################################################
# 1) Mock domain classes
#######################################################
class MockSite:
    """A mock Site model with basic fields."""

    def __init__(self, site_id: int, name: str) -> None:
        """
        Create a mock Site instance.

        Args:
            site_id (int): The ID of the site.
            name (str): The name of the site.
        """
        self.id: int = site_id
        self.name: str = name
        self.created_at: datetime = datetime(2023, 1, 1)
        self.updated_at: datetime = datetime(2023, 1, 2)


class MockUser:
    """A mock User model containing a username and a list of mock sites."""

    def __init__(self, username: str, sites: list[MockSite]) -> None:
        """
        Create a mock User instance.

        Args:
            username (str): The username.
            sites (list[MockSite]): A list of mock Site instances.
        """
        self.username: str = username
        self.sites: list[MockSite] = sites


class MockViolation:
    """
    A mock Violation object that includes fields referenced in the real code
    (e.g. cone_polygon_json, pole_polygon_json).
    """

    def __init__(
        self,
        violation_id: int,
        site: str,
        detection_time: datetime | None = None,
        stream_name: str = 'Cam1',
        image_path: str = 'some.jpg',
    ) -> None:
        """
        Create a mock Violation instance.

        Args:
            violation_id (int): The ID of the violation.
            site (str): The site name where the violation occurred.
            detection_time (datetime | None): The time of detection.
                Defaults to now.
            stream_name (str): The name of the stream or camera.
            image_path (str): The path to the violation image.
        """
        self.id: int = violation_id
        self.site: str = site
        self.stream_name: str = stream_name
        self.detection_time: datetime = detection_time or datetime.now()
        self.image_path: str = image_path
        self.created_at: datetime = datetime(2023, 1, 3)

        # Match fields in the real Violation model
        self.detections_json: str = 'some detection'
        self.warnings_json: str = 'some warning'
        self.cone_polygon_json: str = 'some cone polygons'
        self.pole_polygon_json: str = 'some pole polygons'


#######################################################
# 2) Fake DB result simulation classes
#######################################################
class FakeScalarsResult:
    """
    Simulate the result of (await db.scalars(stmt)).all(),
    returning a list of mock items.
    """

    def __init__(self, items: list) -> None:
        """Store the items to return when all() is called."""
        self._items: list = items

    def all(self) -> list:
        """Return the stored items."""
        return self._items


class FakeExecuteResult:
    """
    Simulate the result of (await db.execute(stmt)), including .scalar() and
    .scalars().all().
    """

    def __init__(self, scalar_result=None, scalars_result=None) -> None:
        """
        Args:
            scalar_result: The single item returned by .scalar().
            scalars_result: The list of items returned by .scalars().all().
        """
        self._scalar_result = scalar_result
        self._scalars_result = scalars_result

    def scalar(self):
        """Simulate returning a single scalar result."""
        return self._scalar_result

    def scalars(self):
        """Simulate returning an object whose .all() method gives a list."""
        class NonAsyncScalars:
            def __init__(self, data):
                self._data = data

            def all(self):
                return self._data

        return NonAsyncScalars(self._scalars_result or [])


class FakeAsyncDB:
    """
    Simulate an asynchronous database session that provides .execute() and
    .scalars().
    """

    def __init__(self) -> None:
        """Create a fake DB with mocked execute/scalars methods."""
        self.execute: AsyncMock = AsyncMock()
        self.scalars: AsyncMock = AsyncMock()


#######################################################
# 3) Test class for violation routers
#######################################################
class TestViolationRouters(unittest.IsolatedAsyncioTestCase):
    """
    A test suite for violation-related endpoints.
    """
    fake_db: ClassVar[FakeAsyncDB]
    client: ClassVar[TestClient]

    @classmethod
    def setUpClass(cls) -> None:
        """Set up once before running all tests in this suite."""
        super().setUpClass()

        # Create a FastAPI app, include the router
        app = FastAPI()
        app.include_router(router, prefix='/api')

        # Create a single Fake DB instance
        cls.fake_db = FakeAsyncDB()  # <== typed as ClassVar[FakeAsyncDB]

        async def override_get_db():
            yield cls.fake_db

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
        user_obj: MockUser | None,
    ) -> None:
        """
        Simulate the DB returning a user object.

        Args:
            user_obj (MockUser | None): The mock user object to return.
        """
        self.fake_db.execute.side_effect = None
        self.fake_db.execute.return_value = FakeExecuteResult(
            scalar_result=user_obj,
        )

    def append_site_query(
        self,
        site_obj: MockSite | None,
    ) -> None:
        """
        Append a site query result to the side_effect queue.

        Args:
            site_obj (MockSite | None): The mock site object to return.
        """
        cur = list(
            self.fake_db.execute.side_effect,
        ) if self.fake_db.execute.side_effect else []
        cur.append(FakeExecuteResult(scalar_result=site_obj))
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
        cur.append(FakeExecuteResult(scalar_result=count_val))
        self.fake_db.execute.side_effect = cur

    def simulate_scalars_list(self, items: list) -> None:
        """
        Simulate db.scalars(stmt).all() returning a list of items.

        Args:
            items (list): A list of mock items to return.
        """
        self.fake_db.scalars.return_value = FakeScalarsResult(items)

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
        user = MockUser('test_user', [])
        self.simulate_user_query(user)
        resp = self.client.get('/api/my_sites')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [])

    async def test_get_my_sites_success(self) -> None:
        """
        If the user has multiple sites, return their info as a list of dicts.
        """
        siteA = MockSite(1, 'SiteA')
        siteB = MockSite(2, 'SiteB')
        user = MockUser('test_user', [siteA, siteB])
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
        path_mock.is_relative_to.return_value = True
        path_mock.exists.return_value = True
        path_mock.suffix.lower.return_value = '.png'
        path_mock.name = 'image.png'
        mock_path.return_value = path_mock

        resp = self.client.get('/api/get_violation_image?image_path=image.png')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers['content-type'], 'image/png')

    ###################################################
    # /api/violations Tests
    ###################################################
    async def test_get_violations_user_not_found(self) -> None:
        """
        If no user is found in the DB, return 404.
        """
        self.simulate_user_query(None)
        resp = self.client.get('/api/violations')
        self.assertEqual(resp.status_code, 404)

    async def test_get_violations_no_sites(self) -> None:
        """
        If the user has no sites, return total=0 and items=[].
        """
        user = MockUser('test_user', [])
        self.simulate_user_query(user)
        resp = self.client.get('/api/violations')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {'total': 0, 'items': []})

    async def test_get_violations_site_id_403(self) -> None:
        """
        If the user tries to access a site ID that does not match their site,
        return 403.
        """
        siteA = MockSite(1, 'SiteA')
        user = MockUser('test_user', [siteA])
        self.fake_db.execute.side_effect = [
            FakeExecuteResult(scalar_result=user),
            FakeExecuteResult(scalar_result=MockSite(2, 'SiteB')),
        ]
        resp = self.client.get('/api/violations?site_id=2')
        self.assertEqual(resp.status_code, 403)

    async def test_get_violations_with_filters(self) -> None:
        """
        If keyword, start_time, end_time, limit, and offset are provided,
        verify the response returns the expected data.
        """
        siteA = MockSite(1, 'SiteA')
        user = MockUser('test_user', [siteA])
        self.fake_db.execute.side_effect = [
            FakeExecuteResult(scalar_result=user),
            FakeExecuteResult(scalar_result=2),  # total count
        ]
        v1 = MockViolation(123, 'SiteA')
        v2 = MockViolation(456, 'SiteA')
        self.fake_db.scalars.return_value = FakeScalarsResult([v1, v2])

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

    async def test_get_violations_success(self) -> None:
        """
        If the user and site are valid, and there's 1 violation, return it.
        """
        siteA = MockSite(1, 'SiteA')
        user = MockUser('test_user', [siteA])
        # user, siteObj, count=1
        self.fake_db.execute.side_effect = [
            FakeExecuteResult(scalar_result=user),
            FakeExecuteResult(scalar_result=siteA),
            FakeExecuteResult(scalar_result=1),
        ]
        viol = MockViolation(101, 'SiteA')
        self.fake_db.scalars.return_value = FakeScalarsResult([viol])

        resp = self.client.get('/api/violations?site_id=1')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['total'], 1)
        self.assertEqual(len(data['items']), 1)
        self.assertEqual(data['items'][0]['id'], 101)

    ###################################################
    # /api/violations/{violation_id} Tests
    ###################################################
    async def test_get_single_violation_user_not_found(self) -> None:
        """
        If there's no user, return 404.
        """
        self.simulate_user_query(None)
        resp = self.client.get('/api/violations/9999')
        self.assertEqual(resp.status_code, 404)

    async def test_get_single_violation_forbidden_violation_none(self) -> None:
        """
        If the DB returns None for the violation, respond with 403 because it's
        not accessible.
        """
        siteA = MockSite(1, 'SiteA')
        user = MockUser('test_user', [siteA])
        self.fake_db.execute.side_effect = [
            FakeExecuteResult(scalar_result=user),
            FakeExecuteResult(scalar_result=None),
        ]
        resp = self.client.get('/api/violations/1234')
        self.assertEqual(resp.status_code, 403)

    async def test_get_single_violation_forbidden_site_mismatch(self) -> None:
        """
        If the violation's site doesn't match the user's site,
        respond with 403.
        """
        siteA = MockSite(1, 'SiteA')
        user = MockUser('test_user', [siteA])
        viol = MockViolation(88, 'SiteB')
        self.fake_db.execute.side_effect = [
            FakeExecuteResult(scalar_result=user),
            FakeExecuteResult(scalar_result=viol),
        ]
        resp = self.client.get('/api/violations/88')
        self.assertEqual(resp.status_code, 403)

    async def test_get_single_violation_success(self) -> None:
        """
        If the violation matches the user's site,
        return 200 with violation data.
        """
        siteA = MockSite(1, 'SiteA')
        user = MockUser('test_user', [siteA])
        viol = MockViolation(77, 'SiteA')
        self.fake_db.execute.side_effect = [
            FakeExecuteResult(scalar_result=user),
            FakeExecuteResult(scalar_result=viol),
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
        siteA = MockSite(1, 'SiteA')
        user = MockUser('test_user', [siteA])
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

        """
        siteA = MockSite(1, 'SiteA')
        user = MockUser('test_user', [siteA])
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
        siteA = MockSite(1, 'SiteA')
        user = MockUser('test_user', [siteA])
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
        siteA = MockSite(1, 'SiteA')
        user = MockUser('test_user', [siteA])
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
        siteA = MockSite(1, 'SiteA')
        user = MockUser('test_user', [siteA])
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
