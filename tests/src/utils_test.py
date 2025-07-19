from __future__ import annotations

import asyncio
import base64
import os
import unittest
from datetime import datetime
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
from sklearn.cluster import HDBSCAN

from src.utils import FileEventHandler
from src.utils import RedisManager
from src.utils import TokenManager
from src.utils import Utils


class MockSharedToken:
    """
    A mock shared token object to simulate race conditions
    for TokenManager tests.

    Attributes:
        _data (dict[str, str | bool]):
            Internal dictionary storing token values.
        _call_count (int):
            Tracks the number of times 'get' is called for 'refresh_token'.
    """

    _data: dict[str, str | bool]
    _call_count: int

    def __init__(self) -> None:
        """
        Initialise the mock shared token with default values.
        """
        self._data = {
            'access_token': 'OLD',
            'refresh_token': 'ORIGINAL',
            'is_refreshing': False,
        }
        self._call_count = 0

    def get(self, key: str, default: str | bool | None = None) -> str | bool:
        """
        Retrieve a value from the mock token dictionary, simulating a change
        in 'refresh_token' after the first call.

        Args:
            key (str): The key to retrieve.
            default (str | bool | None, optional): Default value if key is not
                present.

        Returns:
            str | bool: The value associated with the key, or the default.
        """
        if key == 'refresh_token':
            self._call_count += 1
            if self._call_count == 1:
                # First call returns the original token for testing purposes
                return 'ORIGINAL'
            else:
                # Subsequent calls simulate a changed token to test race
                # conditions
                return 'CHANGED'
        value = self._data.get(key, default)
        if value is None:
            # Always return a str (empty string) if no value is found and no
            # default is provided
            value = ''
        assert isinstance(value, (str, bool)), 'Value must be str or bool.'
        return value

    def __getitem__(self, key: str) -> str | bool:
        """
        Enable bracket access to the internal dictionary.

        Args:
            key (str): The key to retrieve.

        Returns:
            str | bool: The value associated with the key.
        """
        value: str | bool = self._data[key]
        assert isinstance(value, (str, bool)), 'Value must be str or bool.'
        return value

    def __setitem__(self, key: str, value: str | bool) -> None:
        """
        Set a value in the internal dictionary.

        Args:
            key (str): The key to set.
            value (str | bool): The value to assign.
        """
        self._data[key] = value


class TestTokenManager(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the TokenManager class,
    covering authentication and token refresh logic.
    """

    shared_token: dict[str, str | bool]
    tm: TokenManager

    def setUp(self) -> None:
        """
        Set up a fresh TokenManager and shared_token for each test.
        """
        # Initialise shared_token dictionary for each test
        self.shared_token = {
            'access_token': '',
            'refresh_token': '',
            'is_refreshing': False,
        }
        # Create a new TokenManager instance
        self.tm = TokenManager(
            api_url='http://example.com/api',
            shared_token=self.shared_token,
        )

    @patch.dict(os.environ, {'API_USERNAME': 'dummy', 'API_PASSWORD': 'dummy'})
    @patch('aiohttp.ClientSession')
    async def test_refresh_token_changed_before_post(
        self, m_sess: AsyncMock,
    ) -> None:
        """
        Verify early return if refresh_token changes before HTTP POST
        (covers concurrency line 134).
        """
        post_mock = AsyncMock()
        m_sess.return_value.__aenter__.return_value.post = post_mock

        # Create a MockSharedToken and patch it using patch.object
        mock_shared_token = MockSharedToken()

        with patch.object(self.tm, 'shared_token', mock_shared_token):
            await self.tm.refresh_token()

        post_mock.assert_not_awaited()
        self.assertEqual(
            mock_shared_token['access_token'], 'OLD',
        )

    @patch.dict(
        os.environ,
        {'API_USERNAME': 'test_user', 'API_PASSWORD': 'test_pass'},
    )
    @patch('aiohttp.ClientSession')
    async def test_authenticate_success(self, m_session: AsyncMock) -> None:
        """
        Verify successful authentication sets correct tokens.
        """
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = {
            'access_token': 'A', 'refresh_token': 'B',
        }

        mock_sessinst = AsyncMock()
        mock_sessinst.post.return_value = mock_resp
        m_session.return_value.__aenter__.return_value = mock_sessinst

        await self.tm.authenticate(force=True)
        self.assertEqual(self.shared_token['access_token'], 'A')
        self.assertEqual(
            self.shared_token['refresh_token'], 'B',
        )

    @patch.dict(
        os.environ,
        {'API_USERNAME': 'test_user', 'API_PASSWORD': 'test_pass'},
    )
    @patch('aiohttp.ClientSession')
    async def test_authenticate_fail(self, m_session: AsyncMock) -> None:
        """
        Test authentication failure results in RuntimeError.
        """
        mock_resp = AsyncMock()
        mock_resp.status = 401
        mock_sessinst = AsyncMock()
        mock_sessinst.post.return_value = mock_resp
        m_session.return_value.__aenter__.return_value = mock_sessinst

        with self.assertRaises(RuntimeError):
            await self.tm.authenticate(force=True)

    @patch('aiohttp.ClientSession')
    async def test_authenticate_missing_env(
        self, m_session: AsyncMock,
    ) -> None:
        """
        Test that missing environment variables raises ValueError.
        """
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await self.tm.authenticate(force=True)
        m_session.assert_not_called()

    @patch.dict(os.environ, {'API_USERNAME': 'dummy', 'API_PASSWORD': 'dummy'})
    async def test_authenticate_no_force_noop(self) -> None:
        """
        Test authenticate is skipped when force is False and access_token
        exists.
        """
        self.shared_token['access_token'] = 'EXIST'
        with patch('aiohttp.ClientSession') as mock_sess:
            await self.tm.authenticate(force=False)
        mock_sess.assert_not_called()

    @patch.dict(os.environ, {'API_USERNAME': 'dummy', 'API_PASSWORD': 'dummy'})
    async def test_refresh_token_no_refresh_token(self) -> None:
        """
        Test that no refresh_token triggers authenticate.
        """
        self.shared_token['refresh_token'] = ''
        with patch.object(
            self.tm,
            'authenticate',
                new_callable=AsyncMock,
        ) as m_auth:
            await self.tm.refresh_token()
            m_auth.assert_awaited_once()

    @patch.dict(os.environ, {'API_USERNAME': 'dummy', 'API_PASSWORD': 'dummy'})
    @patch('aiohttp.ClientSession')
    async def test_refresh_token_already_refreshing(
        self, m_sess: AsyncMock,
    ) -> None:
        """
        Test that refresh_token exits early if already refreshing.
        """
        self.shared_token['is_refreshing'] = True
        await self.tm.refresh_token()
        m_sess.assert_not_called()

    @patch.dict(os.environ, {'API_USERNAME': 'dummy', 'API_PASSWORD': 'dummy'})
    @patch('aiohttp.ClientSession')
    async def test_refresh_token_changed_refresh_token_midway(
        self, m_sess: AsyncMock,
    ) -> None:
        """
        Test refresh_token behaviour when refresh_token changes during POST.
        """
        self.shared_token['refresh_token'] = 'X'
        self.shared_token['access_token'] = 'OLD'

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = {
            'access_token': 'NEW', 'refresh_token': 'NEWREF',
        }

        async def side_effect(*_, **__):
            await asyncio.sleep(0.01)
            self.shared_token['refresh_token'] = 'Y'
            return mock_resp

        mock_sessinst = AsyncMock()
        mock_sessinst.post.side_effect = side_effect
        m_sess.return_value.__aenter__.return_value = mock_sessinst

        await self.tm.refresh_token()
        self.assertEqual(
            self.shared_token['access_token'], 'NEW',
        )

    @patch.dict(os.environ, {'API_USERNAME': 'dummy', 'API_PASSWORD': 'dummy'})
    @patch('aiohttp.ClientSession')
    async def test_refresh_token_200_ok(self, m_sess: AsyncMock) -> None:
        """
        Verify token update on HTTP 200 OK response.
        """
        self.shared_token['access_token'] = 'OLD'
        self.shared_token['refresh_token'] = 'RRR'

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = {
            'access_token': 'NNN', 'refresh_token': 'RRR2',
        }

        mock_sessinst = AsyncMock()
        mock_sessinst.post.return_value = mock_resp
        m_sess.return_value.__aenter__.return_value = mock_sessinst

        await self.tm.refresh_token()
        self.assertEqual(self.shared_token['access_token'], 'NNN')
        self.assertEqual(
            self.shared_token['refresh_token'], 'RRR2',
        )
        self.assertFalse(self.shared_token['is_refreshing'])

    @patch.dict(os.environ, {'API_USERNAME': 'dummy', 'API_PASSWORD': 'dummy'})
    @patch('aiohttp.ClientSession')
    async def test_refresh_token_fail_401(self, m_sess: AsyncMock) -> None:
        """
        Trigger fallback to authenticate on 401 response.
        """
        self.shared_token['access_token'] = 'OLD'
        self.shared_token['refresh_token'] = 'RRR'

        mock_resp = AsyncMock()
        mock_resp.status = 401

        mock_sessinst = AsyncMock()
        mock_sessinst.post.return_value = mock_resp
        m_sess.return_value.__aenter__.return_value = mock_sessinst

        with patch.object(
            self.tm,
            'authenticate',
            new_callable=AsyncMock,
        ) as m_auth:
            await self.tm.refresh_token()
            m_auth.assert_awaited_once()

    @patch.dict(os.environ, {'API_USERNAME': 'dummy', 'API_PASSWORD': 'dummy'})
    @patch('aiohttp.ClientSession')
    async def test_refresh_token_fail_other(self, m_sess: AsyncMock) -> None:
        """
        Raise RuntimeError on non-401 HTTP errors.
        """
        self.shared_token['access_token'] = 'X'
        self.shared_token['refresh_token'] = 'Y'

        mock_resp = AsyncMock()
        mock_resp.status = 500

        mock_sessinst = AsyncMock()
        mock_sessinst.post.return_value = mock_resp
        m_sess.return_value.__aenter__.return_value = mock_sessinst

        with self.assertRaises(RuntimeError):
            await self.tm.refresh_token()

    @patch.dict(os.environ, {'API_USERNAME': 'dummy', 'API_PASSWORD': 'dummy'})
    async def test_ensure_token_valid_no_token(self) -> None:
        """
        Trigger authenticate if access_token is missing.
        """
        self.shared_token['access_token'] = ''
        with patch.object(
            self.tm,
            'authenticate',
            new_callable=AsyncMock,
        ) as m_auth:
            await self.tm.ensure_token_valid()
            m_auth.assert_awaited_once()

    @patch.dict(os.environ, {'API_USERNAME': 'dummy', 'API_PASSWORD': 'dummy'})
    async def test_ensure_token_valid_has_token(self) -> None:
        """
        Skip authentication if access_token exists.
        """
        self.shared_token['access_token'] = 'EXIST'
        with patch.object(
            self.tm,
            'authenticate',
            new_callable=AsyncMock,
        ) as m_auth:
            await self.tm.ensure_token_valid()
            m_auth.assert_not_awaited()

    async def test_handle_401_over_retries(self) -> None:
        """
        Raise RuntimeError if retry limit exceeded on 401 handling.
        """
        with self.assertRaises(RuntimeError):
            await self.tm.handle_401(retry_count=5)

    @patch.object(TokenManager, 'refresh_token', new_callable=AsyncMock)
    async def test_handle_401_refresh_error(self, m_ref: AsyncMock) -> None:
        """
        Fallback to authenticate if refresh_token fails during 401 handling.
        """
        m_ref.side_effect = Exception('some error')
        with patch.object(
            self.tm,
            'authenticate',
            new_callable=AsyncMock,
        ) as m_auth:
            await self.tm.handle_401(retry_count=0)
            m_auth.assert_awaited_once()

    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_refresh_token_wait_timeout(
        self, mock_sleep: AsyncMock,
    ) -> None:
        """
        Ensure refresh_token exits after timeout if is_refreshing remains True.
        """
        counter = 0.0

        async def fake_sleep(duration: float) -> None:
            nonlocal counter
            counter += duration

        mock_sleep.side_effect = fake_sleep
        self.shared_token['is_refreshing'] = True

        await self.tm.refresh_token()
        self.assertGreaterEqual(counter, 10)

    async def test_ensure_token_valid_over_retries(self) -> None:
        """
        Raise RuntimeError if ensure_token_valid exceeds retry limit.
        """
        with self.assertRaises(RuntimeError) as ctx:
            await self.tm.ensure_token_valid(retry_count=10)
        self.assertIn(
            'Exceeded max_retries in ensure_token_valid',
            str(ctx.exception),
        )

    @patch.dict(os.environ, {'API_USERNAME': 'dummy', 'API_PASSWORD': 'dummy'})
    @patch('aiohttp.ClientSession')
    async def test_refresh_token_changed_refresh_token_immediately(
        self, m_sess: AsyncMock,
    ) -> None:
        """
        Handle token change just before post is executed.
        """
        self.shared_token['refresh_token'] = 'X'
        self.shared_token['access_token'] = 'OLD'

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = {
            'access_token': 'NEW', 'refresh_token': 'NEWREF',
        }

        mock_sessinst = AsyncMock()
        mock_sessinst.post.return_value = mock_resp
        m_sess.return_value.__aenter__.return_value = mock_sessinst

        async def change_token_side_effect(*_, **__):
            self.shared_token['refresh_token'] = 'CHANGED'
            return mock_resp
        mock_sessinst.post = AsyncMock(side_effect=change_token_side_effect)

        await self.tm.refresh_token()
        self.assertEqual(
            self.shared_token['access_token'], 'NEW',
        )
        self.assertEqual(
            self.shared_token['refresh_token'], 'NEWREF',
        )

    @patch.dict(os.environ, {'API_USERNAME': 'dummy', 'API_PASSWORD': 'dummy'})
    async def test_refresh_token_exit_while(self) -> None:
        """
        Test loop exits once is_refreshing becomes False within timeout.
        """
        self.shared_token['is_refreshing'] = True

        async def stop_refresh() -> None:
            await asyncio.sleep(0.5)
            self.shared_token['is_refreshing'] = False

        asyncio.create_task(stop_refresh())
        await self.tm.refresh_token()
        self.assertFalse(self.shared_token['is_refreshing'])


class TestUtils(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the Utils class, covering all utility functions and edge
    cases.

    This test class provides comprehensive coverage for the Utils static
    methods,
    including error handling, boundary conditions, and various data scenarios.
    """

    def test_file_event_handler_init_and_on_modified(self) -> None:
        """
        Test FileEventHandler initialisation and on_modified method branches
        for coverage.

        # Verifies that the FileEventHandler correctly processes file
        # modification events and handles both matching and non-matching
        # file paths appropriately.
        """
        called: list[str] = []

        def callback() -> None:
            """Mock callback function to track invocations."""
            called.append('called')

        loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        handler: FileEventHandler = FileEventHandler(
            '/tmp/testfile', callback, loop,
        )

        # Simulate event with matching path
        event: Any = MagicMock()
        event.src_path = '/tmp/testfile'

        # Patch asyncio.run_coroutine_threadsafe to avoid actual scheduling
        with patch('asyncio.run_coroutine_threadsafe') as mock_run:
            handler.on_modified(event)
            mock_run.assert_called()

        # Simulate event with non-matching path
        event.src_path = '/tmp/otherfile'
        with patch('asyncio.run_coroutine_threadsafe') as mock_run:
            handler.on_modified(event)
            mock_run.assert_not_called()

    def test_normalise_bbox_extra_fields(self) -> None:
        """
        Test normalise_bbox with more than 4 fields (should preserve extra
        fields).

        # Verifies that bounding boxes with additional fields beyond the
        # standard [left, top, right, bottom] format retain their extra data
        # correctly.
        """
        bbox: list[float] = [1, 2, 3, 4, 5, 6]
        result: list[float] = Utils.normalise_bbox(bbox)
        self.assertEqual(result, [1, 2, 3, 4, 5, 6])

    def test_detect_polygon_from_cones_no_cones(self) -> None:
        """
        # Test detect_polygon_from_cones with no cones present (should return
        # empty list).

        Ensures the method handles detection data that contains no safety
        # cones (class_id != 6) gracefully by returning an empty polygon
        # list.
        """
        datas: list[list[float]] = [
            [10, 10, 20, 20, 0.9, 1],  # Non-cone object
            [30, 30, 40, 40, 0.9, 2],  # Non-cone object
        ]
        clusterer: MagicMock = MagicMock()
        result: list[Polygon] = Utils.detect_polygon_from_cones(
            datas, clusterer,
        )
        self.assertEqual(result, [])

    def test_calculate_people_in_controlled_area_no_polygons(self) -> None:
        """
        # Test calculate_people_in_controlled_area with no polygons (should
        # return 0).

        Verifies that when no controlled area polygons are defined, the
        # method correctly returns zero people count regardless of detection
        # data.
        """
        polygons: list[Polygon] = []
        datas: list[list[float]] = [
            [1, 1, 3, 3, 0.9, 5],  # Person detection
            [4, 4, 8, 8, 0.9, 5],  # Person detection
        ]
        count: int = Utils.calculate_people_in_controlled_area(polygons, datas)
        self.assertEqual(count, 0)

    def test_is_expired_with_valid_date(self) -> None:
        """
        Test is_expired method with valid ISO 8601 date strings.

        # Verifies that the method correctly identifies expired and
        # non-expired dates when given properly formatted ISO 8601 date
        # strings.
        """
        # Test with a past date (should return True)
        past_date: str = (datetime.now() - timedelta(days=1)).isoformat()
        self.assertTrue(Utils.is_expired(past_date))

        # Test with a future date (should return False)
        future_date: str = (datetime.now() + timedelta(days=1)).isoformat()
        self.assertFalse(Utils.is_expired(future_date))

    def test_is_expired_with_invalid_date(self) -> None:
        """
        Test is_expired method with invalid ISO 8601 date string.

        # Ensures that malformed date strings are handled gracefully by
        # returning False rather than raising an exception.
        """
        # Test with an invalid ISO 8601 date (should return False)
        invalid_date: str = '2024-13-01T00:00:00'
        self.assertFalse(Utils.is_expired(invalid_date))

    def test_is_expired_with_none(self) -> None:
        """
        Test is_expired method with None input.

        Verifies that None input is handled appropriately by returning False.
        """
        # Test with None (should return False)
        self.assertFalse(Utils.is_expired(None))

    def test_encode(self) -> None:
        """
        Test the encode method for URL-safe Base64 encoding.

        # Validates that the method correctly encodes strings using URL-safe
        # Base64 encoding and returns the expected encoded output.
        """
        # Test encoding a string
        value: str = 'test_value'
        encoded_value: str = Utils.encode(value)
        expected_value: str = base64.urlsafe_b64encode(
            value.encode('utf-8'),
        ).decode('utf-8')
        self.assertEqual(encoded_value, expected_value)

    def test_encode_frame_success(self) -> None:
        """
        Test encoding a valid frame using encode_frame method.

        # Verifies that a valid NumPy array representing an image frame can be
        # successfully encoded to PNG format as bytes.
        """
        frame: np.ndarray = np.zeros((100, 100, 3), dtype=np.uint8)
        encoded_frame: bytes = Utils.encode_frame(frame)
        self.assertIsNotNone(encoded_frame)

    def test_encode_frame_failure(self) -> None:
        """
        Test encoding an invalid frame using encode_frame method.

        # Ensures that invalid input (e.g., None) is handled gracefully by
        # returning empty bytes rather than raising an exception.
        """
        invalid_frame: None = None
        encoded_frame: bytes = Utils.encode_frame(invalid_frame)
        self.assertEqual(
            encoded_frame, b'',
            'Expected empty bytes when encoding fails.',
        )

    def test_filter_warnings_by_working_hour_working_hours(self) -> None:
        """
        Test filtering warnings during working hours.

        # Verifies that during working hours, all warnings are returned
        # without any filtering applied to the warnings dictionary.
        """
        # Arrange - prepare warnings data in expected format
        warnings: dict[str, dict[str, int]] = {
            'warning_people_in_controlled_area': {'count': 2},
            'warning_no_safety_vest': {'count': 1},
        }
        is_working_hour: bool = True  # During working hours

        # Act - filter warnings based on working hour status
        message: dict[str, dict[str, int]] = (
            Utils.filter_warnings_by_working_hour(warnings, is_working_hour)
        )

        # Assert - all warnings should be present during working hours
        self.assertIn('warning_people_in_controlled_area', message)
        self.assertIn('warning_no_safety_vest', message)

        # Verify content integrity
        self.assertEqual(
            message['warning_people_in_controlled_area']['count'], 2,
        )
        self.assertEqual(message['warning_no_safety_vest']['count'], 1)

    def test_filter_warnings_by_working_hour_non_working_hours(self) -> None:
        """
        Test filtering warnings outside working hours.

        # Ensures that outside working hours, only controlled area warnings
        # are retained whilst other warning types are filtered out.
        """
        # Arrange - prepare warnings with multiple types
        warnings: dict[str, dict[str, int]] = {
            'warning_people_in_controlled_area': {'count': 3},
            'warning_no_safety_vest': {'count': 2},
        }
        is_working_hour: bool = False  # Outside working hours

        # Act - filter warnings for non-working hours
        message: dict[str, dict[str, int]] = (
            Utils.filter_warnings_by_working_hour(warnings, is_working_hour)
        )

        # Assert - only controlled area warnings should remain
        self.assertIn('warning_people_in_controlled_area', message)
        self.assertNotIn('warning_no_safety_vest', message)
        self.assertEqual(
            message['warning_people_in_controlled_area']['count'], 3,
        )

    def test_filter_warnings_by_working_hour_no_message(self) -> None:
        """
        Test filtering warnings when no warnings are present.

        # Verifies that an empty warnings dictionary returns an empty result
        # regardless of working hour status.
        """
        warnings: dict[str, dict[str, int]] = {}
        is_working_hour: bool = True

        message: dict[str, dict[str, int]] = (
            Utils.filter_warnings_by_working_hour(warnings, is_working_hour)
        )
        # When warnings are empty, function should return empty dictionary
        self.assertEqual(
            message, {}, 'Expected empty dict when no warnings.',
        )

    def test_should_notify_true(self) -> None:
        """
        Test should_notify returning True when cooldown period has passed.

        # Verifies that notifications are allowed when sufficient time has
        # elapsed since the last notification based on the cooldown period.
        """
        timestamp: int = int(datetime.now().timestamp())
        last_notification_time: int = timestamp - 400  # 400 seconds ago
        cooldown_period: int = 300  # 300 second cooldown

        self.assertTrue(
            Utils.should_notify(
                timestamp, last_notification_time, cooldown_period,
            ),
        )

    def test_should_notify_false(self) -> None:
        """
        Test should_notify returning False when cooldown period has not passed.

        # Ensures that notifications are blocked when insufficient time has
        # elapsed since the last notification based on the cooldown period.
        """
        timestamp: int = int(datetime.now().timestamp())
        last_notification_time: int = timestamp - 200  # 200 seconds ago
        cooldown_period: int = 300  # 300 second cooldown

        self.assertFalse(
            Utils.should_notify(
                timestamp, last_notification_time, cooldown_period,
            ),
        )

    def test_is_driver(self) -> None:
        """
        Test case for checking if a person is driving based on bounding boxes.
        """
        person_bbox: list[float] = [100, 200, 150, 250]
        vehicle_bbox: list[float] = [50, 100, 200, 300]
        self.assertTrue(Utils.is_driver(person_bbox, vehicle_bbox))

        person_bbox = [100, 200, 200, 400]
        self.assertFalse(Utils.is_driver(person_bbox, vehicle_bbox))

    def test_driver_detection_coverage(self) -> None:
        """
        Test case to ensure driver detection code coverage.
        """
        # Case where person is likely the driver
        person_bbox = Utils.normalise_bbox([150, 250, 170, 350])
        vehicle_bbox = Utils.normalise_bbox([100, 200, 300, 400])
        self.assertTrue(Utils.is_driver(person_bbox, vehicle_bbox))

        # Case where person is not the driver
        # due to horizontal position (left outside bounds)
        person_bbox = Utils.normalise_bbox([50, 250, 90, 300])
        self.assertFalse(Utils.is_driver(person_bbox, vehicle_bbox))

        # Case where person is not the driver
        # due to horizontal position (right outside bounds)
        person_bbox = Utils.normalise_bbox([310, 250, 350, 300])
        self.assertFalse(Utils.is_driver(person_bbox, vehicle_bbox))

        # Case where person is not the driver
        # due to vertical position (above vehicle)
        person_bbox = Utils.normalise_bbox([100, 50, 150, 100])
        self.assertFalse(Utils.is_driver(person_bbox, vehicle_bbox))

        # Case where person is the driver
        # due to person's top being below vehicle's top
        person_bbox = Utils.normalise_bbox([150, 210, 180, 300])
        self.assertTrue(Utils.is_driver(person_bbox, vehicle_bbox))

        # Case where person is not the driver
        # due to person's height being more than half vehicle's height
        person_bbox = Utils.normalise_bbox([150, 300, 180, 450])
        self.assertFalse(Utils.is_driver(person_bbox, vehicle_bbox))

        person_bbox = Utils.normalise_bbox([80, 250, 110, 300])
        print(
            f"Testing with person_bbox: "
            f"{person_bbox} and vehicle_bbox: {vehicle_bbox}",
        )
        self.assertFalse(Utils.is_driver(person_bbox, vehicle_bbox))

    def test_horizontal_position_check(self) -> None:
        """
        Test case to ensure coverage for horizontal position check.
        """
        # Case where person is within the acceptable horizontal bounds
        person_bbox = Utils.normalise_bbox([200, 200, 240, 240])
        vehicle_bbox = Utils.normalise_bbox([190, 150, 250, 300])
        self.assertTrue(Utils.is_driver(person_bbox, vehicle_bbox))

        # Case where person is to the left of the acceptable horizontal bounds
        person_bbox = Utils.normalise_bbox([50, 200, 90, 240])
        vehicle_bbox = Utils.normalise_bbox([100, 150, 200, 300])
        self.assertFalse(Utils.is_driver(person_bbox, vehicle_bbox))

        # Case where person is to the right of the acceptable horizontal bounds
        person_bbox = Utils.normalise_bbox([210, 200, 250, 240])
        vehicle_bbox = Utils.normalise_bbox([100, 150, 200, 300])
        self.assertFalse(Utils.is_driver(person_bbox, vehicle_bbox))

        # Case where person is exactly on the left boundary
        person_bbox = Utils.normalise_bbox([150, 200, 190, 240])
        vehicle_bbox = Utils.normalise_bbox([190, 150, 250, 300])
        self.assertFalse(Utils.is_driver(person_bbox, vehicle_bbox))

        # Case where person is exactly on the right boundary
        person_bbox = Utils.normalise_bbox([250, 200, 290, 240])
        vehicle_bbox = Utils.normalise_bbox([190, 150, 250, 300])
        self.assertFalse(Utils.is_driver(person_bbox, vehicle_bbox))

        # Case where person's height is exactly half the vehicle's height
        person_bbox = Utils.normalise_bbox([100, 200, 150, 300])
        vehicle_bbox = Utils.normalise_bbox([50, 100, 200, 400])
        self.assertTrue(Utils.is_driver(person_bbox, vehicle_bbox))

        # Case where person's height is more than half the vehicle's height
        person_bbox = Utils.normalise_bbox([100, 200, 150, 350])
        self.assertFalse(Utils.is_driver(person_bbox, vehicle_bbox))

    def test_height_check(self) -> None:
        """
        Test case to ensure coverage for height check.
        """
        # Case where person's height is more than half the vehicle's height
        person_bbox = Utils.normalise_bbox([150, 250, 180, 400])
        vehicle_bbox = Utils.normalise_bbox([100, 200, 300, 300])
        self.assertFalse(Utils.is_driver(person_bbox, vehicle_bbox))

        # Case where the driver's height condition is evaluated
        vehicle_bbox = Utils.normalise_bbox([100, 100, 300, 400])
        person_bbox = Utils.normalise_bbox([160, 150, 190, 310])
        self.assertFalse(Utils.is_driver(person_bbox, vehicle_bbox))

    def test_overlap_percentage(self) -> None:
        """
        Test calculating overlap percentage between two bounding boxes.
        """
        bbox1: list[float] = [100, 100, 200, 200]
        bbox2: list[float] = [150, 150, 250, 250]
        self.assertAlmostEqual(
            Utils.overlap_percentage(bbox1, bbox2), 0.142857, places=6,
        )

        bbox1 = [100, 100, 200, 200]
        bbox2 = [300, 300, 400, 400]
        self.assertEqual(
            Utils.overlap_percentage(
                Utils.normalise_bbox(
                    bbox1,
                ), Utils.normalise_bbox(bbox2),
            ), 0.0,
        )

    def test_is_dangerously_close(self) -> None:
        """
        Test case for checking if a person is dangerously close to a vehicle.
        """
        person_bbox: list[float] = [100, 100, 120, 120]
        vehicle_bbox: list[float] = [100, 100, 200, 200]
        self.assertTrue(
            Utils.is_dangerously_close(
                Utils.normalise_bbox(
                    person_bbox,
                ), Utils.normalise_bbox(vehicle_bbox), 'Vehicle',
            ),
        )

        person_bbox = [0, 0, 10, 10]
        vehicle_bbox = [100, 100, 200, 200]
        self.assertFalse(
            Utils.is_dangerously_close(
                Utils.normalise_bbox(
                    person_bbox,
                ), Utils.normalise_bbox(vehicle_bbox), 'Vehicle',
            ),
        )

    def test_calculate_people_in_controlled_area(self) -> None:
        """
        Test case for calculating the number of people in the controlled area.
        """
        datas: list[list[float]] = [
            [50, 50, 150, 150, 0.95, 0],    # Hardhat
            [200, 200, 300, 300, 0.85, 5],  # Person
            [400, 400, 500, 500, 0.75, 9],  # Vehicle
        ]
        normalised_datas = Utils.normalise_data(datas)
        clusterer = HDBSCAN(min_samples=3, min_cluster_size=2)
        polygons = Utils.detect_polygon_from_cones(normalised_datas, clusterer)
        people_count = Utils.calculate_people_in_controlled_area(
            polygons, normalised_datas,
        )
        self.assertEqual(people_count, 0)

        datas = [
            [100, 100, 120, 120, 0.9, 6],  # Safety cone
            [150, 150, 170, 170, 0.85, 6],  # Safety cone
            [130, 130, 140, 140, 0.95, 5],  # Person inside the area
            [300, 300, 320, 320, 0.85, 5],  # Person outside the area
            [200, 200, 220, 220, 0.89, 6],  # Safety cone
            [250, 250, 270, 270, 0.85, 6],  # Safety cone
            [450, 450, 470, 470, 0.92, 6],  # Safety cone
            [500, 500, 520, 520, 0.88, 6],  # Safety cone
            [550, 550, 570, 570, 0.86, 6],  # Safety cone
            [600, 600, 620, 620, 0.84, 6],  # Safety cone
            [650, 650, 670, 670, 0.82, 6],  # Safety cone
            [700, 700, 720, 720, 0.80, 6],  # Safety cone
            [750, 750, 770, 770, 0.78, 6],  # Safety cone
            [800, 800, 820, 820, 0.76, 6],  # Safety cone
            [850, 850, 870, 870, 0.74, 6],  # Safety cone
        ]

        normalised_datas = Utils.normalise_data(datas)
        polygons = Utils.detect_polygon_from_cones(normalised_datas, clusterer)
        people_count = Utils.calculate_people_in_controlled_area(
            polygons, normalised_datas,
        )
        self.assertEqual(people_count, 1)

    def test_no_cones(self) -> None:
        """
        Test case for checking behavior when no cones are detected.
        """
        data: list[list[float]] = [
            [50, 50, 150, 150, 0.95, 0],  # Hardhat
            [200, 200, 300, 300, 0.85, 5],  # Person
            [400, 400, 500, 500, 0.75, 2],  # No-Safety Vest
        ]
        normalised_data = Utils.normalise_data(data)
        clusterer = HDBSCAN(min_samples=3, min_cluster_size=2)
        polygons = Utils.detect_polygon_from_cones(normalised_data, clusterer)
        self.assertEqual(len(polygons), 0)

    def test_person_inside_polygon(self) -> None:
        """
        Test case for checking behavior when a person is inside a polygon.
        """
        polygons = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        ]
        data: list[list[float]] = [
            [2, 2, 8, 8, 0.95, 5],  # Person inside the polygon
        ]
        normalised_data = Utils.normalise_data(data)
        people_count = Utils.calculate_people_in_controlled_area(
            polygons, normalised_data,
        )
        self.assertEqual(people_count, 1)

    def test_build_utility_pole_union_no_pole(self) -> None:
        """
        Test building utility pole union with no pole data.

        Verifies that when no utility pole data is provided, the function
        returns an empty Polygon object, ensuring proper handling of
        empty datasets in the utility pole union calculation.
        """
        datas: list[list[float]] = []  # No utility pole data
        cluster = HDBSCAN(min_samples=3, min_cluster_size=2)
        poly = Utils.build_utility_pole_union(datas, cluster)
        self.assertTrue(isinstance(poly, Polygon))
        self.assertTrue(poly.is_empty)

    def test_build_utility_pole_union_single_pole(self) -> None:
        """
        Test building utility pole union with a single pole.

        Verifies that when only one utility pole is provided, the function
        correctly creates a non-empty Polygon representing the controlled
        area around the single pole.
        """
        datas: list[list[float]] = [[10, 2, 20, 30, 0.9, 9]]
        cluster = HDBSCAN(min_samples=3, min_cluster_size=2)
        poly = Utils.build_utility_pole_union(datas, cluster)
        self.assertTrue(isinstance(poly, Polygon))
        self.assertFalse(poly.is_empty)

    def test_build_utility_pole_union_multiple_poles(self) -> None:
        """
        Test building utility pole union with multiple poles.

        Verifies that when multiple utility poles are provided, the function
        correctly clusters them and creates a union polygon with positive
        area covering all pole positions.
        """
        datas: list[list[float]] = [
            [10, 2, 20, 30, 0.9, 9],
            [25, 5, 35, 35, 0.9, 9],
            [40, 1, 50, 30, 0.9, 9],
        ]
        # Lower parameters for clustering
        cluster = HDBSCAN(min_samples=2, min_cluster_size=2)
        poly = Utils.build_utility_pole_union(datas, cluster)
        self.assertTrue(isinstance(poly, Polygon))
        self.assertGreater(poly.area, 0)

    def test_is_dangerously_close_large_person_area(self) -> None:
        """
        Test case to ensure coverage for large person area ratio condition.

        # Verifies that when the person area to vehicle area ratio exceeds
        # the acceptable threshold, the function correctly returns False,
        # indicating no dangerous proximity.

        Returns:
            None
        """
        # person_area = 100 * 100 = 10,000
        person_bbox: list[float] = [0, 0, 100, 100]
        # vehicle_area = 50 * 50 = 2,500
        vehicle_bbox: list[float] = [0, 0, 50, 50]
        label: str = 'vehicle'  # acceptable_ratio=0.1

        # person_area / vehicle_area = 10000 / 2500 = 4 > 0.1
        # Triggers if person_area / vehicle_area > acceptable_ratio =>
        # return False
        self.assertFalse(
            Utils.is_dangerously_close(
                person_bbox, vehicle_bbox, label,
            ),
        )

    def test_detect_polygon_from_cones_empty_list(self) -> None:
        """
        Test polygon detection with empty cone data list.

        # Verifies that when an empty list is passed to
        # detect_polygon_from_cones, the function returns an empty list,
        # properly covering the early return branch for empty input data.

        Returns:
            None
        """
        clusterer = HDBSCAN(min_samples=3, min_cluster_size=2)
        result = Utils.detect_polygon_from_cones([], clusterer)
        self.assertEqual(
            result, [], 'Expected an empty list when datas is empty.',
        )

    def test_detect_polygon_from_cones_with_noise(self) -> None:
        """
        Test polygon detection when all cones are classified as noise.

        # Verifies that when all safety cones are classified as noise
        # (label -1) by the clustering algorithm, the function returns an
        # empty list, properly handling the case where no valid clusters are
        # formed.

        Returns:
            None
        """

        # Arrange: Prepare test data
        datas: list[list[float]] = [
            [10, 10, 20, 20, 0.9, 6],
            [30, 30, 40, 40, 0.9, 6],
            [50, 50, 60, 60, 0.9, 6],
        ]
        # Use MagicMock to mock clustering behaviour
        dummy_clusterer = MagicMock()
        dummy_clusterer.fit_predict.return_value = [
            -1 for _ in range(len(datas))
        ]
        # Call function where all safety cones are marked as noise
        # (label = -1)
        polygons = Utils.detect_polygon_from_cones(datas, dummy_clusterer)
        # Expected empty list as all points are skipped
        self.assertEqual(
            polygons, [], 'Expected no polygons when all points are noise.',
        )

    def test_calculate_people_in_controlled_area_no_datas(self) -> None:
        """
        Test people count calculation with empty detection data.

        # Verifies that calculate_people_in_controlled_area returns 0 when no
        # detections are provided (empty datas list), ensuring proper
        # handling of scenarios with no detection data.

        Returns:
            None
        """
        polygons: list[Polygon] = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        ]
        datas: list[list[float]] = []  # Empty list

        people_count = Utils.calculate_people_in_controlled_area(
            polygons, datas,
        )
        self.assertEqual(
            people_count, 0, 'Expected 0 when datas is empty.',
        )

    def test_build_utility_pole_union_less_than_min_samples(self) -> None:
        """
        Test utility pole union when pole count is below minimum samples.

        # Verifies that when the number of utility poles is less than the
        # clusterer's minimum samples requirement, the function directly
        # unions all poles without calling the clustering algorithm.

        Returns:
            None
        """
        # Arrange: Only provide two poles, whilst clusterer.min_samples=3
        # => len(utility_poles) (2) < clusterer.min_samples(3)
        datas: list[list[float]] = [
            # x1, y1, x2, y2, score, class_id=9(utility pole)
            [10, 5, 20, 30, 0.9, 9],
            [25, 8, 35, 35, 0.9, 9],
        ]
        cluster = HDBSCAN(min_samples=3, min_cluster_size=2)

        # Act
        poly = Utils.build_utility_pole_union(datas, cluster)

        # Assert
        self.assertTrue(isinstance(poly, Polygon))
        self.assertFalse(
            poly.is_empty,
            'Expected a non-empty union polygon with two poles.',
        )
        self.assertGreater(
            poly.area, 0, 'Union polygon should have a positive area.',
        )

    def test_build_utility_pole_union_multiple_poles_mst(self) -> None:
        """
        Test utility pole union with multiple poles triggering MST calculation.

        Verifies the branch where multiple poles are in the same cluster,
        which triggers the Minimum Spanning Tree (MST) algorithm and
        external tangent calculations for polygon creation.

        Returns:
            None
        """
        # Prepare 3 utility pole detections (class_id=9), ensuring >1 pole
        # with radius>0
        datas: list[list[float]] = [
            [10, 2, 20, 30, 0.9, 9],   # pole1
            [25, 1, 35, 30, 0.9, 9],   # pole2
            [50, 3, 60, 32, 0.9, 9],   # pole3
        ]

        # Use MagicMock to mock clusterer with min_samples and fit_predict
        clusterer = MagicMock()
        clusterer.min_samples = 2  # Let len(utility_poles)=3 >= min_samples=2
        clusterer.fit_predict.return_value = [
            0,
        ] * len(datas)  # All in the same cluster

        poly = Utils.build_utility_pole_union(datas, clusterer)
        self.assertIsInstance(poly, Polygon)
        self.assertFalse(
            poly.is_empty,
            'Expected a non-empty polygon from MST + tangents union.',
        )

    def test_get_outer_tangents_distance_less_than_radius_diff(self) -> None:
        """
        Test outer tangent calculation with insufficient circle separation.

        Verifies that when the distance between circle centres is less than
        the absolute difference of their radii, get_outer_tangents returns
        an empty list, indicating no valid tangent lines exist.

        Returns:
            None
        """
        # Assume large circle radius r1=10, small circle r2=1, centre
        # distance d=1
        # => d=1 < abs(10 - 1)=9 => triggers if d < abs(r1 - r2): return []
        cx1, cy1, r1 = 0, 0, 10
        cx2, cy2, r2 = 0, 1, 1

        lines = Utils.get_outer_tangents(cx1, cy1, r1, cx2, cy2, r2)
        self.assertEqual(
            lines, [], 'Expected empty list when d < abs(r1 - r2).',
        )

    def test_get_outer_tangents_distance_less_than_rdiff(self) -> None:
        """
        Test outer tangent calculation when distance is less than radius
        difference.

        Verifies that when the distance between circle centres d is less than
        r1 - r2 (radius difference), get_outer_tangents returns an empty list,
        properly handling geometric constraints.

        Returns:
            None
        """
        # Set large circle radius r1=10, small circle radius r2=3 => rdiff=7
        # Set centre distance d=3 (<7), this will return []
        cx1, cy1, r1 = 0, 0, 10
        cx2, cy2, r2 = 0, 3, 3  # Centre distance d=3

        lines = Utils.get_outer_tangents(cx1, cy1, r1, cx2, cy2, r2)
        self.assertEqual(lines, [], 'Expected empty list when d < (r1 - r2).')

    def test_get_outer_tangents_second_check(self) -> None:
        """
        Test outer tangent calculation with mocked distance recalculation.

        Verifies the branch where after ensuring r1>=r2, the recomputed
        distance is less than the radius difference, causing get_outer_tangents
        to return an empty list through mocked sqrt behaviour.

        Returns:
            None
        """
        # Configuration:
        # Input circles: First circle: centre(0,0), radius 10; Second circle:
        # centre(20,0), radius 5
        # First calculation:
        #   dx = 20, d = math.sqrt(20^2)=20, satisfies 20 >= abs(10 - 5)=5,
        # Second calculation: We use side_effect to make math.sqrt return 4,
        #   at this point rdiff = 10 - 5 = 5, 4 < 5, so empty list is returned.
        with patch('math.sqrt', side_effect=[20, 4]):
            lines = Utils.get_outer_tangents(0, 0, 10, 20, 0, 5)
        self.assertEqual(
            lines, [], 'Expected empty list when second sqrt result < rdiff.',
        )

    def test_count_people_in_polygon(self) -> None:
        """
        Test counting people within a polygon boundary.

        Verifies that count_people_in_polygon returns the correct number
        of unique people (based on centre points) within a given polygon,
        ensuring accurate person counting for safety monitoring.

        Returns:
            None
        """
        # Create a square polygon with bounds from (0,0) to (10,10)
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        # Create detection data with format: [left, top, right, bottom,
        # confidence, class_id]
        # where class_id 5 represents a person
        datas: list[list[float]] = [
            [1, 1, 3, 3, 0.9, 5],   # Person 1, centre = (2,2), inside polygon
            [4, 4, 8, 8, 0.9, 5],   # Person 2, centre = (6,6), inside polygon
            # Person 3, centre = (13,13), outside polygon
            [12, 12, 14, 14, 0.9, 5],
            [2, 2, 4, 4, 0.9, 5],   # Person 4, centre = (3,3), inside polygon
            [0, 0, 5, 5, 0.8, 2],   # Not a person (class_id != 5)
        ]

        # Calculate number of people inside the polygon
        count = Utils.count_people_in_polygon(poly, datas)

        # Expected three different centre points inside: (2,2), (6,6) and (3,3)
        self.assertEqual(
            count, 3, 'Expected 3 unique people inside the polygon.',
        )

    def test_polygons_to_coords(self) -> None:
        """
        Test polygon coordinates conversion functionality.

        Verifies that polygons_to_coords correctly converts Polygon
        and MultiPolygon objects into a list of coordinate lists,
        and properly skips any empty polygons.

        Returns:
            None
        """
        # Create a normal Polygon
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        # Create an empty Polygon (will be ignored)
        empty_poly = Polygon()
        # Create two additional Polygons and combine into MultiPolygon
        poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        poly3 = Polygon([(4, 4), (5, 4), (5, 5), (4, 5)])
        multipoly = MultiPolygon([poly2, poly3])

        # Call function: includes poly1, empty_poly, multipoly
        result = Utils.polygons_to_coords([poly1, empty_poly, multipoly])

        # Expected results
        expected_poly1 = [list(pt) for pt in poly1.exterior.coords]
        expected_poly2 = [list(pt) for pt in poly2.exterior.coords]
        expected_poly3 = [list(pt) for pt in poly3.exterior.coords]

        # Verify result contains poly1 coordinates
        self.assertIn(
            expected_poly1, result,
            'Expected poly1 coordinates to be in the result.',
        )
        # Verify result contains coordinates from multipoly's sub-Polygons
        self.assertIn(
            expected_poly2, result,
            'Expected poly2 coordinates to be in the result.',
        )
        self.assertIn(
            expected_poly3, result,
            'Expected poly3 coordinates to be in the result.',
        )
        # Since empty Polygon is ignored, result should have exactly three
        # items
        self.assertEqual(
            len(result), 3,
            'Expected 3 coordinate lists when skipping empty polygons.',
        )

    def test_get_outer_tangents_d_less_than_eps(self) -> None:
        """
        Test outer tangent calculation with coincident circle centres.

        Verifies that when the distance between circle centres is smaller
        than the epsilon threshold (i.e., circles have the same centre and
        equal radii), get_outer_tangents returns an empty list.

        Returns:
            None
        """
        # Set both circles with same centre and same radius, so d = 0,
        # abs(r1 - r2) = 0
        # First if condition (d < abs(r1 - r2): return []
        cx1, cy1, r1 = 0, 0, 10
        cx2, cy2, r2 = 0, 0, 10

        lines = Utils.get_outer_tangents(cx1, cy1, r1, cx2, cy2, r2)
        self.assertEqual(
            lines, [], 'Expected empty list when d < abs(r1 - r2).',
        )


class TestRedisManager(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the RedisManager class, covering basic Redis operations.

    This test class provides comprehensive coverage for Redis operations
    including error handling, connection management, and data persistence.
    """

    mock_redis: MagicMock
    rmgr: RedisManager

    @patch('src.utils.redis.Redis')
    def setUp(self, mock_redis: MagicMock) -> None:
        """
        Set up a RedisManager instance with a mocked Redis connection.

        Args:
            mock_redis (MagicMock): Mocked Redis class for testing.
        """
        self.mock_redis = MagicMock()
        self.mock_redis.get = AsyncMock()
        self.mock_redis.set = AsyncMock()
        self.mock_redis.delete = AsyncMock()
        self.mock_redis.close = AsyncMock()
        mock_redis.return_value = self.mock_redis
        self.rmgr = RedisManager()

    async def test_set_success(self) -> None:
        """
        Test successful Redis set operation.

        Verifies that the set method correctly calls the underlying
        Redis set operation with the provided key-value pair.
        """
        await self.rmgr.set('key', b'value')
        self.mock_redis.set.assert_called_once_with('key', b'value')

    async def test_set_error(self) -> None:
        """
        Test Redis set operation error handling.

        Ensures that exceptions during set operations are caught
        and logged appropriately without propagating to the caller.
        """
        self.mock_redis.set.side_effect = Exception('Redis error')
        with self.assertLogs(level='ERROR'):
            await self.rmgr.set('key', b'value')

    async def test_get_success(self) -> None:
        """
        Test successful Redis get operation.

        Verifies that the get method correctly retrieves values
        from Redis and returns the expected data.
        """
        self.mock_redis.get.return_value = b'val'
        val: bytes | None = await self.rmgr.get('key')
        self.mock_redis.get.assert_called_once_with('key')
        self.assertEqual(val, b'val')

    async def test_get_error(self) -> None:
        """
        Test Redis get operation error handling.

        Ensures that exceptions during get operations are caught,
        logged appropriately, and None is returned to the caller.
        """
        self.mock_redis.get.side_effect = Exception('Error')
        with self.assertLogs(level='ERROR'):
            val: bytes | None = await self.rmgr.get('key2')
            self.assertIsNone(val)

    async def test_delete_success(self) -> None:
        """
        Test successful Redis delete operation.

        Verifies that the delete method correctly calls the underlying
        Redis delete operation with the specified key.
        """
        await self.rmgr.delete('del_key')
        self.mock_redis.delete.assert_called_once_with('del_key')

    async def test_delete_error(self) -> None:
        """
        Test Redis delete operation error handling.

        Ensures that exceptions during delete operations are caught
        and logged appropriately without propagating to the caller.
        """
        self.mock_redis.delete.side_effect = Exception('DelErr')
        with self.assertLogs(level='ERROR'):
            await self.rmgr.delete('del_key2')

    async def test_close_connection_success(self) -> None:
        """
        Test successful Redis connection closure.

        Verifies that the close_connection method correctly calls
        the underlying Redis close operation.
        """
        await self.rmgr.close_connection()
        self.mock_redis.close.assert_called_once()

    async def test_close_connection_error(self) -> None:
        """
        Test Redis connection closure error handling.

        Ensures that exceptions during connection closure are caught
        and logged with appropriate error messages for debugging.
        """
        self.mock_redis.close.side_effect = Exception('CloseErr')
        with self.assertLogs(level='ERROR') as log:
            await self.rmgr.close_connection()
            self.assertIn(
                '[ERROR] Failed to close Redis connection: CloseErr',
                log.output[0],
            )


if __name__ == '__main__':
    unittest.main()

"""
pytest \
    --cov=src.utils \
    --cov-report=term-missing tests/src/utils_test.py
"""
