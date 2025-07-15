from __future__ import annotations

import asyncio
import base64
import os
import unittest
from datetime import datetime
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
from shapely import Polygon
from sklearn.cluster import HDBSCAN
from watchdog.events import FileModifiedEvent

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
                # First call returns the original token
                return 'ORIGINAL'
            else:
                # Subsequent calls simulate a changed token
                return 'CHANGED'
        value = self._data.get(key, default)
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
        value = self._data[key]
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

        def change_token_side_effect(*_, **__) -> AsyncMock:
            self.shared_token['refresh_token'] = 'CHANGED'
            return mock_resp

        mock_sessinst.post.side_effect = change_token_side_effect

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
    def test_is_expired_with_valid_date(self):
        # Test with a past date (should return True)
        past_date = (datetime.now() - timedelta(days=1)).isoformat()
        self.assertTrue(Utils.is_expired(past_date))

        # Test with a future date (should return False)
        future_date = (datetime.now() + timedelta(days=1)).isoformat()
        self.assertFalse(Utils.is_expired(future_date))

    def test_is_expired_with_invalid_date(self):
        # Test with an invalid ISO 8601 date (should return False)
        invalid_date = '2024-13-01T00:00:00'
        self.assertFalse(Utils.is_expired(invalid_date))

    def test_is_expired_with_none(self):
        # Test with None (should return False)
        self.assertFalse(Utils.is_expired(None))

    def test_encode(self):
        # Test encoding a string
        value = 'test_value'
        encoded_value = Utils.encode(value)
        expected_value = base64.urlsafe_b64encode(
            value.encode('utf-8'),
        ).decode('utf-8')
        self.assertEqual(encoded_value, expected_value)

    def test_encode_frame_success(self):
        """
        Test encoding a valid frame using `encode_frame`.
        """
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        encoded_frame = Utils.encode_frame(frame)
        self.assertIsNotNone(encoded_frame)

    def test_encode_frame_failure(self):
        """
        Test encoding an invalid frame using `encode_frame`.
        """
        invalid_frame = None
        encoded_frame = Utils.encode_frame(invalid_frame)
        self.assertIsNone(encoded_frame)

    def test_generate_message_working_hours(self):
        """
        Test generating a message during working hours.
        """
        stream_name = 'Test Stream'
        detection_time = datetime.now()
        warnings = ['Warning 1', 'Warning 2']
        controlled_zone_warning = []
        language = 'en'
        is_working_hour = True

        with patch.object(
            Translator,
            'translate_warning',
            return_value=['Translated Warning 1', 'Translated Warning 2'],
        ):
            message = Utils.generate_message(
                stream_name,
                detection_time,
                warnings,
                controlled_zone_warning,
                language,
                is_working_hour,
            )

        self.assertIn('Translated Warning 1', message)
        self.assertIn('Translated Warning 2', message)

    def test_generate_message_non_working_hours(self):
        """
        Test generating a message outside working hours.
        """
        stream_name = 'Test Stream'
        detection_time = datetime.now()
        warnings = []
        controlled_zone_warning = ['Controlled Area Warning']
        language = 'zh-TW'
        is_working_hour = False

        with patch.object(
            Translator,
            'translate_warning',
            return_value=['受控區域警告'],
        ):
            message = Utils.generate_message(
                stream_name,
                detection_time,
                warnings,
                controlled_zone_warning,
                language,
                is_working_hour,
            )

        self.assertIn('受控區域警告', message)

    def test_generate_message_no_message(self):
        """
        Test generating a message when no conditions are met.
        """
        stream_name = 'Test Stream'
        detection_time = datetime.now()
        warnings = []
        controlled_zone_warning = []
        language = 'en'
        is_working_hour = True

        message = Utils.generate_message(
            stream_name,
            detection_time,
            warnings,
            controlled_zone_warning,
            language,
            is_working_hour,
        )

        self.assertIsNone(message)

    def test_should_notify_true(self):
        """
        Test `should_notify` returning True when cooldown period has passed.
        """
        timestamp = int(datetime.now().timestamp())
        last_notification_time = timestamp - 400
        cooldown_period = 300

        self.assertTrue(
            Utils.should_notify(
                timestamp, last_notification_time, cooldown_period,
            ),
        )

    def test_should_notify_false(self):
        """
        Test `should_notify` returning False
        when cooldown period has not passed.
        """
        timestamp = int(datetime.now().timestamp())
        last_notification_time = timestamp - 200
        cooldown_period = 300

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


class TestFileEventHandler(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    async def test_on_modified_triggers_callback(self):
        # Create a mock callback function
        mock_callback = AsyncMock()

        # File path to watch
        file_path = '/path/to/test/file.txt'

        # Create an instance of FileEventHandler
        event_handler = FileEventHandler(
            file_path=file_path,
            callback=mock_callback,
            loop=asyncio.get_running_loop(),
        )

        # Create a mock event for a file modification
        event = FileModifiedEvent(file_path)

        # Trigger the on_modified event
        event_handler.on_modified(event)

        # Assert that the callback was called
        mock_callback.assert_called_once()

    async def test_on_modified_different_file(self):
        # Create a mock callback function
        mock_callback = AsyncMock()

        # File path to watch
        file_path = '/path/to/test/file.txt'

        # Create an instance of FileEventHandler
        event_handler = FileEventHandler(
            file_path=file_path, callback=mock_callback, loop=self.loop,
        )

        # Create a mock event for a different file modification
        different_file_path = '/path/to/other/file.txt'
        event = FileModifiedEvent(different_file_path)

        # Trigger the on_modified event
        event_handler.on_modified(event)

        # Assert that the callback was not called
        mock_callback.assert_not_called()


class TestRedisManager(unittest.IsolatedAsyncioTestCase):
    """
    Test cases for the RedisManager class
    """

    @patch('src.utils.redis.Redis')
    def setUp(self, mock_redis):
        """
        Set up a RedisManager instance with a mocked Redis connection
        """
        # Mock Redis instance
        self.mock_redis_instance = MagicMock()
        self.mock_redis_instance.get = AsyncMock()
        self.mock_redis_instance.set = AsyncMock()
        self.mock_redis_instance.delete = AsyncMock()
        self.mock_redis_instance.xadd = AsyncMock()
        self.mock_redis_instance.xread = AsyncMock()
        self.mock_redis_instance.close = AsyncMock()
        self.mock_redis_instance.delete.side_effect = AsyncMock()

        mock_redis.return_value = self.mock_redis_instance

        # Initialize RedisManager
        self.redis_manager = RedisManager()

    async def test_set_success(self):
        """
        Test successful set operation
        """
        key = 'test_key'
        value = b'test_value'

        # Call the set method
        await self.redis_manager.set(key, value)

        # Assert that the Redis set method was called with correct parameters
        self.mock_redis_instance.set.assert_called_once_with(key, value)

    async def test_set_error(self):
        """
        Simulate an exception during the Redis set operation
        """
        key = 'test_key'
        value = b'test_value'
        self.mock_redis_instance.set.side_effect = Exception('Redis error')

        # Call the set method and verify it handles the exception
        with self.assertLogs(level='ERROR'):
            await self.redis_manager.set(key, value)

    async def test_get_success(self):
        """
        Mock the Redis get method to return a value
        """
        key = 'test_key'
        expected_value = b'test_value'
        self.mock_redis_instance.get.return_value = expected_value

        # Call the get method
        value = await self.redis_manager.get(key)

        # Assert that the Redis get method was called with correct parameters
        self.mock_redis_instance.get.assert_called_once_with(key)

        # Assert the value returned is correct
        self.assertEqual(value, expected_value)

    async def test_get_error(self):
        """
        Simulate an exception during the Redis get operation
        """
        key = 'test_key'
        self.mock_redis_instance.get.side_effect = Exception('Redis error')

        # Call the get method and verify it handles the exception
        with self.assertLogs(level='ERROR'):
            value = await self.redis_manager.get(key)
            self.assertIsNone(value)

    async def test_delete_success(self):
        """
        Test successful delete operation
        """
        key = 'test_key'

        # Call the delete method
        await self.redis_manager.delete(key)

        # Assert that the Redis delete method
        # was called with correct parameters
        self.mock_redis_instance.delete.assert_called_once_with(key)

    async def test_delete_error(self):
        """
        Simulate an exception during the Redis delete operation
        """
        key = 'test_key'
        self.mock_redis_instance.delete.side_effect = Exception('Redis error')

        # Call the delete method and verify it handles the exception
        with self.assertLogs(level='ERROR'):
            await self.redis_manager.delete(key)

    async def test_add_to_stream(self):
        """
        Test adding data to a Redis stream
        """
        stream_name = 'test_stream'
        data = {'field1': b'value1', 'field2': b'value2'}
        maxlen = 5

        # Call the add_to_stream method
        await self.redis_manager.add_to_stream(stream_name, data, maxlen)

        # Assert that the xadd method was called with the correct parameters
        self.mock_redis_instance.xadd.assert_called_once_with(
            stream_name, data, maxlen=maxlen,
        )

    async def test_add_to_stream_error(self):
        """
        Simulate an exception during adding data to a Redis stream
        """
        stream_name = 'test_stream'
        data = {'field1': b'value1', 'field2': b'value2'}
        maxlen = 5
        self.mock_redis_instance.xadd.side_effect = Exception('Redis error')

        # Call the add_to_stream method and verify it handles the exception
        with self.assertLogs(level='ERROR') as log:
            await self.redis_manager.add_to_stream(stream_name, data, maxlen)
            self.assertIn(
                f"Error adding to Redis stream {stream_name}: Redis error",
                log.output[0],
            )

    async def test_read_from_stream(self):
        """
        Test reading data from a Redis stream
        """
        stream_name = 'test_stream'
        last_id = '0'
        expected_messages = [
            ('1-0', {'field1': b'value1', 'field2': b'value2'}),
            ('2-0', {'field3': b'value3'}),
        ]
        self.mock_redis_instance.xread.return_value = expected_messages

        # Call the read_from_stream method
        messages = await self.redis_manager.read_from_stream(
            stream_name,
            last_id,
        )

        # Assert that the xread method was called with correct parameters
        self.mock_redis_instance.xread.assert_called_once_with(
            {stream_name: last_id},
        )

        # Assert that the returned messages match the expected messages
        self.assertEqual(messages, expected_messages)

    async def test_read_from_stream_error(self):
        """
        Simulate an exception during reading from a Redis stream
        """
        stream_name = 'test_stream'
        last_id = '0'
        self.mock_redis_instance.xread.side_effect = Exception('Redis error')

        # Call the read_from_stream method and verify it handles the exception
        with self.assertLogs(level='ERROR') as log:
            messages = await self.redis_manager.read_from_stream(
                stream_name, last_id,
            )
            self.assertEqual(messages, [])
            self.assertIn(
                f"Error reading from Redis stream {stream_name}: Redis error",
                log.output[0],
            )

    async def test_delete_stream_success(self):
        """
        Test successful deletion of a Redis stream
        """
        stream_name = 'test_stream'

        # Call the delete_stream method
        await self.redis_manager.delete_stream(stream_name)

        # Assert that the Redis delete method was called
        self.mock_redis_instance.delete.assert_called_once_with(stream_name)

    async def test_delete_stream_error(self):
        """
        Simulate an exception during deleting a Redis stream
        """
        stream_name = 'test_stream'
        self.mock_redis_instance.delete.side_effect = Exception('Redis error')

        # Call the delete_stream method and verify it handles the exception
        with self.assertLogs(level='ERROR') as log:
            await self.redis_manager.delete_stream(stream_name)
            self.assertIn(
                f"Error deleting Redis stream {stream_name}: Redis error",
                log.output[0],
            )

    async def test_close_connection(self):
        """
        Test closing the Redis connection
        """
        # Call the close_connection method
        await self.redis_manager.close_connection()

        # Assert that the Redis close method was called
        self.mock_redis_instance.close.assert_called_once()

    async def test_close_connection_error(self):
        """
        Simulate an exception during closing the Redis connection
        """
        self.mock_redis_instance.close.side_effect = Exception('Redis error')

        # Call the close_connection method and verify it handles the exception
        with self.assertLogs(level='ERROR') as log:
            await self.redis_manager.close_connection()
            self.assertIn(
                '[ERROR] Failed to close Redis connection: Redis error',
                log.output[0],
            )

    async def test_store_to_redis_success(self):
        """
        Test successfully storing data to Redis.
        """
        site = 'Test Site'
        stream_name = 'Test Stream'
        frame_bytes = b'encoded_frame'
        warnings = ['Warning: Someone is not wearing a hardhat!']
        language = 'en'

        # Mock the translate_warning function to return a specific value
        with patch.object(
            Translator,
            'translate_warning',
            return_value=['Translated Warning'],
        ):
            await self.redis_manager.store_to_redis(
                site, stream_name, frame_bytes, warnings, language,
            )

        # Assert that xadd was called once
        self.mock_redis_instance.xadd.assert_called_once()
        call_args = self.mock_redis_instance.xadd.call_args[0]

        # Assert that the stream name is correctly formed
        self.assertTrue(call_args[0].startswith('stream_frame'))
        # Assert that the frame bytes are correctly included
        self.assertEqual(call_args[1]['frame'], b'encoded_frame')
        # Assert that the warnings are correctly translated and included
        self.assertEqual(
            call_args[1]['warnings'],
            'Translated Warning',
        )

    async def test_store_to_redis_no_frame(self):
        """
        Test not storing data when frame_bytes is None.
        """
        site = 'Test Site'
        stream_name = 'Test Stream'
        frame_bytes = None  # No frame provided
        warnings = ['Warning: Test warning']
        language = 'en'

        await self.redis_manager.store_to_redis(
            site, stream_name, frame_bytes, warnings, language,
        )

        # Assert that xadd was not called
        self.mock_redis_instance.xadd.assert_not_called()

    async def test_store_to_redis_translate_warning(self):
        """
        Test translating warnings before storing to Redis.
        """
        site = 'Test Site'
        stream_name = 'Test Stream'
        frame_bytes = b'encoded_frame'
        warnings = ['Warning: 2 people have entered the controlled area!']
        language = 'zh-TW'

        # Mock the translate_warning function to return a specific value
        with patch.object(
            Translator, 'translate_warning', return_value=['警告: 2個人進入受控區域!'],
        ):
            await self.redis_manager.store_to_redis(
                site, stream_name, frame_bytes, warnings, language,
            )

        # Assert that xadd was called once
        self.mock_redis_instance.xadd.assert_called_once()
        call_args = self.mock_redis_instance.xadd.call_args[0]

        # Assert that the stream name is correctly formed
        self.assertTrue(call_args[0].startswith('stream_frame'))
        # Assert that the frame bytes are correctly included
        self.assertEqual(call_args[1]['frame'], b'encoded_frame')
        # Assert that the warnings are correctly translated and included
        self.assertEqual(
            call_args[1]['warnings'],
            '警告: 2個人進入受控區域!',
        )

    async def test_store_to_redis_no_warning(self):
        """
        Test storing 'No warning' when there are no warnings.
        """
        site = 'Test Site'
        stream_name = 'Test Stream'
        frame_bytes = b'encoded_frame'
        warnings = []
        language = 'zh-TW'

        # Mock the translate_warning function to return a specific value
        with patch.object(
            Translator, 'translate_warning', return_value=['無警告'],
        ):
            await self.redis_manager.store_to_redis(
                site, stream_name, frame_bytes, warnings, language,
            )

        # Assert that xadd was called once
        self.mock_redis_instance.xadd.assert_called_once()
        call_args = self.mock_redis_instance.xadd.call_args[0]

        # Assert that the warnings are correctly translated as 'No warning'
        self.assertEqual(
            call_args[1]['warnings'],
            '無警告',
        )

    async def test_store_to_redis_exception(self):
        """
        Test handling an exception during storing to Redis.
        """
        site = 'Test Site'
        stream_name = 'Test Stream'
        frame_bytes = b'encoded_frame'
        warnings = ['Warning: Test warning']
        language = 'en'

        # Mock add_to_stream to raise an exception
        with patch.object(
            self.redis_manager,
            'add_to_stream',
            side_effect=Exception('Redis error'),
        ):
            with self.assertLogs(level='ERROR') as log:
                await self.redis_manager.store_to_redis(
                    site, stream_name, frame_bytes, warnings, language,
                )

                # Assert that an error was logged
                self.assertIn(
                    'Error storing data to Redis: Redis error', log.output[0],
                )


if __name__ == '__main__':
    unittest.main()
