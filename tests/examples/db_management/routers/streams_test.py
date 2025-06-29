from __future__ import annotations

import datetime
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.db_management.routers import streams
from examples.db_management.schemas.stream_config import StreamConfigCreate
from examples.db_management.schemas.stream_config import StreamConfigUpdate


class TestStreamsRouter(unittest.IsolatedAsyncioTestCase):
    """Unit tests for stream configuration router endpoints.

    This test class covers all key branches and exceptions for the FastAPI
    stream configuration endpoints, using unittest and mock. All tests use
    British English spelling and Google style docstrings.
    """

    def setUp(self) -> None:
        """Set up common mocks for each test.

        This method initialises a mock database session, a mock current user
        (admin by default), and a mock site object for use in all tests.
        """
        self.db: AsyncMock = AsyncMock()
        self.current_user: MagicMock = MagicMock()
        self.current_user.role = 'admin'
        self.current_user.group_id = 1
        self.site_mock: MagicMock = MagicMock(
            group_id=1, group=MagicMock(max_allowed_streams=5),
        )

    @patch('examples.db_management.routers.streams.list_stream_configs')
    @patch('examples.db_management.routers.streams.get_group_stream_limit')
    @patch(
        'examples.db_management.routers.streams.is_super_admin',
        return_value=False,
    )
    async def test_endpoint_list_stream_configs(
        self,
        mock_is_super_admin: MagicMock,
        mock_limit: AsyncMock,
        mock_list: AsyncMock,
    ) -> None:
        """Test listing stream configurations successfully.

        Ensures the endpoint returns a list of configs when the site exists.
        """
        mock_limit.return_value = (2, 5)
        # Mock a complete stream config object
        mock_config: MagicMock = MagicMock(
            id=1,
            stream_name='test',
            video_url='url',
            model_key='yolo',
            detect_with_server=True,
            store_in_redis=False,
            work_start_hour=8,
            work_end_hour=17,
            detect_no_safety_vest_or_helmet=True,
            detect_near_machinery_or_vehicle=False,
            detect_in_restricted_area=False,
            detect_in_utility_pole_restricted_area=False,
            detect_machinery_close_to_pole=False,
            expire_date=None,
            updated_at=datetime.datetime.now(),
            site_id=1,
            site=self.site_mock,
        )
        mock_list.return_value = [mock_config]
        self.db.get.return_value = self.site_mock

        response = await streams.endpoint_list_stream_configs(
            1, self.db, self.current_user,
        )

        self.assertEqual(len(response), 1)
        self.assertEqual(response[0].stream_name, 'test')

    @patch('examples.db_management.routers.streams.create_stream_config')
    @patch('examples.db_management.routers.streams.get_group_stream_limit')
    @patch(
        'examples.db_management.routers.streams.is_super_admin',
        return_value=False,
    )
    async def test_endpoint_create_stream_config(
        self,
        mock_is_super_admin: MagicMock,
        mock_limit: AsyncMock,
        mock_create: AsyncMock,
    ) -> None:
        """Test creating a stream configuration successfully.

        Ensures the endpoint returns the new config ID when creation succeeds.
        """
        mock_limit.return_value = (1, 5)
        mock_create.return_value = MagicMock(id=1)

        payload: StreamConfigCreate = StreamConfigCreate(
            site_id=1,
            stream_name='stream',
            video_url='url',
        )
        self.db.get.return_value = self.site_mock

        response = await streams.endpoint_create_stream_config(
            payload, self.db, self.current_user,
        )

        self.assertEqual(response['id'], 1)

    async def test_endpoint_create_stream_config_limit_reached(self) -> None:
        """Test creating a stream configuration when limit is reached.

        Should raise HTTP 403 if the group stream limit is already reached.
        """
        self.db.get.return_value = self.site_mock

        with patch(
            'examples.db_management.routers.streams.get_group_stream_limit',
            AsyncMock(return_value=(5, 5)),
        ):
            payload: StreamConfigCreate = StreamConfigCreate(
                site_id=1,
                stream_name='stream',
                video_url='url',
            )
            with self.assertRaises(HTTPException) as ctx:
                await streams.endpoint_create_stream_config(
                    payload, self.db, self.current_user,
                )

            self.assertEqual(ctx.exception.status_code, 403)

    @patch('examples.db_management.routers.streams.update_stream_config')
    @patch(
        'examples.db_management.routers.streams.is_super_admin',
        return_value=False,
    )
    async def test_endpoint_update_stream_config(
        self,
        mock_is_super_admin: MagicMock,
        mock_update: AsyncMock,
    ) -> None:
        """Test updating stream configuration successfully.

        Ensures the endpoint returns a success message when update is valid.
        """
        cfg_mock: MagicMock = MagicMock(site=self.site_mock, stream_name='old')
        self.db.get.return_value = cfg_mock
        self.db.scalar.return_value = None

        payload: StreamConfigUpdate = StreamConfigUpdate(stream_name='new')

        response = await streams.endpoint_update_stream_config(
            1, payload, self.db, self.current_user,
        )

        self.assertEqual(
            response['message'],
            'Stream configuration updated successfully.',
        )

    @patch(
        'examples.db_management.routers.streams.is_super_admin',
        return_value=False,
    )
    async def test_endpoint_update_stream_config_name_conflict(
        self,
        mock_is_super_admin: MagicMock,
    ) -> None:
        """Test updating stream configuration with name conflict.

        Should raise HTTP 400 if the new name already exists in the site.
        """
        cfg_mock: MagicMock = MagicMock(site=self.site_mock, stream_name='old')
        self.db.get.return_value = cfg_mock
        self.db.scalar.return_value = MagicMock()

        payload: StreamConfigUpdate = StreamConfigUpdate(
            stream_name='conflict',
        )

        with self.assertRaises(HTTPException) as ctx:
            await streams.endpoint_update_stream_config(
                1, payload, self.db, self.current_user,
            )

        self.assertEqual(ctx.exception.status_code, 400)

    @patch('examples.db_management.routers.streams.delete_stream_config')
    @patch(
        'examples.db_management.routers.streams.is_super_admin',
        return_value=False,
    )
    async def test_endpoint_delete_stream_config(
        self,
        mock_is_super_admin: MagicMock,
        mock_delete: AsyncMock,
    ) -> None:
        """Test deleting stream configuration successfully.

        Ensures the endpoint returns a success message when deletion is valid.
        """
        cfg_mock: MagicMock = MagicMock(site=self.site_mock)
        self.db.get.return_value = cfg_mock

        response = await streams.endpoint_delete_stream_config(
            1, self.db, self.current_user,
        )

        mock_delete.assert_awaited_with(cfg_mock, self.db)
        self.assertEqual(
            response['message'],
            'Stream configuration deleted successfully.',
        )

    @patch('examples.db_management.routers.streams.get_group_stream_limit')
    @patch(
        'examples.db_management.routers.streams.is_super_admin',
        return_value=False,
    )
    async def test_endpoint_group_stream_limit(
        self,
        mock_is_super_admin: MagicMock,
        mock_limit: AsyncMock,
    ) -> None:
        """Test retrieving group stream limit successfully.

        Ensures the endpoint returns correct stream count
        and limit for the group.
        """
        mock_limit.return_value = (3, 10)
        self.current_user.role = 'admin'
        self.current_user.group_id = 1

        response = await streams.endpoint_group_stream_limit(
            1, self.db, self.current_user,
        )

        self.assertEqual(response['max_allowed_streams'], 10)
        self.assertEqual(response['current_streams_count'], 3)

    async def test_list_stream_configs_site_not_found(self) -> None:
        """Should raise 404 if site not found.

        This test ensures the endpoint returns HTTP 404
        if the site does not exist.
        """
        self.db.get.return_value = None
        with self.assertRaises(HTTPException) as ctx:
            await streams.endpoint_list_stream_configs(
                1, self.db, self.current_user,
            )
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_create_stream_config_site_not_found(self) -> None:
        """Should raise 404 if site not found when creating config.

        This test ensures the endpoint returns HTTP 404
        if the site does not exist when creating a config.
        """
        self.db.get.return_value = None
        payload: StreamConfigCreate = StreamConfigCreate(
            site_id=1,
            stream_name='stream',
            video_url='url',
        )
        with self.assertRaises(HTTPException) as ctx:
            await streams.endpoint_create_stream_config(
                payload, self.db, self.current_user,
            )
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_update_stream_config_not_found(self) -> None:
        """Should raise 404 if config not found when updating.

        This test ensures the endpoint returns HTTP 404
        if the config does not exist when updating.
        """
        self.db.get.return_value = None
        payload: StreamConfigUpdate = StreamConfigUpdate(stream_name='new')
        with self.assertRaises(HTTPException) as ctx:
            await streams.endpoint_update_stream_config(
                1, payload, self.db, self.current_user,
            )
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_delete_stream_config_not_found(self) -> None:
        """Should raise 404 if config not found when deleting.

        This test ensures the endpoint returns HTTP 404
        if the config does not exist when deleting.
        """
        self.db.get.return_value = None
        with self.assertRaises(HTTPException) as ctx:
            await streams.endpoint_delete_stream_config(
                1, self.db, self.current_user,
            )
        self.assertEqual(ctx.exception.status_code, 404)

    @patch('examples.db_management.routers.streams.get_group_stream_limit')
    @patch(
        'examples.db_management.routers.streams.is_super_admin',
        return_value=False,
    )
    async def test_group_stream_limit_permission_denied(
        self,
        mock_is_super_admin: MagicMock,
        mock_limit: AsyncMock,
    ) -> None:
        """Should raise 403 if not super admin and not group admin.

        This test ensures the endpoint returns HTTP 403
        if the user is neither super admin nor group admin.
        """
        self.current_user.role = 'user'
        self.current_user.group_id = 2
        with self.assertRaises(HTTPException) as ctx:
            await streams.endpoint_group_stream_limit(
                1, self.db, self.current_user,
            )
        self.assertEqual(ctx.exception.status_code, 403)


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.db_management.routers.streams\
    --cov-report=term-missing\
        tests/examples/db_management/routers/streams_test.py
'''
