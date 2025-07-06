from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.db_management.services import stream_config_services
from examples.db_management.services.stream_config_services import (
    StreamConfigCreate,
)
from examples.db_management.services.stream_config_services import (
    StreamConfigUpdate,
)


class TestStreamConfigServices(unittest.IsolatedAsyncioTestCase):
    """Unit tests for stream_config_services module.

    Tests are designed using asynchronous mocks to simulate database
    and ORM model behaviour.
    """

    def setUp(self) -> None:
        """Set up common mock objects for each test.

        This method initialises mock database and configuration objects
        for use in each test case.
        """
        self.db: AsyncMock = AsyncMock()
        self.cfg: MagicMock = MagicMock()
        self.cfg.id = 1  # type: ignore[attr-defined]
        self.site_id: int = 123
        self.group_id: int = 456

        # Required fields for creating a new stream configuration
        self.payload: StreamConfigCreate = StreamConfigCreate(
            stream_name='test_stream',
            video_url='http://test/video',
            site_id=self.site_id,
        )

        # Fields for updating an existing stream configuration
        self.updates: StreamConfigUpdate = StreamConfigUpdate(
            stream_name='updated_stream',
            video_url='http://test/updated',
        )

    async def test_list_stream_configs(self) -> None:
        """Test listing stream configurations for a given site.

        Ensures that the list_stream_configs function returns the correct
        list of configurations for a specified site ID.
        """
        mock_result: MagicMock = MagicMock()
        mock_result.scalars.return_value.all.return_value = ['cfg1', 'cfg2']
        self.db.execute = AsyncMock(return_value=mock_result)

        configs: list = await stream_config_services.list_stream_configs(
            site_id=self.site_id,
            db=self.db,
        )

        self.assertEqual(configs, ['cfg1', 'cfg2'])

    async def test_create_stream_config_success(self) -> None:
        """Test successful creation of a stream configuration.

        Verifies that a new stream configuration is created and committed
        to the database without error.
        """
        self.db.commit = AsyncMock()
        self.db.refresh = AsyncMock()
        self.db.add = MagicMock()

        with patch(
            'examples.db_management.services.'
            'stream_config_services.StreamConfig',
        ) as MockStreamConfig:
            mock_cfg: MagicMock = MagicMock()
            MockStreamConfig.return_value = mock_cfg

            result: MagicMock = (
                await stream_config_services.create_stream_config(
                    payload=self.payload,
                    db=self.db,
                )
            )

            self.assertEqual(result, mock_cfg)
            self.db.add.assert_called_with(mock_cfg)
            self.db.commit.assert_awaited()
            self.db.refresh.assert_awaited_with(mock_cfg)

    async def test_create_stream_config_exception(self) -> None:
        """Test stream configuration creation handling database exception.

        Ensures that an HTTPException is raised and rollback is called
        if the database commit fails during creation.
        """
        self.db.commit = AsyncMock(side_effect=Exception('Database failure'))
        self.db.rollback = AsyncMock()
        self.db.add = MagicMock()

        with patch(
            'examples.db_management.services.'
            'stream_config_services.StreamConfig',
        ) as MockStreamConfig:
            mock_cfg: MagicMock = MagicMock()
            MockStreamConfig.return_value = mock_cfg

            with self.assertRaises(HTTPException) as context:
                await stream_config_services.create_stream_config(
                    payload=self.payload,
                    db=self.db,
                )

            self.assertEqual(context.exception.status_code, 500)
            self.db.rollback.assert_awaited()

    async def test_update_stream_config_success(self) -> None:
        """Test successful update of a stream configuration.

        Verifies that the update_stream_config function correctly updates
        the configuration and commits the changes.
        """
        self.db.commit = AsyncMock()

        await stream_config_services.update_stream_config(
            cfg=self.cfg,
            updates=self.updates,
            db=self.db,
        )

        self.db.commit.assert_awaited()
        # Verify attributes have been updated correctly
        self.assertEqual(self.cfg.stream_name, 'updated_stream')
        self.assertEqual(self.cfg.video_url, 'http://test/updated')

    async def test_update_stream_config_exception(self) -> None:
        """Test stream configuration update handling database exception.

        Ensures that an HTTPException is raised and rollback is called
        if the database commit fails during update.
        """
        self.db.commit = AsyncMock(side_effect=Exception('Database failure'))
        self.db.rollback = AsyncMock()

        updates: StreamConfigUpdate = StreamConfigUpdate(stream_name='fail')

        with self.assertRaises(HTTPException) as context:
            await stream_config_services.update_stream_config(
                cfg=self.cfg,
                updates=updates,
                db=self.db,
            )

        self.assertEqual(context.exception.status_code, 500)
        self.db.rollback.assert_awaited()

    async def test_delete_stream_config_success(self) -> None:
        """Test successful deletion of a stream configuration.

        Verifies that the delete_stream_config function deletes the
        configuration and commits the transaction.
        """
        self.db.delete = AsyncMock()
        self.db.commit = AsyncMock()

        await stream_config_services.delete_stream_config(
            cfg=self.cfg,
            db=self.db,
        )

        self.db.delete.assert_awaited_with(self.cfg)
        self.db.commit.assert_awaited()

    async def test_delete_stream_config_exception(self) -> None:
        """Test stream configuration deletion handling database exception.

        Ensures that an HTTPException is raised and rollback is called
        if the database commit fails during deletion.
        """
        self.db.delete = AsyncMock()
        self.db.commit = AsyncMock(side_effect=Exception('Database failure'))
        self.db.rollback = AsyncMock()

        with self.assertRaises(HTTPException) as context:
            await stream_config_services.delete_stream_config(
                cfg=self.cfg,
                db=self.db,
            )

        self.assertEqual(context.exception.status_code, 500)
        self.db.rollback.assert_awaited()

    async def test_get_group_stream_limit_success(self) -> None:
        """Test retrieving stream limits for a specific group successfully.

        Ensures that the correct current and maximum stream limits are
        returned for a given group ID.
        """
        mock_group: MagicMock = MagicMock()
        mock_group.max_allowed_streams = 5
        self.db.get = AsyncMock(return_value=mock_group)
        self.db.scalar = AsyncMock(return_value=3)

        current_streams: int
        max_streams: int
        current_streams, max_streams = (
            await stream_config_services.get_group_stream_limit(
                group_id=self.group_id,
                db=self.db,
            )
        )

        self.assertEqual(current_streams, 3)
        self.assertEqual(max_streams, 5)

    async def test_get_group_stream_limit_group_not_found(self) -> None:
        """Test handling when group is not found in retrieving stream limits.

        Ensures that an HTTPException with status 404 is raised if the
        group is not found in the database.
        """
        self.db.get = AsyncMock(return_value=None)

        with self.assertRaises(HTTPException) as context:
            await stream_config_services.get_group_stream_limit(
                group_id=self.group_id,
                db=self.db,
            )

        self.assertEqual(context.exception.status_code, 404)


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.db_management.services.stream_config_services\
    --cov-report=term-missing\
        tests/examples/db_management/services/stream_config_services_test.py
'''
