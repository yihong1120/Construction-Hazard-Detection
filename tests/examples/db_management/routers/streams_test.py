from __future__ import annotations

import datetime
import unittest
from datetime import datetime as DateTime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.auth.models import Group
from examples.auth.models import Site
from examples.auth.models import User
from examples.db_management.routers import streams
from examples.db_management.schemas.stream_config import SiteStreamConfigItem
from examples.db_management.schemas.stream_config import SiteStreamConfigUpsert
from examples.db_management.schemas.stream_config import StreamConfigCreate
from examples.db_management.schemas.stream_config import StreamConfigUpdate
from examples.db_management.services import stream_config_services


class TestStreamsRouter(unittest.IsolatedAsyncioTestCase):
    """Unit tests for stream configuration router endpoints."""

    def setUp(self) -> None:
        """Set up common mocks for each test.

        This method initialises a mock database session, a mock current user
        (admin by default), and a mock site object for use in all tests.
        """
        self.db: AsyncMock = AsyncMock()
        self.db.add = MagicMock()
        self.current_user: MagicMock = MagicMock()
        self.current_user.role = 'admin'
        self.current_user.group_id = 1
        group_mock: MagicMock = MagicMock()
        group_mock.id = 1
        group_mock.max_allowed_streams = 5
        other_group_mock: MagicMock = MagicMock()
        other_group_mock.id = 2
        other_group_mock.max_allowed_streams = 5
        self.site_mock: MagicMock = MagicMock()
        self.site_mock.groups = [group_mock, other_group_mock]

    @patch(
        'examples.db_management.services.stream_config_services.'
        'list_stream_configs',
    )
    @patch(
        'examples.db_management.services.stream_config_services.'
        'get_group_stream_limit',
    )
    @patch(
        'examples.db_management.services.stream_config_services.'
        'is_super_admin',
        return_value=False,
    )
    async def test_endpoint_list_stream_configs(
        self,
        mock_is_super_admin: MagicMock,
        mock_limit: AsyncMock,
        mock_list: AsyncMock,
    ) -> None:
        """Test endpoint list stream configs.

        Args:
            mock_is_super_admin: Value used by this callable.
            mock_limit: Value used by this callable.
            mock_list: Value used by this callable.
        """
        mock_limit.return_value = (2, 5)
        # Mock a complete stream config object
        mock_config: MagicMock = MagicMock(
            id=1,
            stream_name='test',
            video_url='url',
            model_key='yolo',
            recognition_enabled=True,
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
            group_id=1,
            site=self.site_mock,
        )
        mock_list.return_value = [mock_config]
        self.db.get.return_value = self.site_mock

        response = await streams.endpoint_list_stream_configs(
            1,
            self.db,
            self.current_user,
        )

        self.assertEqual(len(response), 1)
        self.assertEqual(response[0].stream_name, 'test')
        self.assertTrue(response[0].recognition_enabled)
        mock_list.assert_awaited_once_with(1, self.db, group_id=1)

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
        """Test endpoint create stream config.

        Args:
            mock_is_super_admin: Value used by this callable.
            mock_limit: Value used by this callable.
            mock_create: Value used by this callable.
        """
        mock_limit.return_value = (1, 5)
        mock_create.return_value = MagicMock(id=1)

        payload: StreamConfigCreate = StreamConfigCreate(
            site_id=1,
            stream_name='stream',
            video_url='url',
            group_id=2,
        )
        self.db.get.return_value = self.site_mock

        response = await streams.endpoint_create_stream_config(
            payload,
            self.db,
            self.current_user,
        )

        self.assertEqual(response['id'], 1)
        assert mock_create.await_args is not None
        created_payload = mock_create.await_args.args[0]
        self.assertEqual(created_payload.group_id, 1)

    async def test_endpoint_create_stream_config_limit_reached(self) -> None:
        """Test endpoint create stream config limit reached.
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
                    payload,
                    self.db,
                    self.current_user,
                )

            self.assertEqual(ctx.exception.status_code, 403)

    @patch('examples.db_management.routers.streams.create_stream_config')
    @patch('examples.db_management.routers.streams.update_stream_config')
    @patch('examples.db_management.routers.streams.get_group_stream_limit')
    @patch('examples.db_management.routers.streams.list_stream_configs')
    @patch(
        'examples.db_management.routers.streams.is_super_admin',
        return_value=False,
    )
    async def test_endpoint_put_site_stream_config_uses_site_scope(
        self,
        mock_is_super_admin: MagicMock,
        mock_list: AsyncMock,
        mock_limit: AsyncMock,
        mock_update: AsyncMock,
        mock_create: AsyncMock,
    ) -> None:
        """Site-level upsert uses current admin group, not frontend group."""
        existing = MagicMock(
            id=11,
            stream_name='Old Cam',
            video_url='rtsp://old',
            model_key='yolo26n',
            recognition_enabled=True,
            work_start_hour=7,
            work_end_hour=18,
            detect_no_safety_vest_or_helmet=False,
            detect_near_machinery_or_vehicle=False,
            detect_in_restricted_area=False,
            detect_in_utility_pole_restricted_area=False,
            detect_machinery_close_to_pole=False,
            expire_date=None,
            updated_at=datetime.datetime.now(),
            group_id=1,
        )
        created = MagicMock(id=12)
        mock_create.return_value = created
        mock_limit.return_value = (1, 5)
        mock_list.side_effect = [[existing], [existing]]
        self.db.get.return_value = self.site_mock
        self.db.scalar.return_value = None
        payload = SiteStreamConfigUpsert(
            streams=[
                SiteStreamConfigItem(
                    id=11,
                    stream_name='Old Cam',
                    video_url='rtsp://updated',
                    recognition_enabled=False,
                ),
                SiteStreamConfigItem(
                    stream_name='New Cam',
                    video_url='rtsp://new',
                ),
            ],
        )

        with (
            patch.object(
                stream_config_services,
                'list_stream_configs',
                new_callable=AsyncMock,
                return_value=[existing],
            ),
            patch.object(
                stream_config_services,
                'get_group_stream_limit',
                new_callable=AsyncMock,
                return_value=(2, 5),
            ),
        ):
            response = await streams.endpoint_put_site_stream_config(
                1,
                payload,
                self.db,
                self.current_user,
            )

        mock_list.assert_awaited_once_with(1, self.db)
        mock_update.assert_not_awaited()
        mock_create.assert_not_awaited()
        self.db.add.assert_called_once()
        created_config = self.db.add.call_args.args[0]
        self.assertEqual(created_config.site_id, 1)
        self.assertEqual(created_config.group_id, 1)
        self.assertEqual(created_config.video_url, 'rtsp://new')
        self.assertTrue(created_config.recognition_enabled)
        self.assertFalse(existing.recognition_enabled)
        self.db.commit.assert_awaited_once()
        self.assertEqual(len(response), 1)

    async def test_endpoint_put_site_stream_config_rejects_duplicate_names(
        self,
    ) -> None:
        """Site-level payload cannot contain duplicate stream names."""
        self.db.get.return_value = self.site_mock
        payload = SiteStreamConfigUpsert(
            streams=[
                SiteStreamConfigItem(
                    stream_name='Cam1',
                    video_url='rtsp://one',
                ),
                SiteStreamConfigItem(
                    stream_name='Cam1',
                    video_url='rtsp://two',
                ),
            ],
        )

        with self.assertRaises(HTTPException) as ctx:
            await streams.endpoint_put_site_stream_config(
                1,
                payload,
                self.db,
                self.current_user,
            )

        self.assertEqual(ctx.exception.status_code, 400)

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
        """Test endpoint update stream config.

        Args:
            mock_is_super_admin: Value used by this callable.
            mock_update: Value used by this callable.
        """
        cfg_mock: MagicMock = MagicMock(site=self.site_mock, stream_name='old')
        cfg_mock.group_id = 1
        self.db.get.return_value = cfg_mock
        self.db.scalar.return_value = None

        payload: StreamConfigUpdate = StreamConfigUpdate(stream_name='new')

        response = await streams.endpoint_update_stream_config(
            1,
            payload,
            self.db,
            self.current_user,
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
        """Test endpoint update stream config name conflict.

        Args:
            mock_is_super_admin: Value used by this callable.
        """
        cfg_mock: MagicMock = MagicMock(site=self.site_mock, stream_name='old')
        cfg_mock.group_id = 1
        self.db.get.return_value = cfg_mock
        self.db.scalar.return_value = MagicMock()

        payload: StreamConfigUpdate = StreamConfigUpdate(
            stream_name='conflict',
        )

        with self.assertRaises(HTTPException) as ctx:
            await streams.endpoint_update_stream_config(
                1,
                payload,
                self.db,
                self.current_user,
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
        """Test endpoint delete stream config.

        Args:
            mock_is_super_admin: Value used by this callable.
            mock_delete: Value used by this callable.
        """
        cfg_mock: MagicMock = MagicMock(site=self.site_mock)
        cfg_mock.group_id = 1
        self.db.get.return_value = cfg_mock

        response = await streams.endpoint_delete_stream_config(
            1,
            self.db,
            self.current_user,
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
        """Test endpoint group stream limit.

        Args:
            mock_is_super_admin: Value used by this callable.
            mock_limit: Value used by this callable.
        """
        mock_limit.return_value = (3, 10)
        self.current_user.role = 'admin'
        self.current_user.group_id = 1

        response = await streams.endpoint_group_stream_limit(
            1,
            self.db,
            self.current_user,
        )

        self.assertEqual(response['max_allowed_streams'], 10)
        self.assertEqual(response['current_streams_count'], 3)

    async def test_list_stream_configs_site_not_found(self) -> None:
        """Should raise 404 if site not found.

        This test ensures the endpoint returns HTTP 404 if the site does not
        exist.
        """
        self.db.get.return_value = None
        with self.assertRaises(HTTPException) as ctx:
            await streams.endpoint_list_stream_configs(
                1,
                self.db,
                self.current_user,
            )
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_create_stream_config_site_not_found(self) -> None:
        """Should raise 404 if site not found when creating config.

        This test ensures the endpoint returns HTTP 404 if the site does not
        exist when creating a config.
        """
        self.db.get.return_value = None
        payload: StreamConfigCreate = StreamConfigCreate(
            site_id=1,
            stream_name='stream',
            video_url='url',
        )
        with self.assertRaises(HTTPException) as ctx:
            await streams.endpoint_create_stream_config(
                payload,
                self.db,
                self.current_user,
            )
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_update_stream_config_not_found(self) -> None:
        """Should raise 404 if config not found when updating.

        This test ensures the endpoint returns HTTP 404 if the config does not
        exist when updating.
        """
        self.db.get.return_value = None
        payload: StreamConfigUpdate = StreamConfigUpdate(stream_name='new')
        with self.assertRaises(HTTPException) as ctx:
            await streams.endpoint_update_stream_config(
                1,
                payload,
                self.db,
                self.current_user,
            )
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_update_stream_config_rejects_other_group_config(
        self,
    ) -> None:
        """Admin cannot update another group's stream on a shared site."""
        cfg_mock: MagicMock = MagicMock(site=self.site_mock, stream_name='old')
        cfg_mock.group_id = 2
        self.db.get.return_value = cfg_mock
        payload: StreamConfigUpdate = StreamConfigUpdate(stream_name='new')

        with self.assertRaises(HTTPException) as ctx:
            await streams.endpoint_update_stream_config(
                1,
                payload,
                self.db,
                self.current_user,
            )

        self.assertEqual(ctx.exception.status_code, 403)

    async def test_delete_stream_config_not_found(self) -> None:
        """Should raise 404 if config not found when deleting.

        This test ensures the endpoint returns HTTP 404 if the config does not
        exist when deleting.
        """
        self.db.get.return_value = None
        with self.assertRaises(HTTPException) as ctx:
            await streams.endpoint_delete_stream_config(
                1,
                self.db,
                self.current_user,
            )
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_delete_stream_config_rejects_other_group_config(
        self,
    ) -> None:
        """Admin cannot delete another group's stream on a shared site."""
        cfg_mock: MagicMock = MagicMock(site=self.site_mock)
        cfg_mock.group_id = 2
        self.db.get.return_value = cfg_mock

        with self.assertRaises(HTTPException) as ctx:
            await streams.endpoint_delete_stream_config(
                1,
                self.db,
                self.current_user,
            )

        self.assertEqual(ctx.exception.status_code, 403)

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

        This test ensures the endpoint returns HTTP 403 if the user is neither
        super admin nor group admin.
        """
        self.current_user.role = 'user'
        self.current_user.group_id = 2
        with self.assertRaises(HTTPException) as ctx:
            await streams.endpoint_group_stream_limit(
                1,
                self.db,
                self.current_user,
            )
        self.assertEqual(ctx.exception.status_code, 403)


if __name__ == '__main__':
    unittest.main()


def _site(*group_ids: int) -> Site:
    """Perform site.

    Args:
        *group_ids: Value used by this callable.

    Returns:
        The callable result.
    """
    site = Site()
    site.groups = [Group(id=group_id) for group_id in group_ids]
    return site


class TestStreamRouterCoverage(unittest.IsolatedAsyncioTestCase):
    """Exercise stream ownership and batch upsert guardrails."""

    def setUp(self) -> None:
        """Perform setUp.
        """
        self.db = AsyncMock()
        self.db.add = MagicMock()
        self.admin = User(
            role='admin',
            group_id=1,
            username='admin',
            password_hash='unused-in-helper-tests',
        )
        self.site = _site(1, 2)

    def test_stream_group_helpers_reject_invalid_group_ownership(self) -> None:
        """A stream must have a site group and remain inside that site
        scope."""
        with self.assertRaisesRegex(HTTPException, 'must have a group'):
            stream_config_services._primary_site_group_id(_site())
        self.assertEqual(
            stream_config_services._primary_site_group_id(_site(2, 1)),
            1,
        )

        with patch(
            'examples.db_management.services.stream_config_services.'
            'is_super_admin',
            return_value=True,
        ):
            self.assertEqual(
                stream_config_services._resolve_stream_group_id(
                    self.site,
                    self.admin,
                    2,
                ),
                2,
            )
            with self.assertRaisesRegex(
                HTTPException,
                'not associated',
            ):
                stream_config_services._resolve_stream_group_id(
                    self.site,
                    self.admin,
                    99,
                )

    async def test_stream_name_uniqueness_supports_exclusion_and_conflicts(
        self,
    ) -> None:
        """Renaming excludes itself but still rejects an existing sibling
        name."""
        self.db.scalar = AsyncMock(return_value=None)
        await stream_config_services._ensure_stream_name_available(
            1,
            'Camera A',
            self.db,
            exclude_config_id=7,
        )
        self.assertIn(
            'stream_configs.id !=',
            str(self.db.scalar.await_args.args[0]),
        )

        self.db.scalar = AsyncMock(return_value=object())
        with self.assertRaisesRegex(HTTPException, 'already exists'):
            await stream_config_services._ensure_stream_name_available(
                1,
                'Camera A',
                self.db,
            )

    async def test_site_stream_alias_uses_the_shared_listing_logic(
        self,
    ) -> None:
        """The site-scoped GET is a thin alias of the listing helper."""
        with patch(
            'examples.db_management.routers.streams.'
            '_list_site_stream_config_reads',
            new_callable=AsyncMock,
            return_value=[],
        ) as list_reads:
            result = await streams.endpoint_get_site_stream_config(
                5,
                self.db,
                self.admin,
            )

        self.assertEqual(result, [])
        list_reads.assert_awaited_once_with(5, self.db, self.admin)

    async def test_site_upsert_rejects_limits_and_unknown_config_ids(
        self,
    ) -> None:
        """Batch upserts stop before creating beyond a quota or updating
        ghosts."""
        self.db.get = AsyncMock(return_value=self.site)
        new_stream = SiteStreamConfigUpsert(
            streams=[
                SiteStreamConfigItem(
                    stream_name='New',
                    video_url='rtsp://new',
                ),
            ],
        )

        with (
            patch(
                'examples.db_management.routers.streams.is_super_admin',
                return_value=False,
            ),
            patch(
                'examples.db_management.routers.streams.list_stream_configs',
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                'examples.db_management.routers.streams.'
                'get_group_stream_limit',
                new_callable=AsyncMock,
                return_value=(5, 5),
            ),
        ):
            with self.assertRaisesRegex(HTTPException, 'Stream limit reached'):
                await streams.endpoint_put_site_stream_config(
                    1,
                    new_stream,
                    self.db,
                    self.admin,
                )

        missing_stream = SiteStreamConfigUpsert(
            streams=[
                SiteStreamConfigItem(
                    id=999,
                    stream_name='Missing',
                    video_url='rtsp://missing',
                ),
            ],
        )
        with (
            patch(
                'examples.db_management.routers.streams.is_super_admin',
                return_value=False,
            ),
            patch(
                'examples.db_management.routers.streams.list_stream_configs',
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            with self.assertRaisesRegex(
                HTTPException,
                'not found',
            ):
                await streams.endpoint_put_site_stream_config(
                    1,
                    missing_stream,
                    self.db,
                    self.admin,
                )

    async def test_site_upsert_checks_name_before_updating_existing_stream(
        self,
    ) -> None:
        """Changing a saved camera name validates its replacement name
        first."""
        existing = SimpleNamespace(
            id=7,
            stream_name='Old',
            group_id=1,
            video_url='rtsp://old',
            model_key='yolo26n',
            recognition_enabled=True,
            work_start_hour=7,
            work_end_hour=18,
            detect_no_safety_vest_or_helmet=False,
            detect_near_machinery_or_vehicle=False,
            detect_in_restricted_area=False,
            detect_in_utility_pole_restricted_area=False,
            detect_machinery_close_to_pole=False,
            expire_date=None,
            updated_at=DateTime(2026, 7, 24),
        )
        self.db.get = AsyncMock(return_value=self.site)
        payload = SiteStreamConfigUpsert(
            streams=[
                SiteStreamConfigItem(
                    id=7,
                    stream_name='Renamed',
                    video_url='rtsp://renamed',
                ),
            ],
        )

        with (
            patch(
                'examples.db_management.routers.streams.is_super_admin',
                return_value=False,
            ),
            patch(
                'examples.db_management.routers.streams.list_stream_configs',
                new_callable=AsyncMock,
                return_value=[existing],
            ),
            patch(
                'examples.db_management.routers.streams.'
                '_list_site_stream_config_reads',
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await streams.endpoint_put_site_stream_config(
                1,
                payload,
                self.db,
                self.admin,
            )

        self.assertEqual(result, [])
        self.assertEqual(existing.stream_name, 'Renamed')
        self.assertEqual(existing.video_url, 'rtsp://renamed')
        self.db.flush.assert_awaited_once()
        self.db.commit.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
