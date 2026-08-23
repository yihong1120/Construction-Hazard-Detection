from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.auth.models import Site
from examples.db_management.services import site_services


class TestSiteServices(unittest.IsolatedAsyncioTestCase):
    """Unit tests for site_services.py using unittest and mocks."""

    def setUp(self) -> None:
        """Initialise common mock objects for each test.

        This method sets up mock database and site objects for use in each test
        case.
        """
        self.db: AsyncMock = AsyncMock()
        self.site: MagicMock = MagicMock(spec=Site)
        self.site.id = 1  # type: ignore[attr-defined]
        self.site.name = 'Test Site'  # type: ignore[attr-defined]
        self.group_id: int = 10
        self.user_id: int = 20

    async def test_list_sites_without_group(self) -> None:
        """Test list sites without group.
        """
        mock_result: MagicMock = MagicMock()
        scalars_mock: MagicMock = (
            mock_result.unique.return_value.scalars.return_value
        )
        scalars_mock.all.return_value = ['site1', 'site2']

        self.db.execute = AsyncMock(return_value=mock_result)

        sites: list = await site_services.list_sites(db=self.db)

        self.assertEqual(sites, ['site1', 'site2'])

    async def test_list_sites_with_group(self) -> None:
        """Test list sites with group.
        """
        mock_result: MagicMock = MagicMock()
        scalars_mock: MagicMock = (
            mock_result.unique.return_value.scalars.return_value
        )
        scalars_mock.all.return_value = ['site3']

        self.db.execute = AsyncMock(return_value=mock_result)

        sites: list = await site_services.list_sites(
            db=self.db,
            group_id=self.group_id,
        )

        self.assertEqual(sites, ['site3'])

    async def test_group_preference_helpers_handle_empty_and_chunked_inputs(
        self,
    ) -> None:
        """Preference seeding skips empties and flushes a full bulk chunk."""
        self.assertEqual(
            await site_services._list_user_ids_for_groups([], self.db),
            [],
        )
        self.db.execute = AsyncMock()
        await site_services.seed_site_notification_preferences(
            list(range(site_services._bulk_insert_chunk_size)),
            [self.site.id],
            self.db,
        )
        self.db.execute.assert_awaited_once()

    async def test_list_site_ids_for_group_returns_scalar_ids(self) -> None:
        """Group membership lookup returns the association-table site IDs."""
        result = MagicMock()
        result.scalars.return_value.all.return_value = [4, 9]
        self.db.execute = AsyncMock(return_value=result)

        site_ids = await site_services.list_site_ids_for_group(
            self.group_id,
            self.db,
        )

        self.assertEqual(site_ids, [4, 9])

    async def test_create_site_success(self) -> None:
        """Test create site success.
        """
        self.db.commit = AsyncMock()
        self.db.refresh = AsyncMock()
        self.db.add = MagicMock()

        # Simulate: bulk site_groups insert, group user query,
        # super_admin query, user_sites insert,
        # bulk pref insert for super_admin, and refreshed site query.
        mock_group_insert_result: MagicMock = MagicMock()
        mock_empty_users_result: MagicMock = MagicMock()
        mock_empty_users_result.scalars.return_value.all.return_value = []
        mock_admin_result: MagicMock = MagicMock()
        admin_scalar = mock_admin_result.unique.return_value.scalar_one_or_none
        admin_scalar.return_value = MagicMock(id=999)
        mock_insert_result: MagicMock = MagicMock()
        mock_pref_insert_result: MagicMock = MagicMock()
        mock_refreshed_site_result: MagicMock = MagicMock()
        refreshed_site_scalar = (
            mock_refreshed_site_result.unique.return_value.scalar_one
        )
        refreshed_site_scalar.return_value = MagicMock()
        self.db.execute = AsyncMock(
            side_effect=[
                mock_group_insert_result,  # site_groups insert
                mock_empty_users_result,  # user query for pref seeding
                mock_admin_result,  # super admin query
                mock_insert_result,  # user_sites insert
                mock_pref_insert_result,  # bulk pref insert for super_admin
                mock_refreshed_site_result,  # select refreshed site
            ],
        )

        result: MagicMock = await site_services.create_site(
            name='New Site',
            group_ids=[self.group_id],
            db=self.db,
        )
        expected = refreshed_site_scalar.return_value
        self.assertEqual(result, expected)

        self.db.add.assert_called()
        self.db.commit.assert_awaited()
        # create_site does not call refresh
        self.db.refresh.assert_not_called()

    async def test_create_site_exception(self) -> None:
        """Test create site exception.
        """
        self.db.commit = AsyncMock(side_effect=Exception('DB error'))
        self.db.rollback = AsyncMock()
        self.db.add = MagicMock()
        mock_empty_users_result: MagicMock = MagicMock()
        mock_empty_users_result.scalars.return_value.all.return_value = []
        mock_admin_result: MagicMock = MagicMock()
        admin_scalar = mock_admin_result.unique.return_value.scalar_one_or_none
        admin_scalar.return_value = None
        self.db.execute = AsyncMock(
            side_effect=[
                MagicMock(),  # bulk site_groups insert
                mock_empty_users_result,  # group user query
                mock_admin_result,  # super admin query
            ],
        )

        with self.assertRaises(HTTPException) as context:
            await site_services.create_site(
                name='Fail Site',
                group_ids=[self.group_id],
                db=self.db,
            )

        self.assertEqual(context.exception.status_code, 500)
        self.db.rollback.assert_awaited()

    async def test_update_site_success(self) -> None:
        """Test update site success.
        """
        self.db.commit = AsyncMock()

        await site_services.update_site(
            site=self.site,
            new_name='Updated Site',
            db=self.db,
        )

        self.db.commit.assert_awaited()
        self.assertEqual(self.site.name, 'Updated Site')

    async def test_update_site_exception(self) -> None:
        """Test update site exception.
        """
        self.db.commit = AsyncMock(side_effect=Exception('DB error'))
        self.db.rollback = AsyncMock()

        with self.assertRaises(HTTPException) as context:
            await site_services.update_site(
                site=self.site,
                new_name='Failed Update',
                db=self.db,
            )

        self.assertEqual(context.exception.status_code, 500)
        self.db.rollback.assert_awaited()

    @patch(
        'examples.db_management.services.site_services.'
        'enqueue_site_media_cleanup_for_site',
        new_callable=AsyncMock,
    )
    async def test_delete_site_success(
        self,
        enqueue_cleanup: AsyncMock,
    ) -> None:
        """Test delete site success.

        Args:
            enqueue_cleanup: Value used by this callable.
        """
        self.db.delete = AsyncMock()
        self.db.commit = AsyncMock()

        await site_services.delete_site(site=self.site, db=self.db)

        enqueue_cleanup.assert_awaited_once_with(
            self.site.name, self.db,
        )
        self.db.commit.assert_awaited()
        self.db.delete.assert_awaited_with(self.site)

    async def test_delete_site_exception(self) -> None:
        """Test delete site exception.
        """
        mock_execute_result: MagicMock = MagicMock()
        mock_execute_result.scalars.return_value.all.return_value = [
            'image1.png',
        ]
        self.db.execute = AsyncMock(return_value=mock_execute_result)
        self.db.delete = AsyncMock()
        self.db.commit = AsyncMock(side_effect=Exception('DB error'))
        self.db.rollback = AsyncMock()

        with patch('pathlib.Path.unlink') as mock_unlink:
            mock_unlink.return_value = None

            with self.assertRaises(HTTPException) as context:
                await site_services.delete_site(site=self.site, db=self.db)

            self.assertEqual(context.exception.status_code, 500)
            self.db.rollback.assert_awaited()

    async def test_add_user_to_site(self) -> None:
        """Test add user to site.
        """
        self.db.execute = AsyncMock()
        self.db.commit = AsyncMock()

        await site_services.add_user_to_site(
            user_id=self.user_id,
            site_id=self.site.id,
            db=self.db,
        )

        # Two execute calls: user_sites insert + bulk pref insert
        self.assertEqual(self.db.execute.await_count, 2)
        self.db.commit.assert_awaited()

    async def test_remove_user_from_site(self) -> None:
        # Simulate a user with no group → pref is deleted directly
        """Test remove user from site.
        """
        mock_user: MagicMock = MagicMock()
        mock_user.group_id = None
        self.db.get = AsyncMock(return_value=mock_user)
        self.db.execute = AsyncMock()
        self.db.commit = AsyncMock()

        await site_services.remove_user_from_site(
            user_id=self.user_id,
            site_id=self.site.id,
            db=self.db,
        )

        # Two execute calls: user_sites delete + pref delete
        self.assertEqual(self.db.execute.await_count, 2)
        self.db.commit.assert_awaited()

    async def test_create_site_without_group_id(self) -> None:
        """Test create site without group id.
        """
        with self.assertRaises(HTTPException) as context:
            await site_services.create_site(
                name='NoGroupSite',
                group_ids=[],
                db=self.db,
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(
            context.exception.detail,
            'group_id is required for new site',
        )

    @patch(
        'examples.db_management.services.site_services.'
        'enqueue_site_media_cleanup_for_site',
        new_callable=AsyncMock,
    )
    async def test_delete_site_queues_images_after_commit(
        self,
        enqueue_cleanup: AsyncMock,
    ) -> None:
        """Test image deletion is durably queued before the DB commit."""
        self.db.delete = AsyncMock()
        self.db.commit = AsyncMock()

        await site_services.delete_site(site=self.site, db=self.db)

        enqueue_cleanup.assert_awaited_once_with(
            self.site.name, self.db,
        )
        self.db.commit.assert_awaited()
        self.db.delete.assert_awaited_with(self.site)

    async def test_add_group_to_site_seeds_prefs(self) -> None:
        """add_group_to_site creates prefs for all users in the group."""
        mock_users_result: MagicMock = MagicMock()
        mock_users_result.scalars.return_value.all.return_value = [1, 2]
        self.db.execute = AsyncMock(
            side_effect=[
                MagicMock(),  # site_groups insert
                mock_users_result,  # select users in group
                MagicMock(),  # bulk pref insert
            ],
        )
        self.db.commit = AsyncMock()

        await site_services.add_group_to_site(
            site_id=self.site.id,
            group_id=self.group_id,
            db=self.db,
        )

        self.assertEqual(self.db.execute.await_count, 3)
        self.db.commit.assert_awaited()

    async def test_add_group_to_site_no_users(self) -> None:
        """add_group_to_site commits successfully when group has no users."""
        mock_users_result: MagicMock = MagicMock()
        mock_users_result.scalars.return_value.all.return_value = []
        self.db.execute = AsyncMock(
            side_effect=[
                MagicMock(),  # site_groups insert
                mock_users_result,  # select users → empty
            ],
        )
        self.db.commit = AsyncMock()

        await site_services.add_group_to_site(
            site_id=self.site.id,
            group_id=self.group_id,
            db=self.db,
        )

        self.assertEqual(self.db.execute.await_count, 2)
        self.db.commit.assert_awaited()

    async def test_remove_group_from_site_deletes_prefs(self) -> None:
        """remove_group_from_site deletes prefs for members without direct
        access."""
        self.db.execute = AsyncMock(
            side_effect=[
                MagicMock(),  # site_groups delete
                MagicMock(),  # conditional preference delete
            ],
        )
        self.db.commit = AsyncMock()

        await site_services.remove_group_from_site(
            site_id=self.site.id,
            group_id=self.group_id,
            db=self.db,
        )

        self.assertEqual(self.db.execute.await_count, 2)
        self.db.commit.assert_awaited()

    async def test_remove_group_from_site_no_users(self) -> None:
        """remove_group_from_site commits with no pref deletions when group is
        empty."""
        self.db.execute = AsyncMock(
            side_effect=[
                MagicMock(),  # site_groups delete
                MagicMock(),  # conditional preference delete
            ],
        )
        self.db.commit = AsyncMock()

        await site_services.remove_group_from_site(
            site_id=self.site.id,
            group_id=self.group_id,
            db=self.db,
        )

        self.assertEqual(self.db.execute.await_count, 2)
        self.db.commit.assert_awaited()

    async def test_remove_user_from_site_keeps_pref_when_group_access(
        self,
    ) -> None:
        """Keeps the pref when the user's group still owns the site."""
        mock_user: MagicMock = MagicMock()
        mock_user.group_id = 5  # user still in a group
        self.db.get = AsyncMock(return_value=mock_user)
        # site_groups query returns a non-None row → group still has access
        mock_group_row: MagicMock = MagicMock()
        mock_group_row.first.return_value = (1,)
        self.db.execute = AsyncMock(
            side_effect=[
                MagicMock(),  # user_sites delete
                mock_group_row,  # site_groups check → group still linked
            ],
        )
        self.db.commit = AsyncMock()

        await site_services.remove_user_from_site(
            user_id=self.user_id,
            site_id=self.site.id,
            db=self.db,
        )

        # Only 2 executes: user_sites delete + group access check;
        # NO pref delete because group still has the site
        self.assertEqual(self.db.execute.await_count, 2)
        self.db.commit.assert_awaited()


if __name__ == '__main__':
    unittest.main()
