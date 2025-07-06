from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.auth.models import Site
from examples.db_management.services import site_services


class TestSiteServices(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for site_services.py using unittest and mocks.
    """

    def setUp(self) -> None:
        """Initialise common mock objects for each test.

        This method sets up mock database and site objects for use in
        each test case.
        """
        self.db: AsyncMock = AsyncMock()
        self.site: MagicMock = MagicMock(spec=Site)
        self.site.id = 1  # type: ignore[attr-defined]
        self.site.name = 'Test Site'  # type: ignore[attr-defined]
        self.group_id: int = 10
        self.user_id: int = 20

    async def test_list_sites_without_group(self) -> None:
        """Test retrieving all sites when no group_id is provided.

        Ensures that all sites are returned if no group_id is
        specified.
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
        """Test retrieving sites filtered by group_id.

        Ensures that only sites belonging to the specified group_id are
        returned.
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

    async def test_create_site_success(self) -> None:
        """Test successful creation of a new site.

        Verifies that a new site is created and committed to the
        database without error.
        """
        self.db.commit = AsyncMock()
        self.db.refresh = AsyncMock()
        self.db.add = MagicMock()

        # Simulate super_admin query, user_sites_table.insert,
        # and refreshed_site query
        mock_admin_result: MagicMock = MagicMock()
        mock_admin_result.scalar_one_or_none.return_value = MagicMock(id=999)
        mock_insert_result: MagicMock = MagicMock()
        mock_refreshed_site_result: MagicMock = MagicMock()
        (
            mock_refreshed_site_result
            .unique.return_value
            .scalar_one.return_value
        ) = MagicMock()
        self.db.execute = AsyncMock(
            side_effect=[
                mock_admin_result,
                mock_insert_result,
                mock_refreshed_site_result,
            ],
        )

        result: MagicMock = await site_services.create_site(
            name='New Site',
            group_id=self.group_id,
            db=self.db,
        )
        expected = (
            mock_refreshed_site_result
            .unique.return_value
            .scalar_one.return_value
        )
        self.assertEqual(result, expected)

        self.db.add.assert_called()
        self.db.commit.assert_awaited()
        # create_site does not call refresh
        self.db.refresh.assert_not_called()

    async def test_create_site_exception(self) -> None:
        """Test handling exception during site creation.

        Ensures that an HTTPException is raised and rollback is called
        if the database commit fails during site creation.
        """
        self.db.commit = AsyncMock(side_effect=Exception('DB error'))
        self.db.rollback = AsyncMock()
        self.db.add = MagicMock()

        with self.assertRaises(HTTPException) as context:
            await site_services.create_site(
                name='Fail Site',
                group_id=self.group_id,
                db=self.db,
            )

        self.assertEqual(context.exception.status_code, 500)
        self.db.rollback.assert_awaited()

    async def test_update_site_success(self) -> None:
        """Test successful site name update.

        Verifies that the site name is updated and committed to the
        database.
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
        """Test handling exception during site update.

        Ensures that an HTTPException is raised and rollback is called
        if the database commit fails during site update.
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

    async def test_delete_site_success(self) -> None:
        """Test successful deletion of a site and related records.

        Verifies that the site and its related image records are deleted
        and the transaction is committed.
        """
        mock_execute_result: MagicMock = MagicMock()
        mock_execute_result.scalars.return_value.all.return_value = [
            'image1.png', 'image2.png',
        ]
        self.db.execute = AsyncMock(return_value=mock_execute_result)
        self.db.delete = AsyncMock()
        self.db.commit = AsyncMock()

        with patch('pathlib.Path.unlink') as mock_unlink:
            mock_unlink.return_value = None

            await site_services.delete_site(site=self.site, db=self.db)

            self.db.commit.assert_awaited()
            self.db.delete.assert_awaited_with(self.site)

    async def test_delete_site_exception(self) -> None:
        """Test handling exception during site deletion.

        Ensures that an HTTPException is raised and rollback is called
        if the database commit fails during site deletion.
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
        """Test adding a user to a site.

        Verifies that a user is added to a site and the transaction is
        committed.
        """
        self.db.execute = AsyncMock()
        self.db.commit = AsyncMock()

        await site_services.add_user_to_site(
            user_id=self.user_id,
            site_id=self.site.id,
            db=self.db,
        )

        self.db.commit.assert_awaited()

    async def test_remove_user_from_site(self) -> None:
        """Test removing a user from a site.

        Verifies that a user is removed from a site and the transaction
        is committed.
        """
        self.db.execute = AsyncMock()
        self.db.commit = AsyncMock()

        await site_services.remove_user_from_site(
            user_id=self.user_id,
            site_id=self.site.id,
            db=self.db,
        )

        self.db.commit.assert_awaited()

    async def test_create_site_without_group_id(self) -> None:
        """Test exception raised when creating site without a group_id.

        Ensures that an HTTPException with status 400 is raised if
        group_id is not provided.
        """
        with self.assertRaises(HTTPException) as context:
            await site_services.create_site(
                name='NoGroupSite',
                group_id=None,
                db=self.db,
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(
            context.exception.detail,
            'group_id is required for new site',
        )

    async def test_delete_site_removes_images(self) -> None:
        """Test file deletion during site deletion.

        Verifies that image files are deleted from the filesystem
        when a site is deleted.
        """
        mock_execute_result: MagicMock = MagicMock()
        mock_execute_result.scalars.return_value.all.return_value = [
            '/fake/path/image1.png',
        ]
        self.db.execute = AsyncMock(return_value=mock_execute_result)
        self.db.delete = AsyncMock()
        self.db.commit = AsyncMock()

        with patch('pathlib.Path.is_file', return_value=True) as mock_is_file:
            with patch('pathlib.Path.unlink') as mock_unlink:
                await site_services.delete_site(site=self.site, db=self.db)

                mock_is_file.assert_called_once()
                mock_unlink.assert_called_once_with(missing_ok=True)

        self.db.commit.assert_awaited()
        self.db.delete.assert_awaited_with(self.site)


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.db_management.services.site_services\
    --cov-report=term-missing\
        tests/examples/db_management/services/site_services_test.py
'''
