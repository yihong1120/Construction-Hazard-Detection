from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.auth.models import Site
from examples.db_management.services import site_services


class TestSiteServices(unittest.IsolatedAsyncioTestCase):
    """Unit tests for site_services.py using unittest and mocks.

    All tests utilise asynchronous mocks to simulate database interactions.
    """

    def setUp(self) -> None:
        """Initialise common mock objects for each test."""
        self.db: AsyncMock = AsyncMock()
        self.site: MagicMock = MagicMock(spec=Site)
        self.site.id = 1
        self.site.name = 'Test Site'
        self.group_id: int = 10
        self.user_id: int = 20

    async def test_list_sites_without_group(self) -> None:
        """Test retrieving all sites when no group_id is provided."""
        mock_result = MagicMock()
        scalars_mock = mock_result.unique.return_value.scalars.return_value
        scalars_mock.all.return_value = ['site1', 'site2']

        self.db.execute = AsyncMock(return_value=mock_result)

        sites = await site_services.list_sites(db=self.db)

        self.assertEqual(sites, ['site1', 'site2'])

    async def test_list_sites_with_group(self) -> None:
        """Test retrieving sites filtered by group_id."""
        mock_result = MagicMock()
        scalars_mock = mock_result.unique.return_value.scalars.return_value
        scalars_mock.all.return_value = ['site3']

        self.db.execute = AsyncMock(return_value=mock_result)

        sites = await site_services.list_sites(
            db=self.db,
            group_id=self.group_id,
        )

        self.assertEqual(sites, ['site3'])

    async def test_create_site_success(self) -> None:
        """Test successful creation of a new site."""
        self.db.commit = AsyncMock()
        self.db.refresh = AsyncMock()
        self.db.add = MagicMock()

        mock_admin_result = MagicMock()
        mock_admin_result.scalar_one_or_none.return_value = MagicMock(id=999)
        self.db.execute = AsyncMock(return_value=mock_admin_result)

        with patch(
            'examples.db_management.services.site_services.Site',
        ) as MockSite:
            mock_site = MagicMock()
            MockSite.return_value = mock_site

            result = await site_services.create_site(
                name='New Site',
                group_id=self.group_id,
                db=self.db,
            )

            self.assertEqual(result, mock_site)
            self.db.add.assert_called_with(mock_site)
            self.db.commit.assert_awaited()
            self.db.refresh.assert_awaited_with(mock_site)

    async def test_create_site_exception(self) -> None:
        """Test handling exception during site creation."""
        self.db.commit = AsyncMock(side_effect=Exception('DB error'))
        self.db.rollback = AsyncMock()
        self.db.add = MagicMock()  # 改為 MagicMock 即可解決警告

        with self.assertRaises(HTTPException) as context:
            await site_services.create_site(
                name='Fail Site',
                group_id=self.group_id,
                db=self.db,
            )

        self.assertEqual(context.exception.status_code, 500)
        self.db.rollback.assert_awaited()

    async def test_update_site_success(self) -> None:
        """Test successful site name update."""
        self.db.commit = AsyncMock()

        await site_services.update_site(
            site=self.site,
            new_name='Updated Site',
            db=self.db,
        )

        self.db.commit.assert_awaited()
        self.assertEqual(self.site.name, 'Updated Site')

    async def test_update_site_exception(self) -> None:
        """Test handling exception during site update."""
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
        """Test successful deletion of a site and related records."""
        mock_execute_result = MagicMock()
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
        """Test handling exception during site deletion."""
        mock_execute_result = MagicMock()
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
        """Test adding a user to a site."""
        self.db.execute = AsyncMock()
        self.db.commit = AsyncMock()

        await site_services.add_user_to_site(
            user_id=self.user_id,
            site_id=self.site.id,
            db=self.db,
        )

        self.db.commit.assert_awaited()

    async def test_remove_user_from_site(self) -> None:
        """Test removing a user from a site."""
        self.db.execute = AsyncMock()
        self.db.commit = AsyncMock()

        await site_services.remove_user_from_site(
            user_id=self.user_id,
            site_id=self.site.id,
            db=self.db,
        )

        self.db.commit.assert_awaited()

    async def test_create_site_without_group_id(self) -> None:
        """Test exception raised when creating site without a group_id."""
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
        """Test file deletion during site deletion."""
        mock_execute_result = MagicMock()
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
