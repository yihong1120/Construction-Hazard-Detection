from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import User
from examples.db_management.deps import SUPER_ADMIN_NAME
from examples.db_management.routers.sites import endpoint_add_user_to_site
from examples.db_management.routers.sites import endpoint_create_site
from examples.db_management.routers.sites import endpoint_delete_site
from examples.db_management.routers.sites import endpoint_list_sites
from examples.db_management.routers.sites import endpoint_remove_user_from_site
from examples.db_management.routers.sites import endpoint_update_site
from examples.db_management.schemas.site import SiteCreate
from examples.db_management.schemas.site import SiteDelete
from examples.db_management.schemas.site import SiteUpdate
from examples.db_management.schemas.site import SiteUserOp


class TestSiteMgmtRouter(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for site management router endpoints.
    """

    def setUp(self) -> None:
        """Prepare common test mocks for each test case."""
        self.db: AsyncMock = AsyncMock(spec=AsyncSession)
        self.user: MagicMock = MagicMock(spec=User)
        self.user.role = 'admin'
        self.user.group_id = 1

    @patch('examples.db_management.routers.sites.list_sites')
    @patch(
        'examples.db_management.routers.sites.is_super_admin',
        return_value=True,
    )
    async def test_endpoint_list_sites_super_admin(
        self,
        mock_is_super_admin: MagicMock,
        mock_list_sites: MagicMock,
    ) -> None:
        """Test listing sites as a super admin.

        Args:
            mock_is_super_admin (MagicMock): Patched is_super_admin function.
            mock_list_sites (MagicMock): Patched list_sites function.
        """
        self.user.role = 'super_admin'
        mock_list_sites.return_value = []
        result = await endpoint_list_sites(self.db, self.user)
        self.assertEqual(result, [])

    async def test_endpoint_list_sites_permission_denied(self) -> None:
        """Ensure permission error for non-admin users."""
        self.user.role = 'user'
        with self.assertRaises(HTTPException) as ctx:
            await endpoint_list_sites(self.db, self.user)
        self.assertEqual(ctx.exception.status_code, 403)

    async def test_list_sites_forbidden(self) -> None:
        """Test list_sites forbidden for non-admin/non-super_Admin users."""
        self.user.role = 'user'
        with self.assertRaises(HTTPException) as ctx:
            await endpoint_list_sites(self.db, self.user)
        self.assertEqual(ctx.exception.status_code, 403)

    @patch('examples.db_management.routers.sites.create_site')
    async def test_endpoint_create_site_success(
        self, mock_create_site: MagicMock,
    ) -> None:
        """Test successful creation of a new site.

        Args:
            mock_create_site (MagicMock): Patched create_site function.
        """
        site = MagicMock()
        site.id = 1
        site.name = 'New Site'
        site.group_id = 1
        site.group = MagicMock()
        site.group.name = 'GroupName'
        site.users = [MagicMock(id=2)]
        mock_create_site.return_value = site
        payload = SiteCreate(name='New Site')
        result = await endpoint_create_site(payload, self.db, self.user)
        self.assertEqual(result.id, 1)
        self.assertEqual(result.name, 'New Site')
        self.assertEqual(result.group_name, 'GroupName')
        self.assertEqual(result.user_ids, [2])

    async def test_endpoint_create_site_permission_denied(self) -> None:
        """Ensure creating site in other groups is denied."""
        payload = SiteCreate(name='Site', group_id=2)
        with self.assertRaises(HTTPException) as ctx:
            await endpoint_create_site(payload, self.db, self.user)
        self.assertEqual(ctx.exception.status_code, 403)

    @patch('examples.db_management.routers.sites.update_site')
    async def test_endpoint_update_site_success(
        self, mock_update_site: MagicMock,
    ) -> None:
        """Test successful update of a site's name.

        Args:
            mock_update_site (MagicMock): Patched update_site function.
        """
        site = MagicMock()
        site.id = 1
        site.name = 'Site'
        site.group_id = 1
        site.group = MagicMock()
        site.group.name = 'GroupName'
        site.users = [MagicMock(id=2)]
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = site
        self.db.execute.return_value = mock_result
        payload = SiteUpdate(site_id=1, new_name='Updated Name')
        result = await endpoint_update_site(payload, self.db, self.user)
        self.assertEqual(result['message'], 'Site updated successfully.')

    async def test_update_site_not_found(self) -> None:
        """Test update_site when site not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        self.db.execute.return_value = mock_result
        payload = SiteUpdate(site_id=999, new_name='X')
        with self.assertRaises(HTTPException) as ctx:
            await endpoint_update_site(payload, self.db, self.user)
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_endpoint_delete_site_success(self) -> None:
        """Test successful deletion of a site."""
        site = MagicMock()
        site.id = 1
        site.name = 'Site'
        site.group_id = 1
        site.group = MagicMock()
        site.group.name = 'GroupName'
        site.users = [MagicMock(id=2)]
        mock_result = MagicMock()
        mock_result.unique.return_value = mock_result
        mock_result.scalar_one_or_none.return_value = site
        self.db.execute.return_value = mock_result

        # Mock redis object
        mock_redis = MagicMock()
        mock_redis.keys = AsyncMock(return_value=[b'key1', b'key2'])
        mock_redis.delete = AsyncMock()

        payload = SiteDelete(site_id=1)
        result = await endpoint_delete_site(
            payload,
            self.db,
            self.user,
            mock_redis,
        )
        self.assertEqual(
            result['message'],
            'Site and related data deleted successfully.',
        )
        mock_redis.keys.assert_called_once()
        mock_redis.delete.assert_called_once_with(b'key1', b'key2')

    async def test_delete_site_not_found(self) -> None:
        """Test delete_site when site not found."""
        mock_result = MagicMock()
        mock_result.unique.return_value = mock_result
        mock_result.scalar_one_or_none.return_value = None
        self.db.execute.return_value = mock_result
        payload = SiteDelete(site_id=999)
        with self.assertRaises(HTTPException) as ctx:
            await endpoint_delete_site(payload, self.db, self.user)
        self.assertEqual(ctx.exception.status_code, 404)

    @patch('examples.db_management.routers.sites.add_user_to_site')
    async def test_endpoint_add_user_to_site_success(
        self,
        mock_add_user: MagicMock,
    ) -> None:
        """Test successful addition of user to site.

        Args:
            mock_add_user (MagicMock): Patched add_user_to_site function.
        """
        site = MagicMock()
        site.group_id = 1
        user_to_add = MagicMock()
        user_to_add.username = 'testuser'
        user_to_add.group_id = 1
        mock_result = MagicMock()
        mock_result.unique.return_value = mock_result
        # Two calls: first returns site, second returns user_to_add
        mock_result.scalar_one_or_none.side_effect = [site, user_to_add]
        self.db.execute.return_value = mock_result
        payload = SiteUserOp(site_id=1, user_id=2)
        result = await endpoint_add_user_to_site(payload, self.db, self.user)
        self.assertEqual(
            result['message'],
            'User linked to site successfully.',
        )

    async def test_add_user_to_site_site_not_found(self) -> None:
        """Test add_user_to_site when site not found."""
        mock_result = MagicMock()
        mock_result.unique.return_value = mock_result
        mock_result.scalar_one_or_none.side_effect = [None]
        self.db.execute.return_value = mock_result
        payload = SiteUserOp(site_id=999, user_id=2)
        with self.assertRaises(HTTPException) as ctx:
            await endpoint_add_user_to_site(payload, self.db, self.user)
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_add_user_to_site_user_not_found(self) -> None:
        """Test add_user_to_site when user not found."""
        site = MagicMock()
        site.group_id = 1
        mock_result = MagicMock()
        mock_result.unique.return_value = mock_result
        mock_result.scalar_one_or_none.side_effect = [site, None]
        self.db.execute.return_value = mock_result
        payload = SiteUserOp(site_id=1, user_id=999)
        with self.assertRaises(HTTPException) as ctx:
            await endpoint_add_user_to_site(payload, self.db, self.user)
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_add_user_to_site_super_admin_forbidden(self) -> None:
        """Test add_user_to_site forbidden for super admin user."""
        site = MagicMock()
        site.group_id = 1
        user_to_add = MagicMock()
        user_to_add.username = SUPER_ADMIN_NAME
        user_to_add.group_id = 1
        mock_result = MagicMock()
        mock_result.unique.return_value = mock_result
        mock_result.scalar_one_or_none.side_effect = [site, user_to_add]
        self.db.execute.return_value = mock_result
        payload = SiteUserOp(site_id=1, user_id=2)
        with self.assertRaises(HTTPException) as ctx:
            await endpoint_add_user_to_site(payload, self.db, self.user)
        self.assertEqual(ctx.exception.status_code, 403)

    async def test_add_user_to_site_group_mismatch(self) -> None:
        """Test add_user_to_site forbidden for group mismatch."""
        site = MagicMock()
        site.group_id = 1
        user_to_add = MagicMock()
        user_to_add.username = 'testuser'
        user_to_add.group_id = 2
        mock_result = MagicMock()
        mock_result.unique.return_value = mock_result
        mock_result.scalar_one_or_none.side_effect = [site, user_to_add]
        self.db.execute.return_value = mock_result
        payload = SiteUserOp(site_id=1, user_id=2)
        with self.assertRaises(HTTPException) as ctx:
            await endpoint_add_user_to_site(payload, self.db, self.user)
        self.assertEqual(ctx.exception.status_code, 403)

    @patch(
        'examples.db_management.routers.sites.is_super_admin',
        return_value=False,
    )
    async def test_endpoint_remove_user_from_site_super_admin(
        self, mock_is_super_admin: MagicMock,
    ) -> None:
        """Ensure super admin cannot be removed from a site.

        Args:
            mock_is_super_admin (MagicMock): Patched is_super_admin function.
        """
        site = MagicMock()
        site.group_id = 1
        super_admin_user = MagicMock()
        super_admin_user.role = 'super_admin'
        super_admin_user.username = SUPER_ADMIN_NAME
        super_admin_user.group_id = 1
        site.users = [super_admin_user]
        user_to_remove = super_admin_user
        mock_result = MagicMock()
        mock_result.unique.return_value = mock_result
        mock_result.scalar_one_or_none.side_effect = [site, user_to_remove]
        self.db.execute.return_value = mock_result
        payload = SiteUserOp(site_id=1, user_id=2)
        with self.assertRaises(HTTPException) as ctx:
            await endpoint_remove_user_from_site(payload, self.db, self.user)
        self.assertEqual(ctx.exception.status_code, 403)

    async def test_remove_user_from_site_site_not_found(self) -> None:
        """Test remove_user_from_site when site not found."""
        mock_result = MagicMock()
        mock_result.unique.return_value = mock_result
        mock_result.scalar_one_or_none.side_effect = [None]
        self.db.execute.return_value = mock_result
        payload = SiteUserOp(site_id=999, user_id=2)
        with self.assertRaises(HTTPException) as ctx:
            await endpoint_remove_user_from_site(payload, self.db, self.user)
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_remove_user_from_site_user_not_found(self) -> None:
        """Test remove_user_from_site when user not found."""
        site = MagicMock()
        site.group_id = 1
        mock_result = MagicMock()
        mock_result.unique.return_value = mock_result
        mock_result.scalar_one_or_none.side_effect = [site, None]
        self.db.execute.return_value = mock_result
        payload = SiteUserOp(site_id=1, user_id=999)
        with self.assertRaises(HTTPException) as ctx:
            await endpoint_remove_user_from_site(payload, self.db, self.user)
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_remove_user_from_site_super_admin_forbidden(self) -> None:
        """Test remove_user_from_site forbidden for super admin user."""
        site = MagicMock()
        site.group_id = 1
        user_to_remove = MagicMock()
        user_to_remove.username = SUPER_ADMIN_NAME
        user_to_remove.group_id = 1
        mock_result = MagicMock()
        mock_result.unique.return_value = mock_result
        mock_result.scalar_one_or_none.side_effect = [site, user_to_remove]
        self.db.execute.return_value = mock_result
        payload = SiteUserOp(site_id=1, user_id=2)
        with self.assertRaises(HTTPException) as ctx:
            await endpoint_remove_user_from_site(payload, self.db, self.user)
        self.assertEqual(ctx.exception.status_code, 403)

    @patch('examples.db_management.routers.sites.list_sites')
    async def test_endpoint_list_sites_admin(
        self, mock_list_sites: MagicMock,
    ) -> None:
        """Test listing sites as an admin (group-specific)."""
        self.user.role = 'admin'
        self.user.group_id = 42
        mock_list_sites.return_value = []
        result = await endpoint_list_sites(self.db, self.user)
        self.assertEqual(result, [])
        mock_list_sites.assert_called_once_with(self.db, group_id=42)

    @patch('examples.db_management.routers.sites.remove_user_from_site')
    async def test_endpoint_remove_user_from_site_success(
        self, mock_remove_user: MagicMock,
    ) -> None:
        """Test successful removal of a user from a site.

        Args:
            mock_remove_user (MagicMock):
                Patched remove_user_from_site function.
        """
        site = MagicMock()
        site.id = 1
        site.group_id = 1
        user_to_remove = MagicMock()
        user_to_remove.id = 2
        user_to_remove.username = 'normal_user'
        user_to_remove.group_id = 1
        mock_result = MagicMock()
        mock_result.unique.return_value = mock_result
        mock_result.scalar_one_or_none.side_effect = [site, user_to_remove]
        self.db.execute.return_value = mock_result
        payload = SiteUserOp(site_id=1, user_id=2)
        result = await endpoint_remove_user_from_site(
            payload,
            self.db,
            self.user,
        )
        self.assertEqual(
            result['message'],
            'User unlinked from site successfully.',
        )
        mock_remove_user.assert_called_once_with(
            user_to_remove.id, site.id, self.db,
        )


if __name__ == '__main__':
    unittest.main()


'''
pytest --cov=examples.db_management.routers.sites\
    --cov-report=term-missing\
        tests/examples/db_management/routers/sites_test.py
'''
