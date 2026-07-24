from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import User
from examples.db_management.deps import SUPER_ADMIN_NAME
from examples.db_management.routers.sites import _delete_matching_redis_keys
from examples.db_management.routers.sites import endpoint_add_group_to_site
from examples.db_management.routers.sites import endpoint_add_user_to_site
from examples.db_management.routers.sites import endpoint_create_site
from examples.db_management.routers.sites import endpoint_delete_site
from examples.db_management.routers.sites import endpoint_list_sites
from examples.db_management.routers.sites import endpoint_remove_group_from_site
from examples.db_management.routers.sites import endpoint_remove_user_from_site
from examples.db_management.routers.sites import endpoint_update_site
from examples.db_management.schemas.site import SiteCreate
from examples.db_management.schemas.site import SiteDelete
from examples.db_management.schemas.site import SiteGroupOp
from examples.db_management.schemas.site import SiteUpdate
from examples.db_management.schemas.site import SiteUserOp


class AsyncKeyIterator:
    """Small async iterator used to mock Redis SCAN results."""

    def __init__(self, keys: list[bytes]) -> None:
        """Support __init__."""
        self._keys = keys

    def __aiter__(self) -> AsyncKeyIterator:
        return self

    async def __anext__(self) -> bytes:
        if not self._keys:
            raise StopAsyncIteration
        return self._keys.pop(0)


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
        group_mock = MagicMock()
        group_mock.id = 1
        group_mock.name = 'GroupName'
        site.groups = [group_mock]
        user_mock = MagicMock(id=2)
        user_mock.group_id = 1
        site.users = [user_mock]
        mock_create_site.return_value = site
        payload = SiteCreate(name='New Site')
        result = await endpoint_create_site(payload, self.db, self.user)
        self.assertEqual(result.id, 1)
        self.assertEqual(result.name, 'New Site')
        self.assertEqual(result.group_names, ['GroupName'])
        self.assertEqual(result.user_ids, [2])

    async def test_endpoint_create_site_permission_denied(self) -> None:
        """Ensure creating site in other groups is denied."""
        payload = SiteCreate(name='Site', group_ids=[2])
        with self.assertRaises(HTTPException) as ctx:
            await endpoint_create_site(payload, self.db, self.user)
        self.assertEqual(ctx.exception.status_code, 403)

    @patch(
        'examples.db_management.routers.sites.is_super_admin',
        return_value=True,
    )
    @patch('examples.db_management.routers.sites.create_site')
    async def test_super_admin_creates_site_for_requested_groups(
        self,
        mock_create_site: MagicMock,
        _mock_is_super_admin: MagicMock,
    ) -> None:
        """Super administrators retain explicitly selected site groups."""
        site = MagicMock()
        site.id = 4
        site.name = 'Shared Site'
        site.groups = []
        site.users = []
        mock_create_site.return_value = site

        result = await endpoint_create_site(
            SiteCreate(name='Shared Site', group_ids=[1, 2]),
            self.db,
            self.user,
        )

        self.assertEqual(result.group_ids, [])
        mock_create_site.assert_awaited_once_with(
            'Shared Site', [1, 2], self.db,
        )

    @patch(
        'examples.db_management.routers.sites.'
        'refresh_site_notification_user_cache',
        new_callable=AsyncMock,
    )
    @patch('examples.db_management.routers.sites.update_site')
    async def test_endpoint_update_site_success(
        self,
        mock_update_site: MagicMock,
        mock_refresh_site_cache: AsyncMock,
    ) -> None:
        """Test successful update of a site's name.

        Args:
            mock_update_site (MagicMock): Patched update_site function.
        """
        site = MagicMock()
        site.id = 1
        site.name = 'Site'
        group_mock = MagicMock()
        group_mock.id = 1
        group_mock.name = 'GroupName'
        site.groups = [group_mock]
        site.users = [MagicMock(id=2)]
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = site
        self.db.execute.return_value = mock_result
        payload = SiteUpdate(site_id=1, new_name='Updated Name')
        mock_redis = AsyncMock()
        result = await endpoint_update_site(
            payload,
            self.db,
            self.user,
            mock_redis,
        )
        self.assertEqual(result['message'], 'Site updated successfully.')
        mock_redis.delete.assert_awaited_once_with(
            'site_notification_users:Site',
            'site_notification_users_ready:Site',
            'site_notification_users_lock:Site',
        )
        mock_refresh_site_cache.assert_awaited_once_with(
            'Updated Name',
            self.db,
            mock_redis,
        )

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
        group_mock = MagicMock()
        group_mock.id = 1
        group_mock.name = 'GroupName'
        site.groups = [group_mock]
        site.users = [MagicMock(id=2)]
        mock_result = MagicMock()
        mock_result.unique.return_value = mock_result
        mock_result.scalar_one_or_none.return_value = site
        self.db.execute.return_value = mock_result

        # Mock redis object
        mock_redis = MagicMock()
        mock_redis.scan_iter.return_value = AsyncKeyIterator(
            [b'key1', b'key2'],
        )
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
        mock_redis.scan_iter.assert_called_once()
        mock_redis.delete.assert_any_call(b'key1', b'key2')
        mock_redis.delete.assert_any_call(
            'site_notification_users:Site',
            'site_notification_users_ready:Site',
            'site_notification_users_lock:Site',
        )

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

    @patch(
        'examples.db_management.routers.sites.'
        'refresh_site_notification_user_cache',
        new_callable=AsyncMock,
    )
    @patch('examples.db_management.routers.sites.add_user_to_site')
    async def test_endpoint_add_user_to_site_success(
        self,
        mock_add_user: MagicMock,
        mock_refresh_site_cache: AsyncMock,
    ) -> None:
        """Test successful addition of user to site.

        Args:
            mock_add_user (MagicMock): Patched add_user_to_site function.
        """
        site = MagicMock()
        site.groups = [MagicMock(id=1)]
        user_to_add = MagicMock()
        user_to_add.username = 'testuser'
        user_to_add.group_id = 1
        mock_result = MagicMock()
        mock_result.unique.return_value = mock_result
        # Two calls: first returns site, second returns user_to_add
        mock_result.scalar_one_or_none.side_effect = [site, user_to_add]
        self.db.execute.return_value = mock_result
        payload = SiteUserOp(site_id=1, user_id=2)
        mock_redis = AsyncMock()
        site.name = 'Site One'
        result = await endpoint_add_user_to_site(
            payload,
            self.db,
            self.user,
            mock_redis,
        )
        self.assertEqual(
            result['message'],
            'User linked to site successfully.',
        )
        mock_refresh_site_cache.assert_awaited_once_with(
            'Site One',
            self.db,
            mock_redis,
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
        site.groups = [MagicMock(id=1)]
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
        site.groups = [MagicMock(id=1)]
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
        site.groups = [MagicMock(id=1)]
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

    async def test_add_user_to_site_rejects_user_without_group(self) -> None:
        """Group-scoped admins cannot attach an ungrouped user to a site."""
        site = MagicMock(groups=[MagicMock(id=1)])
        user_to_add = MagicMock(username='ungrouped', group_id=None)
        result = MagicMock()
        result.unique.return_value = result
        result.scalar_one_or_none.side_effect = [site, user_to_add]
        self.db.execute.return_value = result

        with self.assertRaises(HTTPException) as raised:
            await endpoint_add_user_to_site(
                SiteUserOp(site_id=1, user_id=2),
                self.db,
                self.user,
            )

        self.assertEqual(raised.exception.status_code, 403)

    @patch(
        'examples.db_management.routers.sites.is_super_admin',
        return_value=True,
    )
    async def test_super_admin_rejects_user_outside_site_groups(
        self,
        _mock_is_super_admin: MagicMock,
    ) -> None:
        """Even super admins cannot link a user to a non-member site group."""
        self.user.username = SUPER_ADMIN_NAME
        self.user.role = 'admin'
        site = MagicMock()
        site.groups = [MagicMock(id=1)]
        user_to_add = MagicMock()
        user_to_add.username = 'other-group'
        user_to_add.group_id = 2
        result = MagicMock()
        result.unique.return_value = result
        result.scalar_one_or_none.side_effect = [site, user_to_add]
        self.db.execute.return_value = result

        with self.assertRaises(HTTPException) as raised:
            await endpoint_add_user_to_site(
                SiteUserOp(site_id=1, user_id=2),
                self.db,
                self.user,
            )

        self.assertEqual(raised.exception.status_code, 403)

    async def test_add_user_to_shared_site_rejects_other_group_user(
        self,
    ) -> None:
        """Admin cannot add another group's user to a shared site."""
        site = MagicMock()
        site.groups = [MagicMock(id=1), MagicMock(id=2)]
        user_to_add = MagicMock()
        user_to_add.username = 'other-group-user'
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
        site.groups = [MagicMock(id=1)]
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
        site.groups = [MagicMock(id=1)]
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
        site.groups = [MagicMock(id=1)]
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

    async def test_remove_user_from_site_rejects_user_without_group(self) -> None:
        """Group-scoped admins cannot remove an ungrouped user from a site."""
        site = MagicMock(groups=[MagicMock(id=1)])
        user_to_remove = MagicMock(username='ungrouped', group_id=None)
        result = MagicMock()
        result.unique.return_value = result
        result.scalar_one_or_none.side_effect = [site, user_to_remove]
        self.db.execute.return_value = result

        with self.assertRaises(HTTPException) as raised:
            await endpoint_remove_user_from_site(
                SiteUserOp(site_id=1, user_id=2),
                self.db,
                self.user,
            )

        self.assertEqual(raised.exception.status_code, 403)

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

    @patch('examples.db_management.routers.sites.list_sites')
    async def test_endpoint_list_sites_admin_filters_shared_site_response(
        self,
        mock_list_sites: MagicMock,
    ) -> None:
        """Admin response hides other groups/users on shared sites."""
        self.user.role = 'admin'
        self.user.group_id = 42
        own_group = MagicMock()
        own_group.id = 42
        own_group.name = 'Own Group'
        other_group = MagicMock()
        other_group.id = 99
        other_group.name = 'Other Group'
        own_user = MagicMock()
        own_user.id = 7
        own_user.group_id = 42
        other_user = MagicMock()
        other_user.id = 8
        other_user.group_id = 99
        site = MagicMock()
        site.id = 3
        site.name = 'Shared Site'
        site.groups = [own_group, other_group]
        site.users = [own_user, other_user]
        mock_list_sites.return_value = [site]

        result = await endpoint_list_sites(self.db, self.user)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].group_ids, [42])
        self.assertEqual(result[0].group_names, ['Own Group'])
        self.assertEqual(result[0].user_ids, [7])

    @patch(
        'examples.db_management.routers.sites.'
        'refresh_site_notification_user_cache',
        new_callable=AsyncMock,
    )
    @patch('examples.db_management.routers.sites.remove_user_from_site')
    async def test_endpoint_remove_user_from_site_success(
        self,
        mock_remove_user: MagicMock,
        mock_refresh_site_cache: AsyncMock,
    ) -> None:
        """Test successful removal of a user from a site.

        Args:
            mock_remove_user (MagicMock):
                Patched remove_user_from_site function.
        """
        site = MagicMock()
        site.id = 1
        site.groups = [MagicMock(id=1)]
        user_to_remove = MagicMock()
        user_to_remove.id = 2
        user_to_remove.username = 'normal_user'
        user_to_remove.group_id = 1
        mock_result = MagicMock()
        mock_result.unique.return_value = mock_result
        mock_result.scalar_one_or_none.side_effect = [site, user_to_remove]
        self.db.execute.return_value = mock_result
        payload = SiteUserOp(site_id=1, user_id=2)
        site.name = 'Site One'
        mock_redis = AsyncMock()
        result = await endpoint_remove_user_from_site(
            payload,
            self.db,
            self.user,
            mock_redis,
        )
        self.assertEqual(
            result['message'],
            'User unlinked from site successfully.',
        )
        mock_remove_user.assert_called_once_with(
            user_to_remove.id, site.id, self.db,
        )
        mock_refresh_site_cache.assert_awaited_once_with(
            'Site One',
            self.db,
            mock_redis,
        )

    async def test_delete_matching_redis_keys_flushes_full_batches(self) -> None:
        """Redis SCAN deletes complete batches without a blocking KEYS call."""
        rds = MagicMock()
        rds.scan_iter.return_value = AsyncKeyIterator([b'first', b'second'])
        rds.delete = AsyncMock()

        await _delete_matching_redis_keys(rds, 'stream_metadata:*', batch_size=1)

        self.assertEqual(rds.delete.await_count, 2)
        rds.delete.assert_any_await(b'first')
        rds.delete.assert_any_await(b'second')

    @patch('examples.db_management.routers.sites.add_group_to_site')
    async def test_add_group_to_site_handles_success_and_missing_site(
        self,
        mock_add_group: MagicMock,
    ) -> None:
        """Group links use site permissions and distinguish a missing site."""
        site = MagicMock()
        site.id = 1
        site.groups = [MagicMock(id=1)]
        found_result = MagicMock()
        found_result.unique.return_value = found_result
        found_result.scalar_one_or_none.return_value = site
        self.db.execute.return_value = found_result

        result = await endpoint_add_group_to_site(
            SiteGroupOp(site_id=1, group_id=1),
            self.db,
            self.user,
        )

        self.assertEqual(
            result['message'],
            'Group linked to site successfully.',
        )
        mock_add_group.assert_awaited_once_with(1, 1, self.db)

        missing_result = MagicMock()
        missing_result.unique.return_value = missing_result
        missing_result.scalar_one_or_none.return_value = None
        self.db.execute.return_value = missing_result
        with self.assertRaises(HTTPException) as raised:
            await endpoint_add_group_to_site(
                SiteGroupOp(site_id=1, group_id=1),
                self.db,
                self.user,
            )
        self.assertEqual(raised.exception.status_code, 404)

    @patch('examples.db_management.routers.sites.remove_group_from_site')
    async def test_remove_group_from_site_handles_success_and_missing_site(
        self,
        mock_remove_group: MagicMock,
    ) -> None:
        """Group unlinks use site permissions and distinguish a missing site."""
        site = MagicMock()
        site.id = 1
        site.groups = [MagicMock(id=1)]
        found_result = MagicMock()
        found_result.unique.return_value = found_result
        found_result.scalar_one_or_none.return_value = site
        self.db.execute.return_value = found_result

        result = await endpoint_remove_group_from_site(
            SiteGroupOp(site_id=1, group_id=1),
            self.db,
            self.user,
        )

        self.assertEqual(
            result['message'],
            'Group unlinked from site successfully.',
        )
        mock_remove_group.assert_awaited_once_with(1, 1, self.db)

        missing_result = MagicMock()
        missing_result.unique.return_value = missing_result
        missing_result.scalar_one_or_none.return_value = None
        self.db.execute.return_value = missing_result
        with self.assertRaises(HTTPException) as raised:
            await endpoint_remove_group_from_site(
                SiteGroupOp(site_id=1, group_id=1),
                self.db,
                self.user,
            )
        self.assertEqual(raised.exception.status_code, 404)


if __name__ == '__main__':
    unittest.main()


'''
pytest --cov=examples.db_management.routers.sites\
    --cov-report=term-missing\
        tests/examples/db_management/routers/sites_test.py
'''
