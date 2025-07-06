from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.auth.models import Site
from examples.auth.models import User
from examples.db_management import deps


class TestDeps(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for authorisation dependencies in deps.py.
    """

    async def asyncSetUp(self) -> None:
        """
        Set up test users and sites for each test case.
        """
        self.admin_user: User = User(
            id=1,
            username='admin_user',
            role='admin',
            group_id=100,
        )
        self.super_admin_user: User = User(
            id=2,
            username='ChangDar',
            role='admin',
            group_id=None,
        )
        self.normal_user: User = User(
            id=3,
            username='user',
            role='user',
            group_id=None,
        )
        self.site_in_group: Site = Site(id=1, name='Test Site', group_id=100)
        self.site_other_group: Site = Site(
            id=2, name='Other Site', group_id=999,
        )

    @patch('examples.db_management.deps.jwt_access', new_callable=AsyncMock)
    @patch('examples.db_management.deps.get_db', new_callable=AsyncMock)
    async def test_get_current_user_success(
        self,
        mock_get_db: AsyncMock,
        mock_jwt_access: AsyncMock,
    ) -> None:
        """
        Test that get_current_user returns a valid user.
        """
        # Correctly mock SQLAlchemy execute chain
        result_mock: MagicMock = MagicMock()
        unique_mock: MagicMock = MagicMock()
        unique_mock.scalar_one_or_none.return_value = self.admin_user
        result_mock.unique.return_value = unique_mock
        db_mock: MagicMock = MagicMock()
        db_mock.execute = AsyncMock(return_value=result_mock)
        mock_get_db.return_value = db_mock
        mock_jwt_access.return_value.subject = {'username': 'admin_user'}

        user: User = await deps.get_current_user(
            credentials=await mock_jwt_access(), db=await mock_get_db(),
        )
        self.assertEqual(user.username, 'admin_user')

    @patch('examples.db_management.deps.jwt_access', new_callable=AsyncMock)
    @patch('examples.db_management.deps.get_db', new_callable=AsyncMock)
    async def test_get_current_user_missing_subject(
        self,
        mock_get_db: AsyncMock,
        mock_jwt_access: AsyncMock,
    ) -> None:
        """
        Test get_current_user raises 401 if JWT subject is missing username.
        """
        mock_jwt_access.return_value.subject = {}
        db_mock: MagicMock = MagicMock()
        mock_get_db.return_value = db_mock
        with self.assertRaises(HTTPException) as ctx:
            await deps.get_current_user(
                credentials=await mock_jwt_access(),
                db=await mock_get_db(),
            )
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.detail, 'Invalid token subject')

    @patch('examples.db_management.deps.jwt_access', new_callable=AsyncMock)
    @patch('examples.db_management.deps.get_db', new_callable=AsyncMock)
    async def test_get_current_user_user_not_found(
        self,
        mock_get_db: AsyncMock,
        mock_jwt_access: AsyncMock,
    ) -> None:
        """
        Test get_current_user raises 401 if user not found in DB.
        """
        mock_jwt_access.return_value.subject = {'username': 'ghost'}
        # Mock DB chain to return None
        result_mock: MagicMock = MagicMock()
        unique_mock: MagicMock = MagicMock()
        unique_mock.scalar_one_or_none.return_value = None
        result_mock.unique.return_value = unique_mock
        db_mock: MagicMock = MagicMock()
        db_mock.execute = AsyncMock(return_value=result_mock)
        mock_get_db.return_value = db_mock
        with self.assertRaises(HTTPException) as ctx:
            await deps.get_current_user(
                credentials=await mock_jwt_access(),
                db=await mock_get_db(),
            )
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.detail, 'User not found')

    async def test_is_super_admin_true(self) -> None:
        """
        Test is_super_admin returns True for super admin.
        """
        self.assertTrue(deps.is_super_admin(self.super_admin_user))

    async def test_is_super_admin_false(self) -> None:
        """
        Test is_super_admin returns False for non-super admin.
        """
        self.assertFalse(deps.is_super_admin(self.admin_user))

    def test_require_admin_success(self) -> None:
        """
        Test require_admin allows admin users.
        """
        result: User = deps.require_admin(self.admin_user)
        self.assertEqual(result, self.admin_user)

    def test_require_admin_rejects_non_admin(self) -> None:
        """
        Test require_admin rejects non-admin users.
        """
        with self.assertRaises(HTTPException) as ctx:
            deps.require_admin(self.normal_user)
        self.assertEqual(ctx.exception.status_code, 403)

    def test_require_super_admin_success(self) -> None:
        """
        Test require_super_admin passes for super admin.
        """
        result: User = deps.require_super_admin(self.super_admin_user)
        self.assertEqual(result, self.super_admin_user)

    def test_require_super_admin_rejects_admin(self) -> None:
        """
        Test require_super_admin rejects normal admins.
        """
        with self.assertRaises(HTTPException) as ctx:
            deps.require_super_admin(self.admin_user)
        self.assertEqual(ctx.exception.status_code, 403)

    def test_ensure_not_super_pass(self) -> None:
        """
        Test ensure_not_super passes for non-super users.
        """
        deps.ensure_not_super(self.admin_user)  # Should not raise

    def test_ensure_not_super_fails_on_super(self) -> None:
        """
        Test ensure_not_super fails when targeting super admin.
        """
        with self.assertRaises(HTTPException) as ctx:
            deps.ensure_not_super(self.super_admin_user)
        self.assertEqual(ctx.exception.status_code, 403)

    def test_ensure_admin_with_group_pass(self) -> None:
        """
        Test ensure_admin_with_group allows valid admin with group.
        """
        deps.ensure_admin_with_group(self.admin_user)  # Should not raise

    def test_ensure_admin_with_group_fails_on_role(self) -> None:
        """
        Test ensure_admin_with_group rejects non-admin users.
        """
        with self.assertRaises(HTTPException) as ctx:
            deps.ensure_admin_with_group(self.normal_user)
        self.assertEqual(ctx.exception.status_code, 403)

    def test_ensure_admin_with_group_fails_on_group(self) -> None:
        """
        Test ensure_admin_with_group rejects admin with no group.
        """
        no_group_admin: User = User(
            id=4, username='admin2', role='admin', group_id=None,
        )
        with self.assertRaises(HTTPException) as ctx:
            deps.ensure_admin_with_group(no_group_admin)
        self.assertEqual(ctx.exception.status_code, 403)

    def test_site_permission_super_admin(self) -> None:
        """
        Test site permission allows super admin for any site or group.
        """
        deps._site_permission(
            self.super_admin_user,
            site=self.site_other_group,
        )
        deps._site_permission(
            self.super_admin_user,
            group_id=123,
        )  # Should not raise

    def test_site_permission_valid_admin_on_own_group(self) -> None:
        """
        Test admin can access site in own group.
        """
        deps._site_permission(self.admin_user, site=self.site_in_group)

    def test_site_permission_rejects_user(self) -> None:
        """
        Test non-admin is denied site operation.
        """
        with self.assertRaises(HTTPException) as ctx:
            deps._site_permission(self.normal_user, site=self.site_in_group)
        self.assertEqual(ctx.exception.status_code, 403)

    def test_site_permission_rejects_admin_no_group(self) -> None:
        """
        Test admin with no group is rejected.
        """
        no_group_admin: User = User(
            id=4, username='admin2', role='admin', group_id=None,
        )
        with self.assertRaises(HTTPException) as ctx:
            deps._site_permission(no_group_admin, site=self.site_in_group)
        self.assertEqual(ctx.exception.status_code, 403)

    def test_site_permission_rejects_cross_group_site(self) -> None:
        """
        Test admin cannot operate on site from another group.
        """
        with self.assertRaises(HTTPException) as ctx:
            deps._site_permission(self.admin_user, site=self.site_other_group)
        self.assertEqual(ctx.exception.status_code, 403)

    def test_site_permission_rejects_cross_group_id(self) -> None:
        """
        Test admin cannot operate on different group id.
        """
        with self.assertRaises(HTTPException) as ctx:
            deps._site_permission(self.admin_user, group_id=999)
        self.assertEqual(ctx.exception.status_code, 403)


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.db_management.deps\
    --cov-report=term-missing\
        tests/examples/db_management/deps_test.py
'''
