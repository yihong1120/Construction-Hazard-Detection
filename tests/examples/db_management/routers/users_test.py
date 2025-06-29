from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from examples.db_management.routers import users
from examples.db_management.schemas.user import SetUserActiveStatus
from examples.db_management.schemas.user import UpdateMyPassword
from examples.db_management.schemas.user import UpdatePassword
from examples.db_management.schemas.user import UpdatePasswordById
from examples.db_management.schemas.user import UpdateUserGroup
from examples.db_management.schemas.user import UpdateUsername
from examples.db_management.schemas.user import UpdateUsernameById
from examples.db_management.schemas.user import UpdateUserRole
from examples.db_management.schemas.user import UserCreate
from examples.db_management.schemas.user import UserProfileUpdate
from examples.db_management.schemas.user import UserRead


class TestUsersRouter(unittest.IsolatedAsyncioTestCase):
    """Unit tests for user management router endpoints."""

    def setUp(self) -> None:
        """Set up common mock objects for tests."""
        self.db: AsyncMock = AsyncMock(spec=AsyncSession)
        self.current_user: MagicMock = MagicMock()
        self.current_user.check_password = AsyncMock(return_value=True)
        self.current_user.username = 'testuser'

    @patch('examples.db_management.routers.users.create_user')
    async def test_add_user(self, mock_create_user: AsyncMock) -> None:
        """Test adding a new user successfully."""
        # 構造一個完整的 user ORM mock
        mock_group = MagicMock()
        mock_group.name = 'group1'
        mock_group.uniform_number = 'G123'
        mock_profile = MagicMock()
        mock_profile.family_name = 'Family'
        mock_profile.middle_name = 'Middle'
        mock_profile.given_name = 'Given'
        mock_profile.email = 'test@example.com'
        mock_profile.mobile_number = '0912345678'
        mock_user = MagicMock(
            id=1,
            username='newuser',
            role='user',
            group=mock_group,
            profile=mock_profile,
        )
        mock_create_user.return_value = mock_user

        payload = UserCreate(username='newuser', password='pass', group_id=1)

        # db.execute 回傳一個 mock，其 scalar_one 是同步方法，回傳 mock_user
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = mock_user
        self.db.execute.return_value = mock_result

        response = await users.add_user(payload, self.db, self.current_user)

        self.assertIsInstance(response, UserRead)
        self.assertEqual(response.username, 'newuser')
        self.assertEqual(response.group.name, 'group1')
        self.assertEqual(response.profile.email, 'test@example.com')

    @patch('examples.db_management.routers.users.get_user_by_id')
    @patch('examples.db_management.routers.users.delete_user')
    async def test_remove_user(
        self,
        mock_delete_user: AsyncMock,
        mock_get_user_by_id: AsyncMock,
    ) -> None:
        """Test removing a user successfully."""
        mock_user = MagicMock()
        mock_get_user_by_id.return_value = mock_user

        payload = {'user_id': 1}
        response = await users.remove_user(payload, self.db)

        mock_delete_user.assert_awaited_with(mock_user, self.db)
        self.assertEqual(response['message'], 'User deleted successfully.')

    async def test_update_my_pwd_incorrect_old_password(self) -> None:
        """Test updating own password with incorrect old password."""
        self.current_user.check_password.return_value = False

        payload = UpdateMyPassword(
            old_password='wrong', new_password='newpass',
        )

        with self.assertRaises(HTTPException) as ctx:
            await users.update_my_pwd(
                payload,
                self.db,
                AsyncMock(),
                self.current_user,
            )

        self.assertEqual(ctx.exception.status_code, 401)

    @patch('examples.db_management.routers.users.update_password')
    async def test_admin_update_pwd(
        self,
        mock_update_password: AsyncMock,
    ) -> None:
        """Test admin updating user's password by username."""
        mock_user = MagicMock()
        mock_user.username = 'user'
        # 修正：db.execute 回傳 mock，其 scalar_one_or_none 是同步方法
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        self.db.execute.return_value = mock_result

        payload = UpdatePassword(username='user', new_password='newpass')
        response = await users.admin_update_pwd(payload, self.db)

        mock_update_password.assert_awaited_with(mock_user, 'newpass', self.db)
        self.assertEqual(response['message'], 'Password updated successfully.')

    async def test_admin_update_pwd_user_not_found(self) -> None:
        """Test admin updating password when user not found."""
        # 修正：db.execute 回傳 mock，其 scalar_one_or_none 是同步方法
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        self.db.execute.return_value = mock_result

        payload = UpdatePassword(username='missing', new_password='pass')

        with self.assertRaises(HTTPException) as ctx:
            await users.admin_update_pwd(payload, self.db)

        self.assertEqual(ctx.exception.status_code, 404)

    @patch('examples.db_management.routers.users.update_username')
    async def test_change_username(
        self,
        mock_update_username: AsyncMock,
    ) -> None:
        """Test changing username successfully."""
        mock_user = MagicMock()
        mock_user.username = 'old'
        # 修正：db.execute 回傳 mock，其 scalar_one_or_none 是同步方法
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        self.db.execute.return_value = mock_result

        payload = UpdateUsername(old_username='old', new_username='new')
        response = await users.change_username(payload, self.db)

        mock_update_username.assert_awaited_with(mock_user, 'new', self.db)
        self.assertEqual(response['message'], 'Username updated successfully.')

    @patch('examples.db_management.routers.users.update_username')
    async def test_change_username_user_not_found(
        self,
        mock_update_username: AsyncMock,
    ) -> None:
        """Test changing username when user is not found (should raise 404)."""
        # db.execute().scalar_one_or_none() 回傳 None
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        self.db.execute.return_value = mock_result
        payload = UpdateUsername(old_username='notfound', new_username='new')
        with self.assertRaises(HTTPException) as ctx:
            await users.change_username(payload, self.db)
        self.assertEqual(ctx.exception.status_code, 404)
        mock_update_username.assert_not_called()

    @patch('examples.db_management.routers.users.create_or_update_profile')
    async def test_update_profile(
        self,
        mock_update_profile: AsyncMock,
    ) -> None:
        """Test updating user profile successfully."""
        mock_user = MagicMock()

        with patch(
            'examples.db_management.routers.users.get_user_by_id',
            AsyncMock(return_value=mock_user),
        ):
            payload = UserProfileUpdate(user_id=1, email='test@example.com')
            response = await users.update_profile(payload, self.db)

            mock_update_profile.assert_awaited()
            self.assertEqual(
                response['message'],
                'User profile updated successfully.',
            )

    async def test_list_users(self) -> None:
        """Test listing all users with group and profile info."""
        # 構造兩個完整的 user ORM mock
        mock_group1 = MagicMock()
        mock_group1.name = 'group1'
        mock_group1.uniform_number = 'G123'
        mock_profile1 = MagicMock()
        mock_profile1.family_name = 'Family1'
        mock_profile1.middle_name = 'Middle1'
        mock_profile1.given_name = 'Given1'
        mock_profile1.email = 'test1@example.com'
        mock_profile1.mobile_number = '0911111111'
        mock_user1 = MagicMock(
            id=1,
            username='user1',
            role='user',
            group=mock_group1,
            profile=mock_profile1,
        )
        mock_group2 = MagicMock()
        mock_group2.name = 'group2'
        mock_group2.uniform_number = 'G456'
        mock_profile2 = MagicMock()
        mock_profile2.family_name = 'Family2'
        mock_profile2.middle_name = 'Middle2'
        mock_profile2.given_name = 'Given2'
        mock_profile2.email = 'test2@example.com'
        mock_profile2.mobile_number = '0922222222'
        mock_user2 = MagicMock(
            id=2,
            username='user2',
            role='admin',
            group=mock_group2,
            profile=mock_profile2,
        )
        # scalars().all() 回傳 user list
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [
            mock_user1, mock_user2,
        ]
        self.db.execute.return_value = mock_result

        result = await users.list_users(self.db)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].username, 'user1')
        self.assertEqual(result[1].group.name, 'group2')
        self.assertEqual(result[1].profile.email, 'test2@example.com')

    @patch('examples.db_management.routers.users.update_password')
    @patch('examples.db_management.routers.users.get_user_by_id')
    async def test_admin_update_pwd_by_id(
        self,
        mock_get_user_by_id: AsyncMock,
        mock_update_password: AsyncMock,
    ) -> None:
        """Test admin updating user's password by user ID."""
        mock_user = MagicMock()
        mock_get_user_by_id.return_value = mock_user
        payload = UpdatePasswordById(user_id=1, new_password='newpass')
        response = await users.admin_update_pwd_by_id(payload, self.db)
        mock_get_user_by_id.assert_awaited_with(1, self.db)
        mock_update_password.assert_awaited_with(mock_user, 'newpass', self.db)
        self.assertEqual(
            response['message'],
            'Password updated successfully by user ID.',
        )

    @patch('examples.auth.cache.set_user_data')
    @patch('examples.auth.cache.get_user_data')
    @patch('examples.db_management.routers.users.update_password')
    async def test_update_my_pwd_success(
        self,
        mock_update_password,
        mock_get_user_data,
        mock_set_user_data,
    ):
        """
        Test updating own password successfully and clearing redis tokens.
        """
        self.current_user.check_password = AsyncMock(return_value=True)
        redis_pool = MagicMock()
        mock_get_user_data.return_value = {
            'jti_list': ['a'], 'refresh_tokens': ['b'],
        }
        payload = UpdateMyPassword(old_password='old', new_password='new')
        response = await users.update_my_pwd(
            payload,
            self.db,
            redis_pool,
            self.current_user,
        )
        mock_update_password.assert_awaited_with(
            self.current_user, 'new', self.db,
        )
        mock_get_user_data.assert_awaited_with(
            redis_pool, self.current_user.username,
        )
        mock_set_user_data.assert_awaited()
        self.assertEqual(
            response['message'],
            'Password changed successfully, please log in again.',
        )

    @patch('examples.db_management.routers.users.update_username')
    @patch('examples.db_management.routers.users.get_user_by_id')
    async def test_change_username_by_id(
        self,
        mock_get_user_by_id: AsyncMock,
        mock_update_username: AsyncMock,
    ) -> None:
        """Test changing username by user ID successfully."""
        mock_user = MagicMock()
        mock_get_user_by_id.return_value = mock_user
        payload = UpdateUsernameById(user_id=1, new_username='newiduser')
        response = await users.change_username_by_id(payload, self.db)
        mock_get_user_by_id.assert_awaited_with(1, self.db)
        mock_update_username.assert_awaited_with(
            mock_user, 'newiduser', self.db,
        )
        self.assertEqual(response['message'], 'Username updated successfully.')

    @patch('examples.db_management.routers.users.set_active_status')
    @patch('examples.db_management.routers.users.get_user_by_id')
    async def test_activate_user(
        self,
        mock_get_user_by_id: AsyncMock,
        mock_set_active_status: AsyncMock,
    ) -> None:
        """Test setting a user's active status successfully."""
        mock_user = MagicMock()
        mock_get_user_by_id.return_value = mock_user
        payload = SetUserActiveStatus(user_id=1, is_active=True)
        response = await users.activate_user(payload, self.db)
        mock_get_user_by_id.assert_awaited_with(1, self.db)
        mock_set_active_status.assert_awaited_with(mock_user, True, self.db)
        self.assertEqual(
            response['message'],
            'User active status updated successfully.',
        )

    @patch('examples.db_management.routers.users.get_user_by_id')
    @patch('examples.db_management.routers.users.ensure_not_super')
    @patch('examples.db_management.routers.users.is_super_admin')
    async def test_change_role_user_to_user(
        self,
        mock_is_super_admin,
        mock_ensure_not_super,
        mock_get_user_by_id,
    ):
        """Test changing role from user to user (normal case)."""
        mock_user = MagicMock()
        mock_user.role = 'user'
        mock_get_user_by_id.return_value = mock_user
        mock_ensure_not_super.return_value = None
        mock_is_super_admin.return_value = False

        payload = UpdateUserRole(user_id=1, new_role='user')
        db = AsyncMock()
        me = MagicMock()
        response = await users.change_role(payload, db, me)
        self.assertEqual(
            response['message'],
            'User role updated successfully.',
        )
        self.assertEqual(mock_user.role, 'user')
        db.commit.assert_awaited()

    @patch('examples.db_management.routers.users.get_user_by_id')
    @patch('examples.db_management.routers.users.ensure_not_super')
    @patch('examples.db_management.routers.users.is_super_admin')
    async def test_change_role_to_admin_without_super_admin(
        self,
        mock_is_super_admin,
            mock_ensure_not_super,
            mock_get_user_by_id,
    ):
        """
        Test changing role to admin by non-super admin (should raise 403).
        """
        mock_user = MagicMock()
        mock_get_user_by_id.return_value = mock_user
        mock_ensure_not_super.return_value = None
        mock_is_super_admin.return_value = False

        payload = UpdateUserRole(user_id=1, new_role='admin')
        db = AsyncMock()
        me = MagicMock()
        with self.assertRaises(HTTPException) as ctx:
            await users.change_role(payload, db, me)
        self.assertEqual(ctx.exception.status_code, 403)
        db.commit.assert_not_awaited()

    @patch('examples.db_management.routers.users.get_user_by_id')
    @patch('examples.db_management.routers.users.ensure_not_super')
    @patch('examples.db_management.routers.users.is_super_admin')
    async def test_change_role_to_admin_with_super_admin(
        self,
        mock_is_super_admin: AsyncMock,
        mock_ensure_not_super: AsyncMock,
        mock_get_user_by_id: AsyncMock,
    ):
        """Test changing role to admin by super admin (success)."""
        mock_user = MagicMock()
        mock_user.role = 'user'
        mock_get_user_by_id.return_value = mock_user
        mock_ensure_not_super.return_value = None
        mock_is_super_admin.return_value = True

        payload = UpdateUserRole(user_id=1, new_role='admin')
        db = AsyncMock()
        me = MagicMock()
        response = await users.change_role(payload, db, me)
        self.assertEqual(
            response['message'],
            'User role updated successfully.',
        )
        self.assertEqual(mock_user.role, 'admin')
        db.commit.assert_awaited()

    @patch('examples.db_management.routers.users.get_user_by_id')
    @patch('examples.db_management.routers.users.ensure_not_super')
    async def test_change_role_ensure_not_super_raises(
        self,
        mock_ensure_not_super: AsyncMock,
        mock_get_user_by_id: AsyncMock,
    ):
        """
        Test changing role when ensure_not_super raises (should raise 403).
        """
        mock_user = MagicMock()
        mock_get_user_by_id.return_value = mock_user
        mock_ensure_not_super.side_effect = HTTPException(
            403, 'Cannot modify super admin.',
        )
        payload = UpdateUserRole(user_id=1, new_role='user')
        db = AsyncMock()
        me = MagicMock()
        with self.assertRaises(HTTPException) as ctx:
            await users.change_role(payload, db, me)
        self.assertEqual(ctx.exception.status_code, 403)
        db.commit.assert_not_awaited()

    @patch('examples.db_management.routers.users.get_user_by_id')
    async def test_change_group_success(self, mock_get_user_by_id):
        """Test updating user's group membership successfully."""
        mock_user = MagicMock()
        mock_get_user_by_id.return_value = mock_user
        payload = UpdateUserGroup(user_id=1, new_group_id=99)
        db = AsyncMock()
        response = await users.change_group(payload, db)
        mock_get_user_by_id.assert_awaited_with(1, db)
        self.assertEqual(mock_user.group_id, 99)
        db.commit.assert_awaited()
        self.assertEqual(
            response['message'],
            'User group updated successfully.',
        )


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.db_management.routers.users\
    --cov-report=term-missing\
        tests/examples/db_management/routers/users_test.py
'''
