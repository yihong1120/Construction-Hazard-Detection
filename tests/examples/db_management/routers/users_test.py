from __future__ import annotations

import unittest
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from examples.db_management.routers import users
from examples.db_management.schemas.user import AdminUserApproval
from examples.db_management.schemas.user import ApproveUserSignup
from examples.db_management.schemas.user import PendingUserReviewRead
from examples.db_management.schemas.user import SetUserStatus
from examples.db_management.schemas.user import UpdateMyPassword
from examples.db_management.schemas.user import UpdatePassword
from examples.db_management.schemas.user import UpdatePasswordById
from examples.db_management.schemas.user import UpdateUserGroup
from examples.db_management.schemas.user import UpdateUsername
from examples.db_management.schemas.user import UpdateUsernameById
from examples.db_management.schemas.user import UpdateUserRole
from examples.db_management.schemas.user import UserCreate
from examples.db_management.schemas.user import UserDelete
from examples.db_management.schemas.user import UserProfileBase
from examples.db_management.schemas.user import UserProfileUpdate
from examples.db_management.schemas.user import UserRead
from examples.db_management.schemas.user import UserSignup
from examples.db_management.services import (
    user_management_services as user_management,
)


class TestUsersRouter(unittest.IsolatedAsyncioTestCase):
    """Unit tests for user management router endpoints."""

    def setUp(self) -> None:
        """Set up common mock objects for tests.

        This method initialises a mock database session and a mock current user
        for use in each test case.
        """
        self.db: AsyncMock = AsyncMock(spec=AsyncSession)
        self.current_user: MagicMock = MagicMock()
        self.current_user.check_password = AsyncMock(return_value=True)
        self.current_user.username = 'testuser'
        self.current_user.role = 'admin'
        self.current_user.group_id = 10

    @patch('examples.db_management.routers.users.create_user')
    async def test_add_user(
        self,
        mock_create_user: AsyncMock,
    ) -> None:
        # Construct a complete user ORM mock
        """Test add user.

        Args:
            mock_create_user: Value used by this callable.
        """
        mock_group: MagicMock = MagicMock()
        mock_group.name = 'group1'
        mock_group.uniform_number = 'G123'
        mock_profile: MagicMock = MagicMock()
        mock_profile.family_name = 'Family'
        mock_profile.middle_name = 'Middle'
        mock_profile.given_name = 'Given'
        mock_profile.email = 'test@example.com'
        mock_profile.mobile_number = '0912345678'
        mock_user: MagicMock = MagicMock(
            id=1,
            username='newuser',
            role='user',
            status='active',
            group=mock_group,
            profile=mock_profile,
        )
        mock_create_user.return_value = mock_user

        payload: UserCreate = UserCreate(
            username='newuser',
            password='pass',
            group_id=10,
        )

        # db.execute returns a mock whose scalar_one is a sync method
        mock_result: MagicMock = MagicMock()
        mock_result.scalar_one.return_value = mock_user
        self.db.execute.return_value = mock_result

        response: UserRead = await users.add_user(
            payload,
            self.db,
            self.current_user,
        )

        self.assertIsInstance(response, UserRead)
        # Defensive checks for optional fields
        self.assertIsNotNone(response.group)
        self.assertIsNotNone(response.profile)
        if response.group is not None:
            self.assertEqual(response.group.name, 'group1')
        if response.profile is not None:
            self.assertEqual(response.profile.email, 'test@example.com')

    @patch(
        'examples.db_management.services.user_management_services.'
        'send_signup_verification_email',
    )
    @patch(
        'examples.db_management.services.user_management_services.'
        'record_user_consent',
    )
    @patch(
        'examples.db_management.services.user_management_services.'
        'validate_signup_consents',
    )
    @patch(
        'examples.db_management.services.user_management_services.create_user',
    )
    async def test_signup_user(
        self,
        mock_create_user: AsyncMock,
        mock_validate_consents: AsyncMock,
        mock_record_consent: AsyncMock,
        mock_send_verification: AsyncMock,
    ) -> None:
        """Test self-service signup creates an email-unverified user."""
        mock_profile: MagicMock = MagicMock()
        mock_profile.family_name = 'Family'
        mock_profile.middle_name = None
        mock_profile.given_name = 'Given'
        mock_profile.email = 'pending@example.com'
        mock_profile.mobile_number = None
        mock_user: MagicMock = MagicMock(
            id=2,
            username='pending',
            role='user',
            status='email_unverified',
            email_verified_at=None,
            group=None,
            group_id=None,
            profile=mock_profile,
        )
        mock_create_user.return_value = mock_user

        signup_profile: UserProfileBase = UserProfileBase(
            family_name='Family',
            given_name='Given',
            email='pending@example.com',
        )
        payload: UserSignup = UserSignup(
            username='pending',
            password='pass',
            profile=signup_profile,
            accepted_terms=True,
            terms_version='2026-06-27',
            privacy_version='2026-06-27',
            notification_consent=True,
            ai_terms_accepted=True,
            ai_terms_version='2026-06-27',
        )
        request: MagicMock = MagicMock()
        redis: AsyncMock = AsyncMock()

        mock_result: MagicMock = MagicMock()
        mock_result.scalar_one.return_value = mock_user
        self.db.execute.return_value = mock_result

        response: UserRead = await users.signup_user(
            payload,
            request=request,
            db=self.db,
            redis=redis,
        )

        mock_validate_consents.assert_awaited_once_with(payload, self.db)
        mock_create_user.assert_awaited_once_with(
            username='pending',
            password='pass',
            role='user',
            group_id=None,
            db=self.db,
            profile={
                'family_name': 'Family',
                'middle_name': None,
                'given_name': 'Given',
                'email': 'pending@example.com',
                'mobile_number': None,
            },
            status='email_unverified',
        )
        mock_record_consent.assert_awaited_once_with(
            2,
            payload,
            self.db,
            request,
        )
        mock_send_verification.assert_awaited_once_with(mock_user, redis)
        self.assertEqual(response.status, 'email_unverified')
        self.assertIsNone(response.group_id)

    @patch(
        'examples.db_management.services.user_management_services.'
        'record_user_consent',
    )
    @patch(
        'examples.db_management.services.user_management_services.'
        'validate_signup_consents',
    )
    @patch(
        'examples.db_management.services.user_management_services.create_user',
    )
    async def test_signup_user_rejects_missing_consents(
        self,
        mock_create_user: AsyncMock,
        mock_validate_consents: AsyncMock,
        mock_record_consent: AsyncMock,
    ) -> None:
        """Signup stops before account creation when consents are invalid."""
        mock_validate_consents.side_effect = HTTPException(
            status_code=400,
            detail='accepted_terms is required.',
        )
        signup_profile: UserProfileBase = UserProfileBase(
            family_name='Family',
            given_name='Given',
            email='pending@example.com',
        )
        payload = UserSignup(
            username='pending',
            password='pass',
            profile=signup_profile,
        )

        with self.assertRaises(HTTPException) as ctx:
            await users.signup_user(
                payload,
                request=MagicMock(),
                db=self.db,
                redis=AsyncMock(),
            )

        self.assertEqual(ctx.exception.status_code, 400)
        mock_create_user.assert_not_awaited()
        mock_record_consent.assert_not_awaited()

    @patch('examples.db_management.routers.users.get_user_by_id')
    @patch('examples.db_management.routers.users.delete_user')
    async def test_remove_user(
        self,
        mock_delete_user: AsyncMock,
        mock_get_user_by_id: AsyncMock,
    ) -> None:
        """Test remove user.

        Args:
            mock_delete_user: Value used by this callable.
            mock_get_user_by_id: Value used by this callable.
        """
        mock_user: MagicMock = MagicMock()
        mock_user.group_id = 10
        mock_get_user_by_id.return_value = mock_user

        payload = UserDelete(user_id=1)
        response: dict[str, str] = await users.remove_user(
            payload,
            self.db,
            self.current_user,
        )

        mock_delete_user.assert_awaited_with(mock_user, self.db)
        self.assertEqual(response['message'], 'User deleted successfully.')

    async def test_update_my_pwd_incorrect_old_password(self) -> None:
        """Test update my pwd incorrect old password."""
        self.current_user.check_password.return_value = False

        payload: UpdateMyPassword = UpdateMyPassword(
            old_password='wrong',
            new_password='newpass',
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
        mock_user: MagicMock = MagicMock()
        mock_user.username = 'user'
        mock_user.group_id = 10
        # db.execute returns a mock whose scalar_one_or_none is a sync method
        mock_result: MagicMock = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        self.db.execute.return_value = mock_result

        payload: UpdatePassword = UpdatePassword(
            username='user',
            new_password='newpass',
        )
        response: dict[str, str] = await users.admin_update_pwd(
            payload,
            self.db,
            self.current_user,
        )

        mock_update_password.assert_awaited_with(mock_user, 'newpass', self.db)
        self.assertEqual(response['message'], 'Password updated successfully.')

    async def test_admin_update_pwd_user_not_found(self) -> None:
        """Test admin updating password when user not found."""
        mock_result: MagicMock = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        self.db.execute.return_value = mock_result

        payload: UpdatePassword = UpdatePassword(
            username='missing',
            new_password='pass',
        )

        with self.assertRaises(HTTPException) as ctx:
            await users.admin_update_pwd(payload, self.db, self.current_user)

        self.assertEqual(ctx.exception.status_code, 404)

    @patch('examples.db_management.routers.users.update_username')
    async def test_change_username(
        self,
        mock_update_username: AsyncMock,
    ) -> None:
        """Test changing username successfully."""
        mock_user: MagicMock = MagicMock()
        mock_user.username = 'old'
        mock_user.group_id = 10
        mock_result: MagicMock = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        self.db.execute.return_value = mock_result

        payload: UpdateUsername = UpdateUsername(
            old_username='old',
            new_username='new',
        )
        response: dict[str, str] = await users.change_username(
            payload,
            self.db,
            self.current_user,
        )

        mock_update_username.assert_awaited_with(mock_user, 'new', self.db)
        self.assertEqual(response['message'], 'Username updated successfully.')

    @patch('examples.db_management.routers.users.update_username')
    async def test_change_username_user_not_found(
        self,
        mock_update_username: AsyncMock,
    ) -> None:
        """Test changing username when user is not found (should raise 404)."""
        mock_result: MagicMock = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        self.db.execute.return_value = mock_result
        payload: UpdateUsername = UpdateUsername(
            old_username='notfound',
            new_username='new',
        )
        with self.assertRaises(HTTPException) as ctx:
            await users.change_username(payload, self.db, self.current_user)
        self.assertEqual(ctx.exception.status_code, 404)
        mock_update_username.assert_not_called()

    @patch('examples.db_management.routers.users.create_or_update_profile')
    async def test_update_profile(
        self,
        mock_update_profile: AsyncMock,
    ) -> None:
        """Test updating user profile successfully."""
        mock_user: MagicMock = MagicMock()
        mock_user.group_id = 10

        with patch(
            'examples.db_management.routers.users.get_user_by_id',
            AsyncMock(return_value=mock_user),
        ):
            payload: UserProfileUpdate = UserProfileUpdate(
                user_id=1,
                email='test@example.com',
            )
            response: dict[str, str] = await users.update_profile(
                payload,
                self.db,
                self.current_user,
            )

            mock_update_profile.assert_awaited()
            self.assertEqual(
                response['message'],
                'User profile updated successfully.',
            )

    @patch('examples.db_management.routers.users.list_users_for_operator')
    async def test_list_users(
        self,
        list_users_for_operator: AsyncMock,
    ) -> None:
        """The HTTP handler only delegates to the listing application
        service."""
        expected = users.UserPage(items=[], next_cursor=None)
        list_users_for_operator.return_value = expected
        result = await users.list_users(self.db, self.current_user)
        self.assertIs(result, expected)
        list_users_for_operator.assert_awaited_once_with(
            self.current_user,
            self.db,
            cursor=None,
            page_size=50,
        )

    async def test_list_pending_users(self) -> None:
        """Test listing pending self-signup users."""
        mock_profile: MagicMock = MagicMock()
        mock_profile.family_name = 'Pending'
        mock_profile.middle_name = None
        mock_profile.given_name = 'User'
        mock_profile.email = 'pending@example.com'
        mock_profile.mobile_number = None
        pending_user: MagicMock = MagicMock(
            id=3,
            username='pending-user',
            role='user',
            status='pending_admin_approval',
            email_verified_at=datetime.now(),
            group_id=None,
            group=None,
            profile=mock_profile,
        )
        consent: MagicMock = MagicMock()
        consent.id = 9
        consent.accepted_at = datetime.now()
        consent.terms_version = '2026-06-27'
        consent.privacy_version = '2026-06-27'
        consent.ai_terms_version = '2026-06-27'
        consent.notification_consent = True
        identity: MagicMock = MagicMock()
        identity.provider = 'google'
        pending_user.consents = [consent]
        pending_user.identities = [identity]
        mock_result: MagicMock = MagicMock()
        mock_result.scalars.return_value.all.return_value = [pending_user]
        self.db.execute.return_value = mock_result

        result = await users.list_pending_users(self.db)

        self.assertEqual(len(result.items), 1)
        self.assertIsInstance(result.items[0], PendingUserReviewRead)
        self.assertEqual(result.items[0].username, 'pending-user')
        self.assertEqual(result.items[0].status, 'pending_admin_approval')
        self.assertIsNone(result.items[0].group_id)
        self.assertEqual(result.items[0].email, 'pending@example.com')
        self.assertEqual(result.items[0].terms_version, '2026-06-27')
        self.assertEqual(result.items[0].provider, 'google')

    async def test_list_pending_users_accepts_a_keyset_cursor(self) -> None:
        """Pending-user queries constrain subsequent pages by user ID."""
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        self.db.execute.return_value = result

        page = await users.list_pending_users(self.db, cursor=7)

        self.assertEqual(page.items, [])
        self.db.execute.assert_awaited_once()

    @patch(
        'examples.db_management.services.user_management_services.'
        'load_user_read',
    )
    @patch(
        'examples.db_management.services.user_management_services.'
        'get_group_or_404',
    )
    @patch('examples.db_management.routers.users.get_user_by_id')
    async def test_approve_user_signup_admin_uses_own_group(
        self,
        mock_get_user_by_id: AsyncMock,
        mock_get_group: AsyncMock,
        mock_load_user_read: AsyncMock,
    ) -> None:
        """Test admin approval defaults to the admin's own group."""
        now = datetime.now()
        pending_user: MagicMock = MagicMock(
            id=4,
            username='pending',
            role='user',
            status='pending_admin_approval',
            email_verified_at=now,
            group_id=None,
        )
        mock_get_user_by_id.return_value = pending_user
        mock_get_group.return_value = MagicMock(id=10)
        mock_load_user_read.return_value = UserRead(
            id=4,
            username='pending',
            role='user',
            status='active',
            group_id=10,
            group=None,
            profile=None,
            created_at=now,
            updated_at=now,
        )
        payload: ApproveUserSignup = ApproveUserSignup(user_id=4)

        # db.execute for site_ids query returns no sites → no pref inserts
        mock_site_result: MagicMock = MagicMock()
        mock_site_result.scalars.return_value.all.return_value = []
        self.db.execute.return_value = mock_site_result

        result = await users.approve_user_signup(
            payload,
            self.db,
            self.current_user,
        )

        self.assertEqual(pending_user.group_id, 10)
        self.assertEqual(pending_user.status, 'active')
        mock_get_group.assert_awaited_once_with(10, self.db)
        mock_load_user_read.assert_awaited_once_with(4, self.db)
        self.db.commit.assert_awaited_once()
        self.assertEqual(result.username, 'pending')

    @patch(
        'examples.db_management.services.user_management_services.'
        'get_group_or_404',
    )
    @patch('examples.db_management.routers.users.get_user_by_id')
    async def test_approve_user_signup_rejects_other_group(
        self,
        mock_get_user_by_id: AsyncMock,
        mock_get_group: AsyncMock,
    ) -> None:
        """Test admin cannot approve signup into another group."""
        pending_user: MagicMock = MagicMock(
            id=5,
            username='pending-other',
            role='user',
            status='pending_admin_approval',
            email_verified_at=datetime.now(),
            group_id=None,
        )
        mock_get_user_by_id.return_value = pending_user
        payload: ApproveUserSignup = ApproveUserSignup(user_id=5, group_id=99)

        with self.assertRaises(HTTPException) as ctx:
            await users.approve_user_signup(
                payload,
                self.db,
                self.current_user,
            )

        self.assertEqual(ctx.exception.status_code, 403)
        self.db.commit.assert_not_awaited()
        mock_get_group.assert_not_called()

    @patch(
        'examples.db_management.services.user_management_services.'
        'get_group_or_404',
    )
    @patch('examples.db_management.routers.users.get_user_by_id')
    async def test_approve_user_signup_requires_pending_state(
        self,
        mock_get_user_by_id: AsyncMock,
        mock_get_group: AsyncMock,
    ) -> None:
        """Test approval rejects users that are not pending signups."""
        active_user: MagicMock = MagicMock(
            id=6,
            username='active-user',
            role='user',
            status='active',
            group_id=10,
        )
        mock_get_user_by_id.return_value = active_user
        payload: ApproveUserSignup = ApproveUserSignup(user_id=6, group_id=10)

        with self.assertRaises(HTTPException) as ctx:
            await users.approve_user_signup(
                payload,
                self.db,
                self.current_user,
            )

        self.assertEqual(ctx.exception.status_code, 400)
        mock_get_group.assert_not_called()

    @patch('examples.db_management.routers.users.update_password')
    @patch('examples.db_management.routers.users.get_user_by_id')
    async def test_admin_update_pwd_by_id(
        self,
        mock_get_user_by_id: AsyncMock,
        mock_update_password: AsyncMock,
    ) -> None:
        """Test admin updating user's password by user ID."""
        mock_user: MagicMock = MagicMock()
        mock_user.group_id = 10
        mock_get_user_by_id.return_value = mock_user
        payload: UpdatePasswordById = UpdatePasswordById(
            user_id=1,
            new_password='newpass',
        )
        response: dict[str, str] = await users.admin_update_pwd_by_id(
            payload,
            self.db,
            self.current_user,
        )
        mock_get_user_by_id.assert_awaited_with(1, self.db)
        mock_update_password.assert_awaited_with(mock_user, 'newpass', self.db)
        self.assertEqual(
            response['message'],
            'Password updated successfully by user ID.',
        )

    @patch('examples.auth.cache.rate_limiter_service.set_user_data')
    @patch('examples.auth.cache.rate_limiter_service.get_user_data')
    @patch('examples.db_management.routers.users.update_password')
    async def test_update_my_pwd_success(
        self,
        mock_update_password: AsyncMock,
        mock_get_user_data: AsyncMock,
        mock_set_user_data: AsyncMock,
    ) -> None:
        """Test updating own password successfully and clearing redis
        tokens."""
        self.current_user.check_password = AsyncMock(return_value=True)
        redis_pool: MagicMock = MagicMock()
        mock_get_user_data.return_value = {
            'jti_list': ['a'],
            'refresh_tokens': ['b'],
        }
        payload: UpdateMyPassword = UpdateMyPassword(
            old_password='old',
            new_password='new',
        )
        response: dict[str, str] = await users.update_my_pwd(
            payload,
            self.db,
            redis_pool,
            self.current_user,
        )
        mock_update_password.assert_awaited_with(
            self.current_user,
            'new',
            self.db,
        )
        mock_get_user_data.assert_awaited_with(
            redis_pool,
            self.current_user.username,
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
        mock_user: MagicMock = MagicMock()
        mock_user.group_id = 10
        mock_get_user_by_id.return_value = mock_user
        payload: UpdateUsernameById = UpdateUsernameById(
            user_id=1,
            new_username='newiduser',
        )
        response: dict[str, str] = await users.change_username_by_id(
            payload,
            self.db,
            self.current_user,
        )
        mock_get_user_by_id.assert_awaited_with(1, self.db)
        mock_update_username.assert_awaited_with(
            mock_user,
            'newiduser',
            self.db,
        )
        self.assertEqual(response['message'], 'Username updated successfully.')

    @patch('examples.db_management.routers.users.set_user_status')
    @patch('examples.db_management.routers.users.get_user_by_id')
    async def test_update_user_status(
        self,
        mock_get_user_by_id: AsyncMock,
        mock_set_user_status: AsyncMock,
    ) -> None:
        """Test setting a user's status successfully."""
        mock_user: MagicMock = MagicMock()
        mock_user.group_id = 10
        mock_get_user_by_id.return_value = mock_user
        payload: SetUserStatus = SetUserStatus(
            user_id=1,
            status='active',
        )
        response: dict[str, str] = await users.update_user_status(
            payload,
            self.db,
            self.current_user,
        )
        mock_get_user_by_id.assert_awaited_with(1, self.db)
        mock_set_user_status.assert_awaited_with(
            mock_user,
            'active',
            self.db,
        )
        self.assertEqual(
            response['message'],
            'User status updated successfully.',
        )

    @patch('examples.db_management.routers.users.get_user_by_id')
    @patch('examples.db_management.routers.users.ensure_not_super')
    @patch('examples.db_management.routers.users.is_super_admin')
    async def test_change_role_user_to_user(
        self,
        mock_is_super_admin: AsyncMock,
        mock_ensure_not_super: AsyncMock,
        mock_get_user_by_id: AsyncMock,
    ) -> None:
        """Test changing role from user to user (normal case)."""
        mock_user: MagicMock = MagicMock()
        mock_user.role = 'user'
        mock_user.group_id = 10
        mock_get_user_by_id.return_value = mock_user
        mock_ensure_not_super.return_value = None
        mock_is_super_admin.return_value = False

        payload: UpdateUserRole = UpdateUserRole(user_id=1, new_role='user')
        db: AsyncMock = AsyncMock()
        me: MagicMock = MagicMock(role='admin', group_id=10, username='admin')
        response: dict[str, str] = await users.change_role(payload, db, me)
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
        mock_is_super_admin: AsyncMock,
        mock_ensure_not_super: AsyncMock,
        mock_get_user_by_id: AsyncMock,
    ) -> None:
        """Test changing role to admin by non-super admin (should raise
        403)."""
        mock_user: MagicMock = MagicMock()
        mock_user.group_id = 10
        mock_get_user_by_id.return_value = mock_user
        mock_ensure_not_super.return_value = None
        mock_is_super_admin.return_value = False

        payload: UpdateUserRole = UpdateUserRole(user_id=1, new_role='admin')
        db: AsyncMock = AsyncMock()
        me: MagicMock = MagicMock(role='admin', group_id=10, username='admin')
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
    ) -> None:
        """Test changing role to admin by super admin (success)."""
        mock_user: MagicMock = MagicMock()
        mock_user.role = 'user'
        mock_user.group_id = 99
        mock_get_user_by_id.return_value = mock_user
        mock_ensure_not_super.return_value = None
        mock_is_super_admin.return_value = True

        payload: UpdateUserRole = UpdateUserRole(user_id=1, new_role='admin')
        db: AsyncMock = AsyncMock()
        me: MagicMock = MagicMock(
            role='admin',
            group_id=None,
            username='ChangDar',
        )
        response: dict[str, str] = await users.change_role(payload, db, me)
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
    ) -> None:
        """Test changing role when ensure_not_super raises (should raise
        403)."""
        mock_user: MagicMock = MagicMock()
        mock_user.group_id = 10
        mock_get_user_by_id.return_value = mock_user
        mock_ensure_not_super.side_effect = HTTPException(
            403,
            'Cannot modify super admin.',
        )
        payload: UpdateUserRole = UpdateUserRole(user_id=1, new_role='user')
        db: AsyncMock = AsyncMock()
        me: MagicMock = MagicMock(role='admin', group_id=10, username='admin')
        with self.assertRaises(HTTPException) as ctx:
            await users.change_role(payload, db, me)
        self.assertEqual(ctx.exception.status_code, 403)
        db.commit.assert_not_awaited()

    @patch('examples.db_management.routers.users.get_user_by_id')
    async def test_change_group_success(
        self,
        mock_get_user_by_id: AsyncMock,
    ) -> None:
        """Test updating user's group membership successfully."""
        mock_user: MagicMock = MagicMock()
        mock_user.group_id = 99
        mock_get_user_by_id.return_value = mock_user
        payload: UpdateUserGroup = UpdateUserGroup(user_id=1, new_group_id=99)
        me: MagicMock = MagicMock(role='admin', group_id=99, username='admin')
        db: AsyncMock = AsyncMock()
        with patch(
            'examples.db_management.routers.users.get_group_or_404',
            AsyncMock(return_value=MagicMock(id=99)),
        ) as mock_get_group:
            response: dict[str, str] = await users.change_group(
                payload,
                db,
                me,
            )
        mock_get_user_by_id.assert_awaited_with(1, db)
        mock_get_group.assert_awaited_with(99, db)
        self.assertEqual(mock_user.group_id, 99)
        db.commit.assert_awaited()
        self.assertEqual(
            response['message'],
            'User group updated successfully.',
        )

    @patch('examples.db_management.routers.users.get_user_by_id')
    async def test_change_group_forbidden_for_other_group(
        self,
        mock_get_user_by_id: AsyncMock,
    ) -> None:
        """Test admin cannot manually assign a user into another group."""
        mock_user: MagicMock = MagicMock()
        mock_user.group_id = 10
        mock_get_user_by_id.return_value = mock_user
        payload: UpdateUserGroup = UpdateUserGroup(user_id=1, new_group_id=99)
        me: MagicMock = MagicMock(role='admin', group_id=10, username='admin')
        db: AsyncMock = AsyncMock()

        with self.assertRaises(HTTPException) as ctx:
            await users.change_group(payload, db, me)

        self.assertEqual(ctx.exception.status_code, 403)
        db.commit.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()


class TestUserRouterCoverage(unittest.IsolatedAsyncioTestCase):
    """Cover the user-management authorization and alias edge cases."""

    def setUp(self) -> None:
        """Perform setUp."""
        self.admin: Any = SimpleNamespace(
            username='admin',
            group_id=10,
            role='admin',
        )
        self.db = AsyncMock()

    async def test_group_lookup_and_scope_failures(self) -> None:
        """Missing groups and cross-group actions return their HTTP errors."""
        missing_result = MagicMock()
        missing_result.unique.return_value.scalar_one_or_none.return_value = (
            None
        )
        self.db.execute.return_value = missing_result

        with self.assertRaisesRegex(HTTPException, 'Group not found'):
            await user_management.get_group_or_404(999, self.db)

        group = SimpleNamespace(id=10)
        found_result = MagicMock()
        found_result.unique.return_value.scalar_one_or_none.return_value = (
            group
        )
        self.db.execute.return_value = found_result
        self.assertIs(
            await user_management.get_group_or_404(10, self.db),
            group,
        )

        with (
            patch.object(
                user_management,
                'is_super_admin',
                return_value=False,
            ),
            patch.object(
                user_management,
                'ensure_admin_with_group',
            ),
        ):
            with self.assertRaisesRegex(HTTPException, 'group_id is required'):
                user_management.resolve_target_group_id(None, self.admin)
            with self.assertRaisesRegex(
                HTTPException,
                'Cannot manage users outside your group',
            ):
                target_user: Any = SimpleNamespace(group_id=11)
                user_management.ensure_user_management_scope(
                    target_user,
                    self.admin,
                )

        with patch.object(
            user_management,
            'is_super_admin',
            return_value=True,
        ):
            self.assertEqual(
                user_management.resolve_target_group_id(99, self.admin),
                99,
            )

    async def test_add_user_seeds_group_sites_and_commits(self) -> None:
        """Adding a grouped user persists notification preferences."""
        payload = UserCreate(
            username='new-user',
            password='secret',
            group_id=10,
        )
        new_user = SimpleNamespace(id=42)
        expected = SimpleNamespace(id=42)

        with (
            patch.object(users, 'is_super_admin', return_value=False),
            patch.object(
                users,
                'create_user',
                AsyncMock(return_value=new_user),
            ),
            patch.object(
                users,
                'list_site_ids_for_group',
                AsyncMock(return_value=[7, 8]),
            ) as list_sites,
            patch.object(
                users,
                'seed_site_notification_preferences',
                AsyncMock(),
            ) as seed_preferences,
            patch.object(
                users,
                'load_user_read',
                AsyncMock(return_value=expected),
            ),
            patch.object(users, 'invalidate_effective_site_cache'),
        ):
            result = await users.add_user(payload, self.db, self.admin)

        self.assertIs(result, expected)
        list_sites.assert_awaited_once_with(10, self.db)
        seed_preferences.assert_awaited_once_with(
            user_ids=[42],
            site_ids=[7, 8],
            db=self.db,
        )
        self.db.commit.assert_awaited_once()

    async def test_signup_approval_routes_cover_rejection_and_approval(
        self,
    ) -> None:
        """Reviewing a pending signup either rejects or delegates approval."""
        pending_user = SimpleNamespace(
            id=8,
            role='user',
            status='pending_admin_approval',
            email_verified_at=datetime(2026, 7, 24),
            group_id=None,
        )
        rejected_result = SimpleNamespace(id=8, status='rejected')

        with (
            patch.object(
                users,
                'get_user_by_id',
                AsyncMock(return_value=pending_user),
            ),
            patch.object(users, 'ensure_not_super'),
            patch.object(
                users,
                'load_user_read',
                AsyncMock(return_value=rejected_result),
            ),
            patch.object(users, 'invalidate_effective_site_cache'),
        ):
            result = await users.review_user_signup(
                8,
                AdminUserApproval(decision='rejected'),
                self.db,
                self.admin,
            )

        self.assertIs(result, rejected_result)
        self.assertEqual(pending_user.status, 'rejected')
        self.db.commit.assert_awaited_once()

        pending_user.status = 'pending_admin_approval'
        approved_result = SimpleNamespace(id=8, status='active')
        with (
            patch.object(
                users,
                'get_user_by_id',
                AsyncMock(return_value=pending_user),
            ),
            patch.object(users, 'ensure_not_super'),
            patch.object(
                users,
                'approve_signup_user',
                AsyncMock(return_value=approved_result),
            ) as approve,
        ):
            result = await users.review_user_signup(
                8,
                AdminUserApproval(decision='approved', group_id=10),
                self.db,
                self.admin,
            )

        self.assertIs(result, approved_result)
        approve.assert_awaited_once_with(pending_user, 10, self.db, self.admin)

    async def test_approval_and_group_change_require_a_resolved_group(
        self,
    ) -> None:
        """A resolved group is mandatory before activating or moving a user."""
        with patch.object(
            user_management,
            'resolve_target_group_id',
            return_value=None,
        ):
            with self.assertRaisesRegex(HTTPException, 'group_id is required'):
                pending_user: Any = SimpleNamespace(
                    id=5,
                    group_id=None,
                    status='pending',
                )
                await user_management.approve_signup_user(
                    pending_user,
                    None,
                    self.db,
                    self.admin,
                )

        target = SimpleNamespace(id=5, group_id=10)
        with (
            patch.object(
                users,
                'get_user_by_id',
                AsyncMock(return_value=target),
            ),
            patch.object(users, 'ensure_not_super'),
            patch.object(users, 'ensure_user_management_scope'),
            patch.object(users, 'resolve_target_group_id', return_value=None),
        ):
            with self.assertRaisesRegex(HTTPException, 'group_id is required'):
                await users.change_group(
                    SimpleNamespace(user_id=5, new_group_id=10),
                    self.db,
                    self.admin,
                )

        invalid_user = SimpleNamespace(
            id=6,
            role='user',
            status='active',
            email_verified_at=datetime(2026, 7, 24),
            group_id=None,
        )
        with (
            patch.object(
                users,
                'get_user_by_id',
                AsyncMock(return_value=invalid_user),
            ),
            patch.object(users, 'ensure_not_super'),
        ):
            with self.assertRaisesRegex(
                HTTPException,
                'not awaiting signup approval',
            ):
                await users.review_user_signup(
                    6,
                    AdminUserApproval(decision='approved', group_id=10),
                    self.db,
                    self.admin,
                )

    async def test_super_admin_lists_all_users(self) -> None:
        """Super administrators delegate to the same scoped-list service."""
        expected = users.UserPage(items=[], next_cursor=None)
        with patch.object(
            users,
            'list_users_for_operator',
            AsyncMock(return_value=expected),
        ) as list_users_for_operator:
            result = await users.list_users(self.db, self.admin)

        self.assertIs(result, expected)
        list_users_for_operator.assert_awaited_once_with(
            self.admin,
            self.db,
            cursor=None,
            page_size=50,
        )


if __name__ == '__main__':
    unittest.main()
