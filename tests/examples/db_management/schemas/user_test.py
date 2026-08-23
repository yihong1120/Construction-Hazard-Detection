from __future__ import annotations

import unittest
from datetime import datetime
from typing import Any
from typing import cast

from pydantic import ValidationError

from examples.db_management.schemas import user
from examples.db_management.schemas.group import GroupRead


class TestUserProfileBase(unittest.TestCase):
    """Unit tests for the UserProfileBase schema."""

    def test_valid(self) -> None:
        """Test creating a valid UserProfileBase instance."""
        data: user.UserProfileBase = user.UserProfileBase(
            family_name='f',
            given_name='g',
            email='a@b.com',
        )
        self.assertEqual(data.family_name, 'f')
        self.assertIsNone(data.middle_name)
        self.assertEqual(data.given_name, 'g')
        self.assertEqual(data.email, 'a@b.com')
        self.assertIsNone(data.mobile_number)

    def test_missing(self) -> None:
        """Test validation error when required fields are missing."""
        with self.assertRaises(ValidationError):
            cast(Any, user.UserProfileBase)(
                family_name='f',
                given_name='g',
            )
        with self.assertRaises(ValidationError):
            cast(Any, user.UserProfileBase)(given_name='g', email='a@b.com')


class TestUserProfileRead(unittest.TestCase):
    """Unit tests for the UserProfileRead schema."""

    def test_valid(self) -> None:
        """Test creating a valid UserProfileRead instance."""
        now: datetime = datetime.now()
        data: user.UserProfileRead = user.UserProfileRead(
            family_name='f',
            given_name='g',
            email='a@b.com',
            created_at=now,
            updated_at=now,
        )
        self.assertEqual(data.created_at, now)
        self.assertEqual(data.updated_at, now)


class TestUserProfileUpdate(unittest.TestCase):
    """Unit tests for the UserProfileUpdate schema."""

    def test_valid(self) -> None:
        """Test updating UserProfileUpdate with required fields."""
        data: user.UserProfileUpdate = user.UserProfileUpdate(
            user_id=1,
            family_name='f',
            email='a@b.com',
        )
        self.assertEqual(data.user_id, 1)
        self.assertEqual(data.family_name, 'f')
        self.assertEqual(data.email, 'a@b.com')

    def test_optional(self) -> None:
        """Test optional fields of UserProfileUpdate."""
        data: user.UserProfileUpdate = user.UserProfileUpdate(user_id=2)
        self.assertIsNone(data.family_name)
        self.assertIsNone(data.email)

    def test_missing(self) -> None:
        """Test validation error when user_id is missing."""
        with self.assertRaises(ValidationError):
            cast(Any, user.UserProfileUpdate)()


class TestUserCreate(unittest.TestCase):
    """Unit tests for the UserCreate schema."""

    def test_valid(self) -> None:
        """Test creating a valid UserCreate instance."""
        data: user.UserCreate = user.UserCreate(
            username='u',
            password='p',
            group_id=1,
        )
        self.assertEqual(data.username, 'u')
        self.assertEqual(data.password, 'p')
        self.assertEqual(data.role, 'user')
        self.assertEqual(data.group_id, 1)
        self.assertIsNone(data.profile)

    def test_with_profile(self) -> None:
        """Test UserCreate instance with profile included."""
        profile: user.UserProfileBase = user.UserProfileBase(
            family_name='f',
            given_name='g',
            email='a@b.com',
        )
        data: user.UserCreate = user.UserCreate(
            username='u',
            password='p',
            group_id=1,
            profile=profile,
        )
        self.assertIsNotNone(data.profile)
        assert data.profile is not None  # Inform the type checker explicitly.
        self.assertEqual(data.profile.family_name, 'f')

    def test_missing(self) -> None:
        """Test validation errors when required fields are missing."""
        with self.assertRaises(ValidationError):
            cast(Any, user.UserCreate)(username='u', password='p')
        with self.assertRaises(ValidationError):
            cast(Any, user.UserCreate)(password='p', group_id=1)


class TestUserSignup(unittest.TestCase):
    """Unit tests for the UserSignup schema."""

    def test_valid(self) -> None:
        """Test creating a valid UserSignup instance."""
        profile: user.UserProfileBase = user.UserProfileBase(
            family_name='f',
            given_name='g',
            email='a@b.com',
        )
        data: user.UserSignup = user.UserSignup(
            username='u',
            password='p',
            profile=profile,
            accepted_terms=True,
            terms_version='2026-06-27',
            privacy_version='2026-06-27',
            notification_consent=True,
            ai_terms_accepted=True,
            ai_terms_version='2026-06-27',
        )
        self.assertEqual(data.username, 'u')
        self.assertEqual(data.profile.email, 'a@b.com')
        self.assertTrue(data.accepted_terms)
        self.assertEqual(data.terms_version, '2026-06-27')

    def test_missing_profile(self) -> None:
        """Test validation error when profile is missing."""
        with self.assertRaises(ValidationError):
            cast(Any, user.UserSignup)(username='u', password='p')


class TestApproveUserSignup(unittest.TestCase):
    """Unit tests for the ApproveUserSignup schema."""

    def test_valid_with_group(self) -> None:
        """Test approving a signup with an explicit group."""
        data: user.ApproveUserSignup = user.ApproveUserSignup(
            user_id=1,
            group_id=2,
        )
        self.assertEqual(data.user_id, 1)
        self.assertEqual(data.group_id, 2)

    def test_valid_without_group(self) -> None:
        """Test approving a signup without explicit group."""
        data: user.ApproveUserSignup = user.ApproveUserSignup(user_id=1)
        self.assertEqual(data.user_id, 1)
        self.assertIsNone(data.group_id)


class TestUserRead(unittest.TestCase):
    """Unit tests for the UserRead schema."""

    def test_valid(self) -> None:
        """Test creating a valid UserRead instance."""
        now: datetime = datetime.now()
        group_obj: GroupRead = GroupRead(
            id=1,
            name='g',
            uniform_number='12345678',
        )
        profile_obj: user.UserProfileRead = user.UserProfileRead(
            family_name='f',
            given_name='g',
            email='a@b.com',
            created_at=now,
            updated_at=now,
        )
        data: user.UserRead = user.UserRead(
            id=1,
            username='u',
            role='admin',
            status='active',
            group_id=1,
            group=group_obj,
            profile=profile_obj,
            created_at=now,
            updated_at=now,
        )
        self.assertEqual(data.id, 1)
        self.assertEqual(data.group_name, 'g')
        assert data.profile is not None
        self.assertEqual(data.profile.family_name, 'f')

    def test_group_name_none(self) -> None:
        """Test UserRead instance with no group or profile."""
        now: datetime = datetime.now()
        data: user.UserRead = user.UserRead(
            id=2,
            username='u',
            role='user',
            status='suspended',
            group_id=None,
            group=None,
            profile=None,
            created_at=now,
            updated_at=now,
        )
        self.assertIsNone(data.group_name)

    def test_missing(self) -> None:
        """Test validation error when fields are missing."""
        with self.assertRaises(ValidationError):
            cast(Any, user.UserRead)()


if __name__ == '__main__':
    unittest.main()
