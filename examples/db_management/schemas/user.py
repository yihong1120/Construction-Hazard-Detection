from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from pydantic import computed_field
from pydantic import ConfigDict
from pydantic import EmailStr

from examples.db_management.schemas.group import GroupRead

UserStatus = Literal[
    'active',
    'email_unverified',
    'pending_admin_approval',
    'rejected',
    'suspended',
]


# ──────────────────────────────────────────────────────────
#  共用：User-Profile Schemas
# ──────────────────────────────────────────────────────────
class UserProfileBase(BaseModel):
    """Shared user profile fields."""

    family_name:   str
    middle_name:   str | None = None
    given_name:    str
    email:         EmailStr
    mobile_number: str | None = None


class UserProfileRead(UserProfileBase):
    """User profile response payload."""

    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserProfileUpdate(BaseModel):
    """User profile update payload."""

    user_id:       int
    family_name:   str | None = None
    middle_name:   str | None = None
    given_name:    str | None = None
    email:         EmailStr | None = None
    mobile_number: str | None = None


class UserCreate(BaseModel):
    """Schema representing user creation details."""

    username:  str
    password:  str
    role:      str = 'user'
    group_id:  int | None
    # 新增：可直接帶 profile 建檔（選填）
    profile:   UserProfileBase | None = None


class UserSignup(BaseModel):
    """Schema representing a public account signup request."""

    username: str
    password: str
    profile: UserProfileBase
    accepted_terms: bool = False
    terms_version: str | None = None
    privacy_version: str | None = None
    notification_consent: bool = False
    ai_terms_accepted: bool = False
    ai_terms_version: str | None = None


class ApproveUserSignup(BaseModel):
    """Schema representing admin approval of a pending signup."""

    user_id: int
    group_id: int | None = None


class AdminUserApproval(BaseModel):
    """Schema for approving or rejecting an email-verified signup."""

    decision: Literal['approved', 'rejected'] = 'approved'
    group_id: int | None = None
    note: str | None = None


class UserRead(BaseModel):
    """Schema representing detailed user information."""

    id: int
    username: str
    role: str
    status: UserStatus
    email_verified_at: datetime | None = None
    group_id: int | None
    group: GroupRead | None
    profile:    UserProfileRead | None
    created_at: datetime
    updated_at: datetime

    @computed_field
    def group_name(self) -> str | None:
        """Computed property returning the user's group name.

        Returns:
            Optional[str]: Name of the group if present, otherwise None.
        """
        return self.group.name if self.group else None

    model_config = ConfigDict(from_attributes=True)


class PendingUserReviewRead(UserRead):
    """Admin review row for an email-verified signup."""

    email: EmailStr | None = None
    terms_version: str | None = None
    privacy_version: str | None = None
    ai_terms_version: str | None = None
    notification_consent: bool | None = None
    provider: str = 'password'


class UpdateUsername(BaseModel):
    """Schema for updating a user's username."""

    old_username: str
    new_username: str


class UpdateUsernameById(BaseModel):
    """Schema for updating username based on user ID."""

    user_id: int
    new_username: str


class UpdatePassword(BaseModel):
    """Schema for updating a user's password by username."""

    username: str
    new_password: str


class UpdatePasswordById(BaseModel):
    """Schema for updating a user's password using user ID."""

    user_id: int
    new_password: str


class UpdateMyPassword(BaseModel):
    """Schema allowing a user to update their own password."""

    old_password: str
    new_password: str


class SetUserStatus(BaseModel):
    """Schema for setting a user's status."""

    user_id: int
    status: UserStatus


class UpdateUserRole(BaseModel):
    """Schema for updating a user's role."""

    user_id: int
    new_role: str


class UpdateUserGroup(BaseModel):
    """Schema for changing a user's assigned group."""

    user_id: int
    new_group_id: int
