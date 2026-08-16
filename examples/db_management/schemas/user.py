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


# Shared profile fields keep account creation and profile editing consistent.
class UserProfileBase(BaseModel):
    """Define the profile fields shared by user requests and responses.

    Attributes:
        family_name: User's family name.
        middle_name: Optional middle name.
        given_name: User's given name.
        email: User's email address.
        mobile_number: Optional mobile telephone number.
    """

    family_name:   str
    middle_name:   str | None = None
    given_name:    str
    email:         EmailStr
    mobile_number: str | None = None


class UserProfileRead(UserProfileBase):
    """Represent a user profile returned from persistent storage.

    Attributes:
        created_at: Time at which the profile was created.
        updated_at: Time at which the profile was last changed.
    """

    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserProfileUpdate(BaseModel):
    """Define a partial update for a user's profile.

    Attributes:
        user_id: Identifier of the user whose profile changes.
        family_name: Replacement family name.
        middle_name: Replacement optional middle name.
        given_name: Replacement given name.
        email: Replacement email address.
        mobile_number: Replacement optional mobile telephone number.
    """

    user_id:       int
    family_name:   str | None = None
    middle_name:   str | None = None
    given_name:    str | None = None
    email:         EmailStr | None = None
    mobile_number: str | None = None


class UserCreate(BaseModel):
    """Define input used by an administrator to create a user.

    Attributes:
        username: Unique account username.
        password: Initial account password.
        role: Role assigned to the new user.
        group_id: Optional group assigned to the new user.
        profile: Optional profile created with the user account.
    """

    username:  str
    password:  str
    role:      str = 'user'
    group_id:  int | None
    # Administrators may create the optional profile with the account.
    profile:   UserProfileBase | None = None


class UserDelete(BaseModel):
    """Identify a user account that should be deleted.

    Attributes:
        user_id: Identifier of the user to delete.
    """

    user_id: int


class UserSignup(BaseModel):
    """Define a public account-registration request.

    Attributes:
        username: Requested account username.
        password: Requested account password.
        profile: Required personal profile for the new account.
        accepted_terms: Whether the general terms were accepted.
        terms_version: Version of the general terms accepted.
        privacy_version: Version of the privacy notice accepted.
        notification_consent: Whether notifications were accepted.
        ai_terms_accepted: Whether AI-specific terms were accepted.
        ai_terms_version: Version of the AI-specific terms accepted.
    """

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
    """Define an administrator approval of a pending signup.

    Attributes:
        user_id: Identifier of the pending user to approve.
        group_id: Optional group assigned when approval succeeds.
    """

    user_id: int
    group_id: int | None = None


class AdminUserApproval(BaseModel):
    """Define an administrator decision for an email-verified signup.

    Attributes:
        decision: Whether the pending account is approved or rejected.
        group_id: Optional group assigned when the account is approved.
        note: Optional audit note explaining the decision.
    """

    decision: Literal['approved', 'rejected'] = 'approved'
    group_id: int | None = None
    note: str | None = None


class UserRead(BaseModel):
    """Represent a user returned from persistent storage.

    Attributes:
        id: Database identifier of the user.
        username: Unique account username.
        role: Role assigned to the user.
        status: Current account lifecycle status.
        email_verified_at: Time at which the email address was verified.
        group_id: Optional assigned group identifier.
        group: Optional assigned group details.
        profile: Optional personal profile details.
        created_at: Time at which the user was created.
        updated_at: Time at which the user was last changed.
    """

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
        """Return the user's assigned group name.

        Returns:
            Assigned group name, or ``None`` when no group is assigned.
        """
        return self.group.name if self.group else None

    model_config = ConfigDict(from_attributes=True)


class PendingUserReviewRead(UserRead):
    """Represent a pending signup in an administrator review queue.

    Attributes:
        email: Email address associated with the pending account.
        terms_version: General terms version accepted during registration.
        privacy_version: Privacy notice version accepted during registration.
        ai_terms_version: AI-specific terms version accepted at registration.
        notification_consent: Whether notification consent was supplied.
        provider: Authentication provider that created the account.
    """

    email: EmailStr | None = None
    terms_version: str | None = None
    privacy_version: str | None = None
    ai_terms_version: str | None = None
    notification_consent: bool | None = None
    provider: str = 'password'


class UpdateUsername(BaseModel):
    """Define a username change identified by the current username.

    Attributes:
        old_username: Current username that must match an existing account.
        new_username: Replacement username.
    """

    old_username: str
    new_username: str


class UpdateUsernameById(BaseModel):
    """Define a username change identified by user identifier.

    Attributes:
        user_id: Identifier of the user to update.
        new_username: Replacement username.
    """

    user_id: int
    new_username: str


class UpdatePassword(BaseModel):
    """Define a password change identified by username.

    Attributes:
        username: Username of the account to update.
        new_password: Replacement account password.
    """

    username: str
    new_password: str


class UpdatePasswordById(BaseModel):
    """Define a password change identified by user identifier.

    Attributes:
        user_id: Identifier of the user to update.
        new_password: Replacement account password.
    """

    user_id: int
    new_password: str


class UpdateMyPassword(BaseModel):
    """Define a self-service password change.

    Attributes:
        old_password: Current password required to authorise the change.
        new_password: Replacement account password.
    """

    old_password: str
    new_password: str


class SetUserStatus(BaseModel):
    """Define an account-status change.

    Attributes:
        user_id: Identifier of the user to update.
        status: Replacement lifecycle status for the account.
    """

    user_id: int
    status: UserStatus


class UpdateUserRole(BaseModel):
    """Define a role change for a user.

    Attributes:
        user_id: Identifier of the user to update.
        new_role: Replacement role for the user.
    """

    user_id: int
    new_role: str


class UpdateUserGroup(BaseModel):
    """Define a group assignment change for a user.

    Attributes:
        user_id: Identifier of the user to update.
        new_group_id: Identifier of the group to assign.
    """

    user_id: int
    new_group_id: int
