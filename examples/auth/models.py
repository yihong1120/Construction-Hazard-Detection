from __future__ import annotations

import asyncio
from datetime import datetime
from datetime import timezone
from uuid import UUID
from uuid import uuid4

from pwdlib import PasswordHash
from pwdlib.exceptions import UnknownHashError
from sqlalchemy import BigInteger
from sqlalchemy import Boolean
from sqlalchemy import CheckConstraint
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import event
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Index
from sqlalchemy import inspect
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import Text
from sqlalchemy import text
from sqlalchemy import UniqueConstraint
from sqlalchemy import Uuid
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from examples.auth.database import Base


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for ORM-side defaults."""
    return datetime.now(timezone.utc)


TENANT_STATUS_ACTIVE = 'active'
TENANT_STATUS_DISABLED = 'disabled'
TENANT_STATUS_VALUES = (TENANT_STATUS_ACTIVE, TENANT_STATUS_DISABLED)
DEPLOYMENT_STATUS_ACTIVE = 'active'
DEPLOYMENT_STATUS_REVOKED = 'revoked'
DEPLOYMENT_STATUS_VALUES = (
    DEPLOYMENT_STATUS_ACTIVE,
    DEPLOYMENT_STATUS_REVOKED,
)


class Tenant(Base):
    """Formal tenant boundary used by every authenticated account."""

    __tablename__ = 'tenants'
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'disabled')",
            name='chk_tenants_status',
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid4,
    )
    name: Mapped[str] = mapped_column(String(160), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=TENANT_STATUS_ACTIVE,
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
        onupdate=utc_now,
    )

    deployments: Mapped[list[Deployment]] = relationship(
        'Deployment', back_populates='tenant', cascade='all, delete-orphan',
    )
    users: Mapped[list[User]] = relationship('User', back_populates='tenant')


class Deployment(Base):
    """One canonical API deployment published through the signed Registry."""

    __tablename__ = 'deployments'
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'revoked')",
            name='chk_deployments_status',
        ),
        CheckConstraint(
            'config_revision >= 1',
            name='chk_deployments_config_revision',
        ),
        UniqueConstraint('api_base_url', name='uq_deployments_api_base_url'),
        Index('idx_deployments_tenant', 'tenant_id'),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid4,
    )
    tenant_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey('tenants.id', ondelete='RESTRICT'),
        nullable=False,
    )
    api_base_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    config_revision: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1,
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=DEPLOYMENT_STATUS_ACTIVE,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
        onupdate=utc_now,
    )

    tenant: Mapped[Tenant] = relationship(
        'Tenant', back_populates='deployments', lazy='joined',
    )


class DeploymentEnrollmentCode(Base):
    """One-time native-device enrollment verifier for a deployment.

    The raw enrollment code is deliberately never represented by this model.
    Only its HMAC-SHA256 verifier is persisted, so a database backup cannot be
    used as a list of usable company activation codes.
    """

    __tablename__ = 'deployment_enrollment_codes'
    __table_args__ = (
        CheckConstraint(
            'expires_at > created_at',
            name='chk_deployment_enrollment_codes_expiry',
        ),
        UniqueConstraint(
            'code_verifier_hash',
            name='uq_deployment_enrollment_codes_verifier',
        ),
        Index(
            'uq_deployment_enrollment_codes_public_id',
            'public_id',
            unique=True,
        ),
        Index(
            'idx_deployment_enrollment_codes_deployment',
            'deployment_id',
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    # The internal bigint key remains private.  Management clients address an
    # invitation by this opaque UUID so database row counts are not exposed.
    public_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
        default=uuid4,
    )
    deployment_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey('deployments.id', ondelete='RESTRICT'),
        nullable=False,
    )
    code_verifier_hash: Mapped[str] = mapped_column(
        String(64), nullable=False,
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    redeemed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
    )
    created_by: Mapped[str] = mapped_column(String(160), nullable=False)


class DeploymentEnrollmentCodeAuditLog(Base):
    """Immutable, non-secret audit event for an enrollment invitation."""

    __tablename__ = 'deployment_enrollment_code_audit_logs'
    __table_args__ = (
        CheckConstraint(
            "action IN ('created', 'revoked')",
            name='chk_deployment_enrollment_code_audit_action',
        ),
        Index(
            'idx_deployment_enrollment_code_audit_code',
            'enrollment_code_id',
            text('created_at DESC'),
        ),
        Index(
            'idx_deployment_enrollment_code_audit_deployment',
            'deployment_id',
            text('created_at DESC'),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    enrollment_code_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey('deployment_enrollment_codes.id', ondelete='RESTRICT'),
        nullable=False,
    )
    deployment_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey('deployments.id', ondelete='RESTRICT'),
        nullable=False,
    )
    tenant_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey('tenants.id', ondelete='RESTRICT'),
        nullable=False,
    )
    actor_user_id: Mapped[int | None] = mapped_column(
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
    )
    action: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
    )


@event.listens_for(Deployment, 'before_insert')
def _canonicalise_new_deployment(
    _mapper: object,
    _connection: object,
    target: Deployment,
) -> None:
    """Reject non-canonical deployment origins before they reach storage."""
    from examples.auth.deployment_context import canonical_api_base_url

    target.api_base_url = canonical_api_base_url(target.api_base_url)
    if target.config_revision < 1:
        raise ValueError('config_revision must start at 1')


@event.listens_for(Deployment, 'before_update')
def _revision_on_deployment_change(
    _mapper: object,
    _connection: object,
    target: Deployment,
) -> None:
    """Invalidate tokens whenever the deployment security contract changes."""
    from examples.auth.deployment_context import canonical_api_base_url

    target.api_base_url = canonical_api_base_url(target.api_base_url)
    state = inspect(target)
    changed = any(
        state.attrs[name].history.has_changes()
        for name in ('tenant_id', 'api_base_url', 'status')
    )
    if changed and not state.attrs.config_revision.history.has_changes():
        target.config_revision += 1


user_sites_table: Table = Table(
    'user_sites',
    Base.metadata,
    Column('user_id', ForeignKey('users.id'), primary_key=True),
    Column('site_id', ForeignKey('sites.id'), primary_key=True),
)


site_groups_table: Table = Table(
    'site_groups',
    Base.metadata,
    Column(
        'site_id', ForeignKey('sites.id', ondelete='CASCADE'),
        primary_key=True,
    ),
    Column(
        'group_id', ForeignKey('group_info.id', ondelete='CASCADE'),
        primary_key=True,
    ),
    Column(
        'created_at',
        DateTime(timezone=True),
        server_default=text('CURRENT_TIMESTAMP'),
    ),
)


# -------------------------------------------------------
#  Site Model
# -------------------------------------------------------
password_hash = PasswordHash.recommended()


# -------------------------------------------------------
#  Feature Model
# -------------------------------------------------------


class Feature(Base):
    """
    Represents a feature in the system, such as safety detection
    capabilities. This model is linked to the Group model
    through a many-to-many relationship.

    Attributes:
        id (int): Primary key.
        feature_name (str): Unique name of the feature.
        description (str | None): Description of the feature.
        created_at (datetime): Timestamp of creation.
        updated_at (datetime): Timestamp of last update.
        groups (list[Group]): Groups that have access to this feature.

    Methods:
        __repr__(): String representation of the Feature object.
    """

    __tablename__ = 'features'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    feature_name: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False,
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text('CURRENT_TIMESTAMP'),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    # Many-to-many relationship to Group
    # related to this feature
    groups: Mapped[list[Group]] = relationship(
        'Group',
        secondary='group_features',
        back_populates='features',
        lazy='joined',
    )

    def __repr__(self) -> str:
        """Perform repr.

        Returns:
            The callable result.
        """
        return f"<Feature id={self.id} name={self.feature_name}>"


# -------------------------------------------------------
#  group_features
# -------------------------------------------------------
group_features_table: Table = Table(
    'group_features',
    Base.metadata,
    Column(
        'group_id', ForeignKey(
            'group_info.id',
            ondelete='CASCADE',
        ), primary_key=True,
    ),
    Column(
        'feature_id', ForeignKey(
            'features.id',
            ondelete='CASCADE',
        ), primary_key=True,
    ),
    Column(
        'created_at',
        DateTime(timezone=True),
        server_default=text('CURRENT_TIMESTAMP'),
    ),
)

# -------------------------------------------------------
#  Group Model
# -------------------------------------------------------


class Group(Base):
    """
    Represents a group of users, including their access
    permissions and associated construction sites.

    Attributes:
        id (int): Primary key.
        name (str): Name of the group.
        uniform_number (str): Unique identifier for the group.
        max_allowed_streams (int): Maximum number of streams allowed.
        created_at (datetime): Timestamp of creation.
        updated_at (datetime): Timestamp of last update.
        sites (list[Site]): Sites associated with this group.
        users (list[User]): Users belonging to this group.
        features (list[Feature]): Features available to this group.

    Methods:
        __repr__(): String representation of the Group object.
    """

    __tablename__ = 'group_info'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)

    uniform_number: Mapped[str] = mapped_column(
        String(8), unique=True, nullable=False, comment='統一編號',
    )

    max_allowed_streams: Mapped[int] = mapped_column(
        Integer, nullable=False, default=8,
    )

    sites: Mapped[list[Site]] = relationship(
        'Site',
        secondary='site_groups',
        back_populates='groups',
        lazy='joined',
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text('CURRENT_TIMESTAMP'),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    # One-to-many relationship to User
    users: Mapped[list[User]] = relationship(
        'User',
        back_populates='group',
    )

    # Many-to-many relationship to Feature
    features: Mapped[list[Feature]] = relationship(
        'Feature',
        secondary=group_features_table,
        back_populates='groups',
        lazy='joined',
    )

    stream_configs: Mapped[list[StreamConfig]] = relationship(
        'StreamConfig', back_populates='group', cascade='all, delete-orphan',
    )

    def __repr__(self) -> str:
        """Perform repr.

        Returns:
            The callable result.
        """
        return f"<Group id={self.id} name={self.name}>"


# -------------------------------------------------------
#  Association Table: Many-to-Many Relationship between User and Site
# -------------------------------------------------------


# -------------------------------------------------------
#  User Model
# -------------------------------------------------------
USER_STATUS_ACTIVE = 'active'
USER_STATUS_EMAIL_UNVERIFIED = 'email_unverified'
USER_STATUS_PENDING_ADMIN_APPROVAL = 'pending_admin_approval'
USER_STATUS_REJECTED = 'rejected'
USER_STATUS_SUSPENDED = 'suspended'
USER_STATUS_VALUES = (
    USER_STATUS_ACTIVE,
    USER_STATUS_EMAIL_UNVERIFIED,
    USER_STATUS_PENDING_ADMIN_APPROVAL,
    USER_STATUS_REJECTED,
    USER_STATUS_SUSPENDED,
)
LEGAL_DOCUMENT_TYPE_TERMS = 'terms'
LEGAL_DOCUMENT_TYPE_PRIVACY = 'privacy'
LEGAL_DOCUMENT_TYPE_AI_TERMS = 'ai_terms'
LEGAL_DOCUMENT_TYPES = (
    LEGAL_DOCUMENT_TYPE_TERMS,
    LEGAL_DOCUMENT_TYPE_PRIVACY,
    LEGAL_DOCUMENT_TYPE_AI_TERMS,
)


class UserProfile(Base):
    """Stores contact and profile details for a user account."""

    __tablename__ = 'user_profiles'

    user_id: Mapped[int] = mapped_column(
        ForeignKey('users.id', ondelete='CASCADE'), primary_key=True,
    )
    family_name: Mapped[str] = mapped_column(String(50), nullable=False)
    middle_name: Mapped[str | None] = mapped_column(String(50))
    given_name: Mapped[str] = mapped_column(String(50), nullable=False)
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False,
    )
    mobile_number: Mapped[str | None] = mapped_column(String(20), unique=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text('CURRENT_TIMESTAMP'),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    # One-to-one relationship.
    user: Mapped[User] = relationship(
        'User', back_populates='profile', uselist=False,
    )


@event.listens_for(UserProfile.email, 'set', retval=True)
def _normalise_profile_email(
    _target: UserProfile,
    value: str | None,
    _oldvalue: object,
    _initiator: object,
) -> str | None:
    """Store e-mail addresses in one canonical form for indexed lookups."""
    return value.strip().lower() if isinstance(value, str) else value


class User(Base):
    """
    Represents a user in the system, including login credentials,
    role-based access, and relationships to assigned construction sites.

    Attributes:
        id (int): Primary key.
        username (str): Unique login identifier.
        password_hash (str): Hashed user password.
        role (str): Access level (e.g., admin, user, guest).
        status (str): Account lifecycle status.
        created_at (datetime): Timestamp of creation.
        updated_at (datetime): Timestamp of last update.
        sites (list[Site]): Sites the user has access to.
    """

    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(
        String(80), unique=True, nullable=False,
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(
        String(20), default='user', nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20), default=USER_STATUS_ACTIVE, nullable=False,
    )
    email_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    tenant_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey('tenants.id', ondelete='RESTRICT'),
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    # group_info.id
    group_id: Mapped[int | None] = mapped_column(
        ForeignKey('group_info.id', ondelete='SET NULL'),
        nullable=True,
    )

    # One-to-many relationship to Group
    # This is a foreign key to the group_info table
    group: Mapped[Group | None] = relationship(
        'Group',
        back_populates='users',
        lazy='joined',
    )

    tenant: Mapped[Tenant] = relationship('Tenant', back_populates='users')

    # Many-to-many relationship to Site
    # This is an association table linking users to sites
    sites: Mapped[list[Site]] = relationship(
        'Site', secondary=user_sites_table, back_populates='users',
    )

    profile: Mapped[UserProfile] = relationship(
        'UserProfile',
        back_populates='user',
        uselist=False,
        lazy='selectin',
        cascade='all, delete-orphan',
        passive_deletes=True,
    )

    identities: Mapped[list[UserIdentity]] = relationship(
        'UserIdentity',
        back_populates='user',
        cascade='all, delete-orphan',
        passive_deletes=True,
    )

    notification_preferences: Mapped[list[SiteNotificationPreference]] = (
        relationship(
            'SiteNotificationPreference',
            back_populates='user',
            cascade='all, delete-orphan',
        )
    )

    notifications: Mapped[list[Notification]] = relationship(
        'Notification',
        back_populates='user',
        cascade='all, delete-orphan',
    )

    fcm_device_tokens: Mapped[list[FcmDeviceToken]] = relationship(
        'FcmDeviceToken',
        back_populates='user',
        cascade='all, delete-orphan',
    )

    consents: Mapped[list[UserConsent]] = relationship(
        'UserConsent',
        back_populates='user',
        cascade='all, delete-orphan',
    )

    def set_password(self, password: str) -> None:
        """
        Hash and store the user's password securely.

        Args:
            password (str): The plain-text password to be hashed.
        """
        self.password_hash = password_hash.hash(password)

    async def check_password(self, password: str) -> bool:
        """
        Verify whether a given password matches the stored hash.

        This is executed in a thread-safe, asynchronous manner.

        Args:
            password (str): The plain-text password to verify.

        Returns:
            bool: True if the password matches, otherwise False.
        """
        try:
            return await asyncio.to_thread(
                password_hash.verify,
                password,
                str(self.password_hash),
            )
        except UnknownHashError:
            return False

    def to_dict(self) -> dict:
        """
        Convert user attributes to a dictionary (e.g., for Redis caching).

        Returns:
            dict: A serialisable dictionary of user information.
        """
        return {
            'id': self.id,
            'username': self.username,
            'role': self.role,
            'status': self.status,
            'tenant_id': str(self.tenant_id),
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }


class UserIdentity(Base):
    """External identity provider account linked to a local user."""

    __tablename__ = 'user_identities'
    __table_args__ = (
        UniqueConstraint(
            'provider',
            'provider_user_id',
            name='uq_user_identities_provider_user_id',
        ),
        UniqueConstraint(
            'user_id',
            'provider',
            name='uq_user_identities_user_provider',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(20), nullable=False)
    provider_user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email_verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )
    display_name: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )
    raw_profile: Mapped[dict[str, object] | None] = mapped_column(
        JSON, nullable=True,
    )
    raw_email_is_private: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )
    linked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text('CURRENT_TIMESTAMP'),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    user: Mapped[User] = relationship(
        'User',
        back_populates='identities',
    )


class LegalDocument(Base):
    """Versioned legal document content shown to users during signup."""

    __tablename__ = 'legal_documents'
    __table_args__ = (
        UniqueConstraint(
            'type',
            'version',
            'locale',
            name='uq_legal_documents_type_version_locale',
        ),
        CheckConstraint(
            "type IN ('terms', 'privacy', 'ai_terms')",
            name='chk_legal_documents_type',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    type: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(40), nullable=False)
    locale: Mapped[str] = mapped_column(
        String(20), nullable=False, default='zh-TW', index=True,
    )
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    effective_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, index=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
    )


class UserConsent(Base):
    """Recorded user consent snapshot for legal and notification terms."""

    __tablename__ = 'user_consents'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )
    terms_version: Mapped[str] = mapped_column(String(40), nullable=False)
    privacy_version: Mapped[str] = mapped_column(String(40), nullable=False)
    ai_terms_version: Mapped[str] = mapped_column(String(40), nullable=False)
    accepted_terms: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )
    notification_consent: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )
    ai_terms_accepted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )
    accepted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, index=True,
    )
    ai_terms_accepted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    notification_consent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    ip_address: Mapped[str | None] = mapped_column(
        String(45), nullable=True,
    )
    user_agent: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )

    user: Mapped[User] = relationship('User', back_populates='consents')


# -------------------------------------------------------
#  site_groups  (Many-to-Many: Site ↔ Group)
# -------------------------------------------------------
class Site(Base):
    """
    Represents a construction site, including its name and associated
    users and safety violations.

    Attributes:
        id (int): Primary key.
        name (str): Name of the site.
        created_at (datetime): Creation timestamp.
        updated_at (datetime): Last update timestamp.
        users (list[User]): Users assigned to this site.
        violations (list[Violation]): Safety violations detected at the site.
    """

    __tablename__ = 'sites'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(
        String(80), unique=True, nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    # Many-to-many relationship to Group
    groups: Mapped[list[Group]] = relationship(
        'Group',
        secondary=site_groups_table,
        back_populates='sites',
        lazy='joined',
    )

    # Many-to-many relationship to User
    # This is an association table linking users to sites
    users: Mapped[list[User]] = relationship(
        'User', secondary=user_sites_table, back_populates='sites',
    )

    notification_preferences: Mapped[list[SiteNotificationPreference]] = (
        relationship(
            'SiteNotificationPreference',
            back_populates='site',
            cascade='all, delete-orphan',
        )
    )

    # One-to-many relationship to Violation
    # This is a foreign key to the violations table
    violations: Mapped[list[Violation]] = relationship(
        'Violation',
        back_populates='site_obj',
    )

    stream_configs: Mapped[list[StreamConfig]] = relationship(
        'StreamConfig', back_populates='site', cascade='all, delete-orphan',
    )


# -------------------------------------------------------
#  StreamConfig
# -------------------------------------------------------
class StreamConfig(Base):
    """
    Represents the configuration for a video stream, including
    detection capabilities and scheduling.

    Attributes:
        id (int): Primary key.
        group_id (int): Foreign key to the group_info table.
        site_id (int): Foreign key to the sites table.
        stream_name (str): Name of the video stream.
        video_url (str): URL of the video stream.
        model_key (str): Key for the detection model.
        detect_no_safety_vest_or_helmet (bool): Detection capability.
        detect_near_machinery_or_vehicle (bool): Detection capability.
        detect_in_restricted_area (bool): Detection capability.
        detect_in_utility_pole_restricted_area (bool): Detection capability.
        detect_machinery_close_to_pole (bool): Detection capability.
        recognition_enabled (bool): Whether recognition processing is enabled.
        expire_date (datetime | None): Expiration date for the configuration.
        work_start_hour (int): Start hour for work scheduling.
        work_end_hour (int): End hour for work scheduling.
        created_at (datetime): Timestamp of creation.
        updated_at (datetime): Timestamp of last update.
    """
    __tablename__ = 'stream_configs'

    id:          Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True,
    )
    group_id:    Mapped[int] = mapped_column(
        ForeignKey('group_info.id', ondelete='CASCADE'), nullable=False,
    )
    site_id:     Mapped[int] = mapped_column(
        ForeignKey('sites.id', ondelete='CASCADE'), nullable=False,
    )

    stream_name: Mapped[str] = mapped_column(String(80), nullable=False)
    video_url:   Mapped[str] = mapped_column(String(255), nullable=False)
    model_key:   Mapped[str] = mapped_column(String(80), nullable=False)

    # Detection capabilities
    # These fields are used to determine the types of violations
    detect_no_safety_vest_or_helmet:        Mapped[bool] = mapped_column(
        Boolean, default=False,
    )
    detect_near_machinery_or_vehicle:       Mapped[bool] = mapped_column(
        Boolean, default=False,
    )
    detect_in_restricted_area:              Mapped[bool] = mapped_column(
        Boolean, default=False,
    )
    detect_in_utility_pole_restricted_area: Mapped[bool] = mapped_column(
        Boolean, default=False,
    )
    detect_machinery_close_to_pole:         Mapped[bool] = mapped_column(
        Boolean, default=False,
    )

    recognition_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=text('TRUE'),
        nullable=False,
    )
    expire_date:        Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    work_start_hour: Mapped[int] = mapped_column(Integer)
    work_end_hour:   Mapped[int] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text('CURRENT_TIMESTAMP'),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    # Foreign key to the group_info table
    group: Mapped[Group] = relationship(
        'Group', back_populates='stream_configs',
    )
    site:  Mapped[Site] = relationship(
        'Site',  back_populates='stream_configs',
    )

    __table_args__ = (
        # Ensure that the combination of site_id and stream_name is unique
        # across the table to prevent duplicate stream configurations
        UniqueConstraint('site_id', 'stream_name', name='uq_sc_site_stream'),
    )


# -------------------------------------------------------
#  Violation Model
# -------------------------------------------------------
class SiteNotificationPreference(Base):
    """Per-user notification preference for a specific site."""

    __tablename__ = 'site_notification_preferences'

    user_id: Mapped[int] = mapped_column(
        ForeignKey('users.id', ondelete='CASCADE'),
        primary_key=True,
    )
    site_id: Mapped[int] = mapped_column(
        ForeignKey('sites.id', ondelete='CASCADE'),
        primary_key=True,
    )
    is_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text('CURRENT_TIMESTAMP'),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    user: Mapped[User] = relationship(
        'User', back_populates='notification_preferences',
    )
    site: Mapped[Site] = relationship(
        'Site', back_populates='notification_preferences',
    )


class Notification(Base):
    """In-app notification record for notification center history."""

    __tablename__ = 'notifications'
    __table_args__ = (
        CheckConstraint(
            "type IN ('signature', 'violation', 'document', 'site_alert', 'system')",
            name='chk_notifications_type',
        ),
        Index(
            'idx_notifications_user_created_id',
            'user_id',
            text('created_at DESC'),
            text('id DESC'),
        ),
        Index(
            'idx_notifications_user_read_created_id',
            'user_id',
            'is_read',
            text('created_at DESC'),
            text('id DESC'),
        ),
        Index(
            'idx_notifications_user_type_created_id',
            'user_id',
            'type',
            text('created_at DESC'),
            text('id DESC'),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
    )
    type: Mapped[str] = mapped_column(String(30), nullable=False)
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    deep_link: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_read: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    metadata_json: Mapped[dict[str, object]] = mapped_column(
        'metadata',
        JSON,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
    )

    user: Mapped[User] = relationship(
        'User',
        back_populates='notifications',
    )


class FcmDeviceToken(Base):
    """Push-notification device token stored as encrypted source of truth."""

    __tablename__ = 'fcm_device_tokens'
    __table_args__ = (
        UniqueConstraint(
            'device_token_hash',
            name='uq_fcm_device_tokens_token_hash',
        ),
        CheckConstraint(
            "platform IN ('android', 'ios', 'web', 'unknown')",
            name='chk_fcm_device_tokens_platform',
        ),
        Index(
            'idx_fcm_device_tokens_user_active',
            'user_id',
            'disabled_at',
        ),
        Index(
            'idx_fcm_device_tokens_user_seen',
            'user_id',
            text('last_seen_at DESC'),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
    )
    device_token_encrypted: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    device_token_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    platform: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default='unknown',
    )
    device_lang: Mapped[str] = mapped_column(String(20), nullable=False)
    permission_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default='unknown',
    )
    app_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    web_vapid_key_available: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )
    web_service_worker_registered: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    last_success_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_failure_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    failure_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    disabled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    user: Mapped[User] = relationship(
        'User',
        back_populates='fcm_device_tokens',
    )


class Violation(Base):
    """
    Represents a safety violation detected at a specific site and time.

    Attributes:
        id (int): Primary key.
        stream_name (str): Name of the video stream or camera.
        detection_time (datetime): Timestamp when violation was detected.
        image_path (str): Path to the saved image of the violation.
        detections_json (str | None): JSON string of detected objects.
        cone_polygon_json (str | None): JSON of safety cone polygon data.
        pole_polygon_json (str | None): JSON of safety pole polygon data.
        warnings_json (str | None): JSON of warning content (translated).
        stream_config_id (int | None): Stable camera configuration identifier.
        violation_type_codes (list[str]): Canonical violation type codes.
        created_at (datetime): Time of record creation.
        site (str): Name of the related site (used for linkage).
        site_obj (Site): ORM relationship to the site object.
    """

    __tablename__ = 'violations'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    stream_name: Mapped[str] = mapped_column(String(80), nullable=False)
    detection_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
    )
    image_path: Mapped[str] = mapped_column(String(255), nullable=False)

    # Optional JSON fields for detection and warning results
    detections_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    cone_polygon_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    pole_polygon_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    warnings_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    stream_config_id: Mapped[int | None] = mapped_column(
        ForeignKey('stream_configs.id', ondelete='SET NULL'),
        nullable=True,
    )
    violation_type_codes: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    is_flagged: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )
    flag_reason: Mapped[str | None] = mapped_column(
        String(120), nullable=True,
    )
    flagged_by: Mapped[int | None] = mapped_column(
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
    )
    flagged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    review_status: Mapped[str | None] = mapped_column(
        String(20), nullable=True, default=None,
    )
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[int | None] = mapped_column(
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True, default=utc_now,
    )

    # Foreign key: name of the associated site
    site: Mapped[str] = mapped_column(
        String(80),
        ForeignKey('sites.name', ondelete='CASCADE'),
        nullable=False,
    )

    # ORM relationship to the actual Site object
    site_obj: Mapped[Site] = relationship(
        'Site',
        back_populates='violations',
    )

    __table_args__ = (
        CheckConstraint(
            "review_status IN ('pending', 'resolved', 'dismissed')",
            name='chk_vio_review_status',
        ),
        # Match the list keyset ordering exactly.  The trailing primary key
        # keeps timestamp ties index-only and cursor seeks deterministic.
        Index(
            'idx_vio_site_detection_id',
            'site',
            detection_time.desc(),
            id.desc(),
        ),
        Index(
            'idx_vio_stream_config_detection_id',
            'stream_config_id',
            detection_time.desc(),
            id.desc(),
        ),
        Index('idx_vio_flagged_status', 'is_flagged', 'review_status'),
        Index('idx_vio_reviewed_at', 'reviewed_at'),
    )


class SiteMediaCleanupJob(Base):
    """Durable post-commit cleanup task for site-owned evidence files."""

    __tablename__ = 'site_media_cleanup_jobs'
    __table_args__ = (
        Index(
            'idx_site_media_cleanup_jobs_pending',
            'completed_at',
            'lease_expires_at',
            'id',
            postgresql_where=text('completed_at IS NULL'),
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    path: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    lease_token: Mapped[str | None] = mapped_column(String(36), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class ViolationFeedback(Base):
    """
    Structured user feedback for a stored violation record.

    Feedback is collected in a pending state first so it can be reviewed before
    becoming training data.
    """

    __tablename__ = 'violation_feedback'
    __table_args__ = (
        CheckConstraint(
            'feedback_type IN ('
            "'false_positive', 'false_negative', 'wrong_class', 'bad_bbox'"
            ')',
            name='chk_vf_feedback_type',
        ),
        CheckConstraint(
            "status IN ('pending', 'reviewed', 'accepted', 'rejected')",
            name='chk_vf_status',
        ),
        Index('idx_vf_violation_created', 'violation_id', 'created_at'),
        Index('idx_vf_type_status', 'feedback_type', 'status'),
        Index('idx_vf_user_created', 'user_id', 'created_at'),
        Index('idx_vf_status_created', 'status', 'created_at'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    violation_id: Mapped[int] = mapped_column(
        ForeignKey('violations.id', ondelete='CASCADE'),
        nullable=False,
    )
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
    )
    anonymous_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    target_detection_id: Mapped[str | None] = mapped_column(
        String(120), nullable=True,
    )
    feedback_type: Mapped[str] = mapped_column(
        String(30), nullable=False,
    )
    original_label: Mapped[str | None] = mapped_column(
        String(120), nullable=True,
    )
    corrected_label: Mapped[str | None] = mapped_column(
        String(120), nullable=True,
    )
    original_bbox: Mapped[list[float] | None] = mapped_column(
        JSON, nullable=True,
    )
    corrected_bbox: Mapped[list[float] | None] = mapped_column(
        JSON, nullable=True,
    )
    model_version: Mapped[str | None] = mapped_column(
        String(120), nullable=True,
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default='pending',
    )
    reviewer_id: Mapped[int | None] = mapped_column(
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
    )


class ViolationReviewAuditLog(Base):
    """Audit trail for review status changes on violation records."""

    __tablename__ = 'violation_review_audit_logs'
    __table_args__ = (
        CheckConstraint(
            "new_status IN ('pending', 'resolved', 'dismissed')",
            name='chk_vral_new_status',
        ),
        Index('idx_vral_violation_time', 'violation_id', 'reviewed_at'),
        Index('idx_vral_reviewer_time', 'reviewed_by', 'reviewed_at'),
        Index('idx_vral_action_time', 'action', 'reviewed_at'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    violation_id: Mapped[int] = mapped_column(
        ForeignKey('violations.id', ondelete='CASCADE'),
        nullable=False,
    )
    action: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        default='review_status_changed',
    )
    old_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    new_status: Mapped[str] = mapped_column(String(20), nullable=False)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    flagged_reason: Mapped[str | None] = mapped_column(
        String(120), nullable=True,
    )
    reviewed_by: Mapped[int | None] = mapped_column(
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
    )
    reviewed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
    )
