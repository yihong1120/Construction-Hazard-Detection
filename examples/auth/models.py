from __future__ import annotations

import asyncio
from datetime import datetime
from datetime import timezone

from pwdlib import PasswordHash
from pwdlib.exceptions import UnknownHashError
from sqlalchemy import Boolean
from sqlalchemy import CheckConstraint
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import Text
from sqlalchemy import text
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from examples.auth.database import Base

password_hash = PasswordHash.recommended()


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for ORM-side defaults."""
    return datetime.now(timezone.utc)


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

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
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
        return f"<Group id={self.id} name={self.name}>"


# -------------------------------------------------------
#  Association Table: Many-to-Many Relationship between User and Site
# -------------------------------------------------------
user_sites_table: Table = Table(
    'user_sites',
    Base.metadata,
    Column('user_id', ForeignKey('users.id'), primary_key=True),
    Column('site_id', ForeignKey('sites.id'), primary_key=True),
)


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


# -------------------------------------------------------
#  User Model
# -------------------------------------------------------
USER_STATUS_ACTIVE = 'active'
USER_STATUS_EMAIL_UNVERIFIED = 'email_unverified'
USER_STATUS_PENDING_ADMIN_APPROVAL = 'pending_admin_approval'
USER_STATUS_REJECTED = 'rejected'
USER_STATUS_SUSPENDED = 'suspended'
# Backward-compatible aliases for older call sites.
USER_STATUS_PENDING = USER_STATUS_PENDING_ADMIN_APPROVAL
USER_STATUS_INACTIVE = USER_STATUS_SUSPENDED
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

    # 一對一
    user: Mapped[User] = relationship(
        'User', back_populates='profile', uselist=False,
    )


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


class Notification(Base):
    """In-app notification record for notification center history."""

    __tablename__ = 'notifications'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )
    type: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    deep_link: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_read: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
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
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )
    device_token_encrypted: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    device_token_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        index=True,
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
        index=True,
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

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
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
        index=True,
    )
    violation_type_codes: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    is_flagged: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True,
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
        String(20), nullable=True, default=None, index=True,
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


class ViolationFeedback(Base):
    """
    Structured user feedback for a stored violation record.

    Feedback is collected in a pending state first so it can be reviewed before
    becoming training data.
    """

    __tablename__ = 'violation_feedback'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    violation_id: Mapped[int] = mapped_column(
        ForeignKey('violations.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
    )
    anonymous_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    target_detection_id: Mapped[str | None] = mapped_column(
        String(120), nullable=True,
    )
    feedback_type: Mapped[str] = mapped_column(
        String(30), nullable=False, index=True,
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
        String(20), nullable=False, default='pending', index=True,
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

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    violation_id: Mapped[int] = mapped_column(
        ForeignKey('violations.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
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
