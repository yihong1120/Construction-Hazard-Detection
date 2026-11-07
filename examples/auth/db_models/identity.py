from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Boolean
from sqlalchemy import CheckConstraint
from sqlalchemy import DateTime
from sqlalchemy import event
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import text
from sqlalchemy import UniqueConstraint
from sqlalchemy import Uuid
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from examples.auth.database import Base
from examples.auth.db_models.associations import group_features_table
from examples.auth.db_models.associations import user_sites_table
from examples.auth.db_models.shared import USER_STATUS_ACTIVE
from examples.auth.db_models.shared import utc_now

if TYPE_CHECKING:
    from examples.auth.db_models.notifications import FcmDeviceToken
    from examples.auth.db_models.notifications import Notification
    from examples.auth.db_models.notifications import (
        SiteNotificationPreference,
    )
    from examples.auth.db_models.sites import Site
    from examples.auth.db_models.sites import StreamConfig
    from examples.auth.db_models.tenancy import Tenant


class Feature(Base):
    """Represents a feature in the system, such as safety detection
    capabilities. This model is linked to the Group model through a many-to-
    many relationship.

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
        String(50),
        unique=True,
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text('CURRENT_TIMESTAMP'),
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


class Group(Base):
    """Represents a group of users, including their access permissions and
    associated construction sites.

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
        String(8),
        unique=True,
        nullable=False,
        comment='統一編號',
    )

    max_allowed_streams: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=8,
    )

    sites: Mapped[list[Site]] = relationship(
        'Site',
        secondary='site_groups',
        back_populates='groups',
        lazy='joined',
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text('CURRENT_TIMESTAMP'),
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
        'StreamConfig',
        back_populates='group',
        cascade='all, delete-orphan',
    )

    def __repr__(self) -> str:
        return f"<Group id={self.id} name={self.name}>"


class UserProfile(Base):
    """Stores contact and profile details for a user account."""

    __tablename__ = 'user_profiles'

    user_id: Mapped[int] = mapped_column(
        ForeignKey('users.id', ondelete='CASCADE'),
        primary_key=True,
    )
    family_name: Mapped[str] = mapped_column(String(50), nullable=False)
    middle_name: Mapped[str | None] = mapped_column(String(50))
    given_name: Mapped[str] = mapped_column(String(50), nullable=False)
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
    )
    mobile_number: Mapped[str | None] = mapped_column(String(20), unique=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text('CURRENT_TIMESTAMP'),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    # One-to-one relationship.
    user: Mapped[User] = relationship(
        'User',
        back_populates='profile',
        uselist=False,
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
    """Represents a user in the system, including login credentials, role-based
    access, and relationships to assigned construction sites.

    Attributes:
        id (int): Primary key.
        username (str): Unique login identifier.
        role (str): Access level (e.g., admin, user, guest).
        status (str): Account lifecycle status.
        created_at (datetime): Timestamp of creation.
        updated_at (datetime): Timestamp of last update.
        sites (list[Site]): Sites the user has access to.
    """

    __tablename__ = 'users'
    __table_args__ = (
        CheckConstraint(
            "NOT deletion_pending OR status='suspended'",
            name='user_pending_deletion_suspended',
        ),
    )
    state_revision: Mapped[int] = mapped_column(
        Integer,
        default=1,
        server_default=text('1'),
        nullable=False,
    )
    __mapper_args__ = {'version_id_col': state_revision}
    deletion_pending: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=text('false'),
        nullable=False,
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(
        String(80),
        unique=True,
        nullable=False,
    )
    role: Mapped[str] = mapped_column(
        String(20),
        default='user',
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default=USER_STATUS_ACTIVE,
        nullable=False,
    )
    email_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    tenant_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey('tenants.id', ondelete='RESTRICT'),
        nullable=False,
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
        'Site',
        secondary=user_sites_table,
        back_populates='users',
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

    def to_dict(self) -> dict:
        """Convert user attributes to a dictionary (e.g., for Redis caching).

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
        Boolean,
        nullable=False,
        default=False,
    )
    display_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    raw_profile: Mapped[dict[str, object] | None] = mapped_column(
        JSON,
        nullable=True,
    )
    raw_email_is_private: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    linked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text('CURRENT_TIMESTAMP'),
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
        String(20),
        nullable=False,
        default='zh-TW',
        index=True,
    )
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    effective_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
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
        Boolean,
        nullable=False,
        default=False,
    )
    notification_consent: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    ai_terms_accepted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    accepted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )
    ai_terms_accepted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    notification_consent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    ip_address: Mapped[str | None] = mapped_column(
        String(45),
        nullable=True,
    )
    user_agent: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    user: Mapped[User] = relationship('User', back_populates='consents')
