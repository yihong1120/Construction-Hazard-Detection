from __future__ import annotations

import asyncio
from datetime import datetime
from datetime import timezone

from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import Text
from sqlalchemy import text
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash

from examples.auth.database import Base

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
        DateTime, server_default=text('CURRENT_TIMESTAMP'),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
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
    Column('created_at', DateTime, server_default=text('CURRENT_TIMESTAMP')),
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
        back_populates='group',
        lazy='joined',
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text('CURRENT_TIMESTAMP'),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
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


# -------------------------------------------------------
#  User Model
# -------------------------------------------------------
class UserProfile(Base):
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
        DateTime, server_default=text('CURRENT_TIMESTAMP'),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
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
        is_active (bool): Whether the user account is active.
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
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
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

    def set_password(self, password: str) -> None:
        """
        Hash and store the user's password securely.

        Args:
            password (str): The plain-text password to be hashed.
        """
        self.password_hash = generate_password_hash(password)

    async def check_password(self, password: str) -> bool:
        """
        Verify whether a given password matches the stored hash.

        This is executed in a thread-safe, asynchronous manner.

        Args:
            password (str): The plain-text password to verify.

        Returns:
            bool: True if the password matches, otherwise False.
        """
        return await asyncio.to_thread(
            check_password_hash,
            str(self.password_hash),
            password,
        )

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
            'is_active': self.is_active,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }


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
    name: Mapped[str] = mapped_column(String(80), nullable=False)

    group_id: Mapped[int | None] = mapped_column(
        ForeignKey('group_info.id', ondelete='SET NULL'),
        nullable=True,
    )
    group: Mapped[Group | None] = relationship(
        'Group',
        back_populates='sites',
        lazy='joined',
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
    )

    # Many-to-many relationship to User
    # This is an association table linking users to sites
    users: Mapped[list[User]] = relationship(
        'User', secondary=user_sites_table, back_populates='sites',
    )

    # One-to-many relationship to Violation
    # This is a foreign key to the violations table
    violations: Mapped[list[Violation]] = relationship(
        'Violation',
        primaryjoin='Site.name == foreign(Violation.site)',
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
        detect_with_server (bool): Whether to use server-side detection.
        expire_date (datetime | None): Expiration date for the configuration.
        work_start_hour (int): Start hour for work scheduling.
        work_end_hour (int): End hour for work scheduling.
        store_in_redis (bool): Whether to store data in Redis.
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

    detect_with_server: Mapped[bool] = mapped_column(Boolean, default=True)
    expire_date:        Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True,
    )

    work_start_hour: Mapped[int] = mapped_column(Integer)
    work_end_hour:   Mapped[int] = mapped_column(Integer)

    store_in_redis: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text('CURRENT_TIMESTAMP'),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
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
        UniqueConstraint('site_id', 'stream_name', name='uq_site_stream'),
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
        created_at (datetime): Time of record creation.
        site (str): Name of the related site (used for linkage).
        site_obj (Site): ORM relationship to the site object.
    """

    __tablename__ = 'violations'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    stream_name: Mapped[str] = mapped_column(String(80), nullable=False)
    detection_time: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.now(timezone.utc),
    )
    image_path: Mapped[str] = mapped_column(String(255), nullable=False)

    # Optional JSON fields for detection and warning results
    detections_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    cone_polygon_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    pole_polygon_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    warnings_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=True, default=datetime.now(timezone.utc),
    )

    # Foreign key: name of the associated site
    site: Mapped[str] = mapped_column(String(80), nullable=False)

    # ORM relationship to the actual Site object
    site_obj: Mapped[Site] = relationship(
        'Site',
        primaryjoin='foreign(Violation.site) == Site.name',
        back_populates='violations',
    )
