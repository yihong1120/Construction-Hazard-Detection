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
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash

from examples.auth.database import Base

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

    # Many-to-many: User <-> Sites
    sites: Mapped[list[Site]] = relationship(
        'Site', secondary=user_sites_table, back_populates='users',
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

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
    )

    # Many-to-many: Site <-> Users
    users: Mapped[list[User]] = relationship(
        'User', secondary=user_sites_table, back_populates='sites',
    )

    # One-to-many: Site -> Violations (linked by site name)
    violations: Mapped[list[Violation]] = relationship(
        'Violation',
        primaryjoin='Site.name == foreign(Violation.site)',
        back_populates='site_obj',
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
