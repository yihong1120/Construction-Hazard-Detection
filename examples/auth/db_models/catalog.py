from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean
from sqlalchemy import CheckConstraint
from sqlalchemy import DateTime
from sqlalchemy import event
from sqlalchemy import Index
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import text
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

from examples.auth.database import Base
from examples.auth.db_models.model_version_schema import install_model_versions


class DetectionModelCatalog(Base):
    """Published model metadata; weights live on Hugging Face."""

    __tablename__ = 'detection_model_catalog'
    model_key: Mapped[str] = mapped_column(String(120), primary_key=True)
    definition: Mapped[dict] = mapped_column(JSON, nullable=False)
    display_order: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False,
    )
    is_default: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False,
    )
    __table_args__ = (
        Index(
            'detection_model_catalog_one_default', 'is_default', unique=True,
            postgresql_where=text('is_default'),
        ),
    )


class ModelPublicationRecord(Base):
    __tablename__ = 'model_publications'
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    phase: Mapped[str] = mapped_column(String(32), nullable=False)
    snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False)


class DetectionModelVersion(Base):
    """Immutable evidence metadata without historical authorization grants."""

    __tablename__ = 'detection_model_versions'
    model_key: Mapped[str] = mapped_column(String(120), primary_key=True)
    version: Mapped[str] = mapped_column(Text, primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    classes: Mapped[list] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text('CURRENT_TIMESTAMP'),
    )
    __table_args__ = (
        CheckConstraint(
            'length(sha256)=64',
            name='model_version_sha256_length',
        ),
    )


event.listen(Base.metadata, 'after_create', install_model_versions)
