from __future__ import annotations

from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Table
from sqlalchemy import text

from examples.auth.database import Base


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
        'site_id',
        ForeignKey('sites.id', ondelete='CASCADE'),
        primary_key=True,
    ),
    Column(
        'group_id',
        ForeignKey('group_info.id', ondelete='CASCADE'),
        primary_key=True,
    ),
    Column(
        'created_at',
        DateTime(timezone=True),
        server_default=text('CURRENT_TIMESTAMP'),
    ),
)


group_features_table: Table = Table(
    'group_features',
    Base.metadata,
    Column(
        'group_id',
        ForeignKey(
            'group_info.id',
            ondelete='CASCADE',
        ),
        primary_key=True,
    ),
    Column(
        'feature_id',
        ForeignKey(
            'features.id',
            ondelete='CASCADE',
        ),
        primary_key=True,
    ),
    Column(
        'created_at',
        DateTime(timezone=True),
        server_default=text('CURRENT_TIMESTAMP'),
    ),
)
