from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import and_
from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy.sql.elements import ColumnElement

from examples.auth.models import Site
from examples.auth.models import Violation


def violation_site_scope(names: Sequence[str]) -> ColumnElement[bool]:
    """Resolve names to IDs; retain name matching for unbackfilled rows."""
    return or_(
        Violation.site_id.in_(select(Site.id).where(Site.name.in_(names))),
        and_(Violation.site_id.is_(None), Violation.site.in_(names)),
    )
