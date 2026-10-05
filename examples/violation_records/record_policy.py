from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import select

from examples.auth.models import Site
from examples.auth.models import Violation
from examples.db_management.services.site_access import require_site
from examples.YOLO_server_api.model_registry import ModelClass
from examples.YOLO_server_api.model_registry import unavailable


async def require_violation(db, user, violation_id: int, *, lock=False):
    query = select(Violation).where(Violation.id == violation_id)
    if lock:
        query = query.with_for_update().execution_options(
            populate_existing=True,
        )
    violation = await db.scalar(query)
    if violation is None:
        raise HTTPException(404, detail='Violation not found')
    await authorize_violation(db, user, violation)
    return violation


async def authorize_violation(db, user, violation):
    site_id = await db.scalar(
        select(Site.id).where(Site.name == violation.site),
    )
    if site_id is None:
        raise HTTPException(403, detail='No access to violation site')
    await require_site(db, user, site_id)
    return violation


def historical_classes(violation) -> list[ModelClass]:
    """Use only the class snapshot persisted with the record."""
    snapshot = getattr(violation, 'model_classes', None)
    if not snapshot:
        return []
    try:
        classes = [ModelClass.model_validate(c) for c in snapshot]
        if len({c.id for c in classes}) != len(classes) or len(
            {c.code for c in classes},
        ) != len(classes):
            raise ValueError('Duplicate class snapshot')
        return classes
    except (TypeError, ValueError) as exc:
        raise unavailable('HISTORICAL_MODEL_METADATA_UNAVAILABLE') from exc


def review_actions(user, violation) -> list[str]:
    if user.role not in {'admin', 'super_admin'} or not violation.is_flagged:
        return []
    transitions = {
        'pending': ['resolved', 'dismissed'],
        'resolved': ['pending'],
        'dismissed': ['pending'],
    }
    return transitions.get(violation.review_status, [])


async def record_options(
    db, user, violation_id: int, kind: str, locale: str = 'zh-TW',
) -> dict:
    """Build record-owned options for the violation API routes."""
    from examples.YOLO_server_api.model_registry import revision_for

    violation = await require_violation(db, user, violation_id)
    if kind == 'feedback':
        options = [
            {
                'id': c.code,
                'label': c.display_name,
                **({'color': c.color} if c.color else {}),
            }
            for c in historical_classes(violation)
        ]
        source = getattr(violation, 'model_version', None)
    elif kind == 'review':
        labels = (
            {'pending': '待審核', 'resolved': '已處理', 'dismissed': '已駁回'}
            if locale.lower().startswith('zh')
            else {}
        )
        options = [
            {'id': state, 'label': labels.get(state, state)}
            for state in review_actions(user, violation)
        ]
        source = violation.review_status
    else:
        raise HTTPException(404, detail='Unknown violation option kind')
    return {
        'schema_version': 1,
        'revision': revision_for(
            {
                'user': user.id,
                'tenant': str(user.tenant_id),
                'role': user.role,
                'violation': violation_id,
                'kind': kind,
                'source': source,
                'options': options,
            },
        ),
        'options': options,
        'default_id': None,
        'policy': {},
    }


async def authenticated_record_options(
    db, credentials, violation_id, kind, locale,
):
    from examples.auth.models import User

    user = await db.get(User, credentials.subject['user_id'])
    if user is None or user.status != 'active':
        raise HTTPException(401, detail='Account unavailable')
    if str(user.tenant_id) != credentials.subject.get('tenant_id'):
        raise HTTPException(403, detail='Tenant mismatch')
    return await record_options(db, user, violation_id, kind, locale)
