from __future__ import annotations

from datetime import datetime
from datetime import timezone
from typing import Protocol

from fastapi import HTTPException
from fastapi import Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import LEGAL_DOCUMENT_TYPE_AI_TERMS
from examples.auth.models import LEGAL_DOCUMENT_TYPE_PRIVACY
from examples.auth.models import LEGAL_DOCUMENT_TYPE_TERMS
from examples.auth.models import LEGAL_DOCUMENT_TYPES
from examples.auth.models import LegalDocument
from examples.auth.models import UserConsent

DEFAULT_LEGAL_LOCALE = 'zh-TW'


class SignupConsentPayload(Protocol):
    """Minimal consent fields required from signup payloads."""

    accepted_terms: bool
    terms_version: str | None
    privacy_version: str | None
    notification_consent: bool
    ai_terms_accepted: bool
    ai_terms_version: str | None


def _now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


async def get_active_legal_documents(
    db: AsyncSession,
    locale: str = DEFAULT_LEGAL_LOCALE,
) -> dict[str, LegalDocument]:
    """Load active legal documents for the requested locale.

    Missing documents fall back to the default locale. A 404 is raised if
    any required document type is unavailable.
    """
    requested = locale or DEFAULT_LEGAL_LOCALE
    docs = await _load_active_documents_for_locale(db, requested)

    if requested != DEFAULT_LEGAL_LOCALE:
        fallback_docs = await _load_active_documents_for_locale(
            db,
            DEFAULT_LEGAL_LOCALE,
        )
        docs = {**fallback_docs, **docs}

    missing = [
        doc_type
        for doc_type in LEGAL_DOCUMENT_TYPES
        if doc_type not in docs
    ]
    if missing:
        raise HTTPException(
            status_code=404,
            detail={
                'code': 'legal_documents_not_found',
                'missing': missing,
            },
        )
    return docs


async def _load_active_documents_for_locale(
    db: AsyncSession,
    locale: str,
) -> dict[str, LegalDocument]:
    """Load the newest active legal document per type for one locale."""
    stmt = (
        select(LegalDocument)
        .where(
            LegalDocument.locale == locale,
            LegalDocument.is_active.is_(True),
            LegalDocument.effective_at <= _now(),
        )
        .order_by(
            LegalDocument.type.asc(),
            LegalDocument.effective_at.desc(),
            LegalDocument.created_at.desc(),
            LegalDocument.id.desc(),
        )
    )
    result = await db.execute(stmt)
    docs: dict[str, LegalDocument] = {}
    for doc in result.scalars().all():
        docs.setdefault(doc.type, doc)
    return docs


async def validate_signup_consents(
    payload: SignupConsentPayload,
    db: AsyncSession,
    locale: str = DEFAULT_LEGAL_LOCALE,
) -> dict[str, LegalDocument]:
    """Validate mandatory signup consents against active document versions."""
    if not payload.accepted_terms:
        raise HTTPException(400, 'accepted_terms is required.')
    if not payload.notification_consent:
        raise HTTPException(400, 'notification_consent is required.')
    if not payload.ai_terms_accepted:
        raise HTTPException(400, 'ai_terms_accepted is required.')

    docs = await get_active_legal_documents(db, locale)
    expected_versions = {
        'terms_version': docs[LEGAL_DOCUMENT_TYPE_TERMS].version,
        'privacy_version': docs[LEGAL_DOCUMENT_TYPE_PRIVACY].version,
        'ai_terms_version': docs[LEGAL_DOCUMENT_TYPE_AI_TERMS].version,
    }
    submitted_versions = {
        'terms_version': payload.terms_version,
        'privacy_version': payload.privacy_version,
        'ai_terms_version': payload.ai_terms_version,
    }
    mismatches = {
        field: {
            'expected': expected,
            'submitted': submitted_versions[field],
        }
        for field, expected in expected_versions.items()
        if submitted_versions[field] != expected
    }
    if mismatches:
        raise HTTPException(
            status_code=400,
            detail={
                'code': 'legal_version_mismatch',
                'mismatches': mismatches,
            },
        )
    return docs


async def record_user_consent(
    user_id: int,
    payload: SignupConsentPayload,
    db: AsyncSession,
    request: Request | None = None,
) -> UserConsent:
    """Persist one consent snapshot for a newly registered user."""
    accepted_at = _now()
    consent = UserConsent(
        user_id=user_id,
        terms_version=str(payload.terms_version),
        privacy_version=str(payload.privacy_version),
        ai_terms_version=str(payload.ai_terms_version),
        accepted_terms=payload.accepted_terms,
        notification_consent=payload.notification_consent,
        ai_terms_accepted=payload.ai_terms_accepted,
        accepted_at=accepted_at,
        ai_terms_accepted_at=(
            accepted_at if payload.ai_terms_accepted else None
        ),
        notification_consent_at=(
            accepted_at if payload.notification_consent else None
        ),
        ip_address=_client_ip(request),
        user_agent=_user_agent(request),
    )
    db.add(consent)
    await db.commit()
    await db.refresh(consent)
    return consent


def _client_ip(request: Request | None) -> str | None:
    """Extract a best-effort client IP from the request."""
    if request is None:
        return None
    forwarded = request.headers.get('x-forwarded-for')
    if forwarded:
        return forwarded.split(',', 1)[0].strip()[:45]
    if request.client is None:
        return None
    return request.client.host[:45]


def _user_agent(request: Request | None) -> str | None:
    """Extract a bounded user-agent string from the request."""
    if request is None:
        return None
    value = request.headers.get('user-agent')
    return value[:255] if value else None
