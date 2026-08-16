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

# Use one locale whenever a caller does not explicitly request a translation.
DEFAULT_LEGAL_LOCALE = 'zh-TW'


class SignupConsentPayload(Protocol):
    """Define the consent values required while registering a user.

    Attributes:
        accepted_terms: Whether the user accepted the general terms.
        terms_version: Version of the general terms accepted by the user.
        privacy_version: Version of the privacy notice accepted by the user.
        notification_consent: Whether the user agreed to notifications.
        ai_terms_accepted: Whether the user accepted the AI-specific terms.
        ai_terms_version: Version of the AI-specific terms accepted by the user.
    """

    accepted_terms: bool
    terms_version: str | None
    privacy_version: str | None
    notification_consent: bool
    ai_terms_accepted: bool
    ai_terms_version: str | None


def _now() -> datetime:
    """Return the current timezone-aware UTC timestamp.

    Returns:
        Current time in UTC for document availability and consent audit data.
    """
    # UTC keeps effective-date comparisons and persisted audit events stable.
    return datetime.now(timezone.utc)


async def get_active_legal_documents(
    db: AsyncSession,
    locale: str = DEFAULT_LEGAL_LOCALE,
) -> dict[str, LegalDocument]:
    """Load every required active legal document for a locale.

    Args:
        db: Asynchronous database session used to read legal documents.
        locale: Preferred locale for the document text.

    Returns:
        Active documents keyed by their document type.

    Raises:
        HTTPException: If one or more required document types are unavailable
            in the requested locale or the default locale.
    """
    requested = locale or DEFAULT_LEGAL_LOCALE
    docs = await _load_active_documents_for_locale(db, requested)

    if requested != DEFAULT_LEGAL_LOCALE:
        # A partial translation inherits only the missing documents from the
        # default locale; translated documents always take precedence.
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
    """Load the newest effective active document of each type for a locale.

    Args:
        db: Asynchronous database session used to query legal documents.
        locale: Locale whose documents should be selected.

    Returns:
        Available active documents keyed by document type.  The mapping may be
        incomplete when a locale does not provide every required document.
    """
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
        # The query ordering makes the first document per type the current one.
        docs.setdefault(doc.type, doc)
    return docs


async def validate_signup_consents(
    payload: SignupConsentPayload,
    db: AsyncSession,
    locale: str = DEFAULT_LEGAL_LOCALE,
) -> dict[str, LegalDocument]:
    """Validate required signup consents against the current document versions.

    Args:
        payload: Typed signup payload containing consent flags and versions.
        db: Asynchronous database session used to retrieve active documents.
        locale: Preferred locale for the documents presented at signup.

    Returns:
        Current legal documents keyed by document type, ready for audit storage.

    Raises:
        HTTPException: If a mandatory consent is absent, a submitted version
            differs from the active version, or a required document is missing.
    """
    if not payload.accepted_terms:
        raise HTTPException(400, 'accepted_terms is required.')
    if not payload.notification_consent:
        raise HTTPException(400, 'notification_consent is required.')
    if not payload.ai_terms_accepted:
        raise HTTPException(400, 'ai_terms_accepted is required.')

    docs = await get_active_legal_documents(db, locale)
    # Pin consent to the exact version displayed during registration rather
    # than accepting a bare affirmative flag.
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
    """Persist an immutable consent snapshot for a newly registered user.

    Args:
        user_id: Identifier of the user whose consents are being recorded.
        payload: Validated signup payload containing accepted versions and flags.
        db: Asynchronous database session used to persist the consent record.
        request: Optional HTTP request used to capture audit metadata.

    Returns:
        Persisted consent record, including its database-generated fields.
    """
    accepted_at = _now()
    # Store the submitted versions with the event so later document updates do
    # not alter the historical consent evidence.
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
    """Extract a bounded client IP address from a request.

    Args:
        request: Optional HTTP request containing forwarded and peer addresses.

    Returns:
        The first forwarded or peer IP address, or ``None`` when unavailable.
    """
    if request is None:
        return None
    # A trusted reverse proxy appends the immediate peer after the originating
    # address, so retain only the left-most value for the consent audit trail.
    forwarded = request.headers.get('x-forwarded-for')
    if forwarded:
        return forwarded.split(',', 1)[0].strip()[:45]
    if request.client is None:
        return None
    return request.client.host[:45]


def _user_agent(request: Request | None) -> str | None:
    """Extract a bounded user-agent string from a request.

    Args:
        request: Optional HTTP request containing client headers.

    Returns:
        The user-agent value truncated to 255 characters, or ``None`` when the
        request or header is unavailable.
    """
    if request is None:
        return None
    # Limit untrusted header data before persisting it with the consent record.
    value = request.headers.get('user-agent')
    return value[:255] if value else None
