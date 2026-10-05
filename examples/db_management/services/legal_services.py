from __future__ import annotations

from datetime import datetime
from datetime import timezone

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import LEGAL_DOCUMENT_TYPES
from examples.auth.models import LegalDocument

# Use one locale whenever a caller does not explicitly request a translation.
DEFAULT_LEGAL_LOCALE = 'zh-TW'


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
        doc_type for doc_type in LEGAL_DOCUMENT_TYPES if doc_type not in docs
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
