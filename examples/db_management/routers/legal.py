from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Query
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.db_management.schemas.legal import LegalDocumentOut
from examples.db_management.schemas.legal import LegalDocumentsResponse
from examples.db_management.services.legal_services import (
    DEFAULT_LEGAL_LOCALE,
)
from examples.db_management.services.legal_services import (
    get_active_legal_documents,
)

router = APIRouter(prefix='/legal', tags=['legal'])


@router.get('/documents', response_model=LegalDocumentsResponse)
async def get_legal_documents(
    locale: str = Query(default=DEFAULT_LEGAL_LOCALE),
    db: AsyncSession = Depends(get_db),
) -> LegalDocumentsResponse:
    """Return active legal documents for signup consent."""
    docs = await get_active_legal_documents(db, locale)
    return LegalDocumentsResponse(
        terms=LegalDocumentOut.model_validate(docs['terms']),
        privacy=LegalDocumentOut.model_validate(docs['privacy']),
        ai_terms=LegalDocumentOut.model_validate(docs['ai_terms']),
    )
