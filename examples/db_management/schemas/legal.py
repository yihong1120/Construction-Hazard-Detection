from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict

LegalDocumentType = Literal['terms', 'privacy', 'ai_terms']


class LegalDocumentOut(BaseModel):
    """Active legal document content returned to clients."""

    version: str
    title: str
    content: str

    model_config = ConfigDict(from_attributes=True)


class LegalDocumentsResponse(BaseModel):
    """Grouped active legal documents required for signup."""

    terms: LegalDocumentOut
    privacy: LegalDocumentOut
    ai_terms: LegalDocumentOut
