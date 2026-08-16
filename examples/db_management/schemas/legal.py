from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict

LegalDocumentType = Literal['terms', 'privacy', 'ai_terms']


class LegalDocumentOut(BaseModel):
    """Represent active legal-document content presented to a client.

    Attributes:
        version: Immutable version identifier accepted by the client.
        title: Localised document title.
        content: Localised document body.
    """

    version: str
    title: str
    content: str

    model_config = ConfigDict(from_attributes=True)


class LegalDocumentsResponse(BaseModel):
    """Group every legal document required during signup.

    Attributes:
        terms: Current general terms of service.
        privacy: Current privacy notice.
        ai_terms: Current AI-specific terms of service.
    """

    terms: LegalDocumentOut
    privacy: LegalDocumentOut
    ai_terms: LegalDocumentOut
