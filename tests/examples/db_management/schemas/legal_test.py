from __future__ import annotations

import unittest

from examples.db_management.schemas.legal import LegalDocumentOut
from examples.db_management.schemas.legal import LegalDocumentsResponse


class TestLegalSchemas(unittest.TestCase):
    """Unit tests for legal response schemas."""

    def test_legal_documents_response(self) -> None:
        """It groups terms, privacy, and AI terms documents."""
        doc = LegalDocumentOut(
            version='2026-06-27',
            title='使用條款',
            content='content',
        )
        response = LegalDocumentsResponse(
            terms=doc,
            privacy=doc,
            ai_terms=doc,
        )

        self.assertEqual(response.terms.version, '2026-06-27')
        self.assertEqual(response.ai_terms.title, '使用條款')


if __name__ == '__main__':
    unittest.main()
