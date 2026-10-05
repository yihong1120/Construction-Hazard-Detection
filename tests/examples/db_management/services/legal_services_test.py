from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from fastapi import HTTPException

from examples.auth.models import LegalDocument
from examples.db_management.services import legal_services as svc


def _doc(doc_type: str, version: str = '2026-06-27') -> LegalDocument:
    """Build a legal document ORM instance for service tests."""
    return LegalDocument(
        id=1,
        type=doc_type,
        version=version,
        locale='zh-TW',
        title=doc_type,
        content=f"{doc_type} content",
        effective_at=datetime(2026, 6, 27),
        is_active=True,
    )


def _db_with_docs(docs: list[LegalDocument]) -> AsyncMock:
    """Build an async DB mock returning the provided documents."""
    result = MagicMock()
    result.scalars.return_value.all.return_value = docs
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result)
    return db


class TestLegalServices(unittest.IsolatedAsyncioTestCase):
    """Unit tests for active legal document queries."""

    async def test_get_active_legal_documents_success(self) -> None:
        """It returns all active legal document types."""
        db = _db_with_docs(
            [
                _doc('terms'),
                _doc('privacy'),
                _doc('ai_terms'),
            ],
        )

        docs = await svc.get_active_legal_documents(db, 'zh-TW')

        self.assertEqual(docs['terms'].version, '2026-06-27')
        self.assertEqual(docs['privacy'].title, 'privacy')
        self.assertEqual(docs['ai_terms'].content, 'ai_terms content')

    async def test_get_active_legal_documents_missing(self) -> None:
        """It reports missing required document types."""
        db = _db_with_docs([_doc('terms')])

        with self.assertRaises(HTTPException) as ctx:
            await svc.get_active_legal_documents(db, 'zh-TW')

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(
            ctx.exception.detail['code'],
            'legal_documents_not_found',
        )

    async def test_get_active_legal_documents_merges_default_locale(
        self,
    ) -> None:
        """A non-default locale inherits required documents from zh-TW."""
        requested_result = MagicMock()
        requested_result.scalars.return_value.all.return_value = [
            _doc('terms', '2026-07-01'),
        ]
        fallback_result = MagicMock()
        fallback_result.scalars.return_value.all.return_value = [
            _doc('terms'),
            _doc('privacy'),
            _doc('ai_terms'),
        ]
        db = AsyncMock()
        db.execute = AsyncMock(
            side_effect=[requested_result, fallback_result],
        )

        docs = await svc.get_active_legal_documents(db, 'en-US')

        self.assertEqual(docs['terms'].version, '2026-07-01')
        self.assertIn('privacy', docs)
        self.assertIn('ai_terms', docs)


if __name__ == '__main__':
    unittest.main()
