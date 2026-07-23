from __future__ import annotations

import unittest
from datetime import datetime
from types import SimpleNamespace
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
        content=f'{doc_type} content',
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
    """Unit tests for legal document and consent services."""

    async def test_get_active_legal_documents_success(self) -> None:
        """It returns all active legal document types."""
        db = _db_with_docs([
            _doc('terms'),
            _doc('privacy'),
            _doc('ai_terms'),
        ])

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

    async def test_validate_signup_consents_success(self) -> None:
        """It accepts matching versions and mandatory booleans."""
        db = _db_with_docs([
            _doc('terms'),
            _doc('privacy'),
            _doc('ai_terms'),
        ])
        payload = SimpleNamespace(
            accepted_terms=True,
            terms_version='2026-06-27',
            privacy_version='2026-06-27',
            notification_consent=True,
            ai_terms_accepted=True,
            ai_terms_version='2026-06-27',
        )

        docs = await svc.validate_signup_consents(payload, db)

        self.assertEqual(docs['terms'].version, '2026-06-27')

    async def test_validate_signup_consents_version_mismatch(self) -> None:
        """It rejects stale legal document versions."""
        db = _db_with_docs([
            _doc('terms'),
            _doc('privacy'),
            _doc('ai_terms'),
        ])
        payload = SimpleNamespace(
            accepted_terms=True,
            terms_version='old',
            privacy_version='2026-06-27',
            notification_consent=True,
            ai_terms_accepted=True,
            ai_terms_version='2026-06-27',
        )

        with self.assertRaises(HTTPException) as ctx:
            await svc.validate_signup_consents(payload, db)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(
            ctx.exception.detail['code'],
            'legal_version_mismatch',
        )

    async def test_record_user_consent(self) -> None:
        """It stores consent versions, timestamps, IP, and user-agent."""
        payload = SimpleNamespace(
            accepted_terms=True,
            terms_version='2026-06-27',
            privacy_version='2026-06-27',
            notification_consent=True,
            ai_terms_accepted=True,
            ai_terms_version='2026-06-27',
        )
        request = SimpleNamespace(
            headers={
                'x-forwarded-for': '203.0.113.1, 10.0.0.1',
                'user-agent': 'FlutterTest/1.0',
            },
            client=SimpleNamespace(host='127.0.0.1'),
        )
        db = AsyncMock()
        db.add = MagicMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()

        consent = await svc.record_user_consent(7, payload, db, request)

        self.assertEqual(consent.user_id, 7)
        self.assertEqual(consent.ip_address, '203.0.113.1')
        self.assertEqual(consent.user_agent, 'FlutterTest/1.0')
        self.assertTrue(consent.accepted_terms)
        self.assertTrue(consent.notification_consent)
        self.assertTrue(consent.ai_terms_accepted)
        db.add.assert_called_once_with(consent)
        db.commit.assert_awaited_once()
        db.refresh.assert_awaited_once_with(consent)


if __name__ == '__main__':
    unittest.main()
