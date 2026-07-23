from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.auth.models import LegalDocument
from examples.db_management.routers.legal import get_legal_documents


class TestLegalRouter(unittest.IsolatedAsyncioTestCase):
    """Unit tests for legal document routes."""

    @patch(
        'examples.db_management.routers.legal.get_active_legal_documents',
        new_callable=AsyncMock,
    )
    async def test_get_legal_documents(
        self,
        mock_get_docs: AsyncMock,
    ) -> None:
        """It returns grouped active legal documents."""
        mock_get_docs.return_value = {
            'terms': LegalDocument(
                type='terms',
                version='2026-06-27',
                title='使用條款',
                content='terms content',
            ),
            'privacy': LegalDocument(
                type='privacy',
                version='2026-06-27',
                title='隱私權政策',
                content='privacy content',
            ),
            'ai_terms': LegalDocument(
                type='ai_terms',
                version='2026-06-27',
                title='LLM 與 AI Agent 使用條款',
                content='ai content',
            ),
        }
        db = MagicMock()

        response = await get_legal_documents(locale='zh-TW', db=db)

        mock_get_docs.assert_awaited_once_with(db, 'zh-TW')
        self.assertEqual(response.terms.version, '2026-06-27')
        self.assertEqual(response.privacy.title, '隱私權政策')
        self.assertEqual(response.ai_terms.content, 'ai content')


if __name__ == '__main__':
    unittest.main()
