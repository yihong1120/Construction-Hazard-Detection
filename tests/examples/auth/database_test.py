from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import Base
from examples.auth.database import engine
from examples.auth.database import get_db


class TestDatabaseEngine(unittest.TestCase):
    """
    Test suite for verifying the SQLAlchemy engine and Base metadata.
    """

    def test_engine_uri_replacement(self) -> None:
        """
        Ensure the engine URL uses 'mysql+asyncmy://' rather than 'mysql://',
        and that Base has valid metadata.
        """
        engine_url_str: str = str(engine.url)
        self.assertIn(
            'mysql+asyncmy://', engine_url_str,
            msg='Engine should use asyncmy in the URL.',
        )
        self.assertIsNotNone(engine, 'Engine instance should not be None.')
        self.assertTrue(
            hasattr(Base, 'metadata'),
            "Base should have a 'metadata' attribute.",
        )


class TestGetDb(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for the get_db() async generator.
    """

    @patch('examples.auth.database.AsyncSessionLocal', new_callable=MagicMock)
    async def test_get_db_yields_session(
        self,
        mock_async_session_local: MagicMock,
    ) -> None:
        """
        Verify that get_db() yields a session from AsyncSessionLocal.
        """
        mock_session = MagicMock(spec=AsyncSession)

        # Simulate __aenter__ / __aexit__ behaviour
        # for an async context manager
        mock_async_session_local.return_value.__aenter__.return_value = (
            mock_session
        )
        mock_async_session_local.return_value.__aexit__.return_value = False

        # Use get_db() and check the yielded session
        async for db_session in get_db():
            self.assertEqual(
                db_session, mock_session,
                'get_db should yield the mock_session',
            )

        # Verify the context manager's __aexit__ is called once
        mock_async_session_local.return_value.__aexit__.assert_called_once()

    async def test_get_db_real_factory(self) -> None:
        """
        Verify that get_db() yields a real AsyncSession instance.
        """
        try:
            # Attempt to yield a real session
            async for db_session in get_db():
                self.assertIsInstance(
                    db_session,
                    AsyncSession,
                    'Should yield a real AsyncSession instance.',
                )
        except Exception:
            self.skipTest('No DB available for real integration test.')


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.auth.database \
    --cov-report=term-missing tests/examples/auth/database_test.py
'''
