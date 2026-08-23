from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.db_management.schemas.auth import AccessTokenSubject
from examples.streaming_web import stream_catalog_service


def _credentials(username: str | None) -> JwtAuthorizationCredentials:
    """Build compact credentials for catalogue authorisation tests.

    Args:
        username: Optional authenticated username claim.

    Returns:
        Minimal JWT credentials accepted by the catalogue service.
    """
    subject: dict[str, object] = {'username': username} if username else {}
    return JwtAuthorizationCredentials(
        subject=cast(AccessTokenSubject, subject),
    )


class TestStreamCatalogueCoverage(unittest.IsolatedAsyncioTestCase):
    """Exercise empty and rejected stream catalogue request paths."""

    async def test_labels_return_empty_for_users_without_sites(self) -> None:
        """Users without effective sites receive no visible labels."""
        db = SimpleNamespace(execute=AsyncMock())
        with patch.object(
            stream_catalog_service,
            'load_user_access_context',
            new=AsyncMock(return_value=(MagicMock(), [], 'admin')),
        ):
            result = await stream_catalog_service.get_visible_labels(
                _credentials('alice'),
                cast(AsyncSession, db),
            )

        self.assertEqual(result.labels, [])
        db.execute.assert_not_awaited()

    async def test_batch_stream_resolution_rejects_empty_and_missing_rows(
        self,
    ) -> None:
        """Batch selection rejects absent identifiers and missing rows."""
        db = SimpleNamespace(
            execute=AsyncMock(return_value=SimpleNamespace(all=lambda: [])),
        )
        self.assertEqual(
            await stream_catalog_service.resolve_configured_stream_names(
                cast(AsyncSession, db),
                [],
            ),
            [],
        )

        with self.assertRaises(HTTPException) as absent:
            await stream_catalog_service.resolve_configured_stream_names(
                cast(AsyncSession, db),
                [('Site A', None, None)],
            )
        self.assertEqual(absent.exception.status_code, 422)

        with self.assertRaises(HTTPException) as missing:
            await stream_catalog_service.resolve_configured_stream_names(
                cast(AsyncSession, db),
                [('Site A', None, 'Camera A')],
            )
        self.assertEqual(missing.exception.status_code, 404)
