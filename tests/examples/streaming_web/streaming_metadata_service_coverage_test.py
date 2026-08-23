from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.streaming_web import streaming_metadata_service as service


class TestStreamingMetadataServiceCoverage(unittest.IsolatedAsyncioTestCase):
    """Verify encoded stream WebSockets delegate to the shared handler."""

    async def test_stream_identifier_websocket_uses_shared_handler(
        self,
    ) -> None:
        """Service forwards dependencies without rebuilding metadata keys."""
        websocket = MagicMock()
        redis = MagicMock()
        db = MagicMock()
        with patch.object(
            service,
            'handle_metadata_stream_id_ws',
            new=AsyncMock(),
        ) as handle_websocket:
            await service.metadata_stream_websocket(
                websocket,
                'Site A',
                '12',
                redis,
                db,
            )

        handle_websocket.assert_awaited_once()
