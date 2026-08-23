from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from examples.streaming_web.playback_demand import active_overlay_languages


class TestPlaybackDemandCoverage(unittest.IsolatedAsyncioTestCase):
    """Verify empty language configuration avoids unnecessary Redis I/O."""

    async def test_empty_allowed_language_set_skips_redis_lookup(self) -> None:
        """No overlay languages means no demand keys need to be read."""
        redis = MagicMock()
        redis.mget = AsyncMock()
        self.assertEqual(
            await active_overlay_languages(redis, 'site-camera', ()),
            set(),
        )
        redis.mget.assert_not_awaited()
