from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from examples.streaming_web.playback_languages import (
    _allowed_overlay_languages,
)


class TestPlaybackLanguageCoverage(unittest.TestCase):
    """Verify invalid overlay language configuration is rejected."""

    def test_unknown_configured_overlay_language_is_rejected(self) -> None:
        """An unrecognised language cannot reach overlay rendering."""
        with patch.dict(
            os.environ,
            {'MEDIA_OVERLAY_ALLOWED_LANGUAGES': 'en,not-a-language'},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                _allowed_overlay_languages()
