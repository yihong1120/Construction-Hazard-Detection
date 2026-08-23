from __future__ import annotations

import unittest

from examples.streaming_web import media_paths


class TestMediaPathParsingCoverage(unittest.TestCase):

    """Provide TestMediaPathParsingCoverage.
    """

    def test_annotated_path_parser_rejects_invalid_path_shapes(self) -> None:
        """Only valid hazard overlay paths decode into a media path and
        locale."""
        self.assertIsNone(
            media_paths.parse_annotated_media_path('hazard_camera'),
        )
        self.assertIsNone(
            media_paths.parse_annotated_media_path('other_annotated_emg'),
        )
        with self.assertRaises(ValueError):
            media_paths.parse_annotated_media_path(
                'hazard_camera_annotated___8',
            )
