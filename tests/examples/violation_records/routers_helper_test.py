from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime
from datetime import timezone
from pathlib import Path

from PIL import Image
from pydantic import ValidationError

from examples.violation_records import violation_services
from examples.violation_records.media_service import _generate_thumbnail_sync
from examples.violation_records.media_service import image_size_for_violation
from examples.violation_records.schemas import FeedbackDetectionItem
from examples.violation_records.schemas import ViolationFeedbackItem
from examples.violation_records.schemas import ViolationListItem


def _feedback(
    feedback_id: int,
    feedback_type: str = 'false_positive',
) -> ViolationFeedbackItem:
    """Create one concise feedback fixture."""
    return ViolationFeedbackItem(
        id=feedback_id,
        type=feedback_type,
        status='pending',
        submitted_at=datetime(2026, 8, 23, tzinfo=timezone.utc),
    )


class TestViolationHelpers(unittest.TestCase):
    """Exercise query, presentation, and media helper behaviour."""

    def test_detection_and_warning_payloads_are_validated(self) -> None:
        """Invalid detector JSON must not be silently accepted."""
        self.assertEqual(
            violation_services._decode_detection_items(
                '[[1, 2, 3, 4, 0.9, 5, 12]]',
            ),
            [[1.0, 2.0, 3.0, 4.0, 0.9, 5.0, 12.0]],
        )
        self.assertEqual(
            violation_services._warning_text_from_json(
                '{"near_vehicle": {"count": 2}}',
            ),
            'near_vehicle: 2',
        )
        with self.assertRaises(ValidationError):
            violation_services._decode_detection_items('{invalid')

    def test_overlay_and_cursor_use_compact_typed_models(self) -> None:
        """The list cursor does not require a full detail response model."""
        detection = FeedbackDetectionItem(
            id='det_0',
            bbox=[10, 10, 40, 40],
        )
        feedback = _feedback(1)
        feedback.target_detection_id = 'det_0'
        feedback.original_bbox = [10, 10, 40, 40]
        overlays = violation_services._overlay_objects_from_feedback(
            [detection],
            [feedback],
            (100, 100),
        )
        self.assertTrue(overlays[0].is_flagged)

        item = ViolationListItem(
            id=7,
            site_name='Site A',
            stream_name='Camera A',
            detection_time=datetime(2026, 8, 23, tzinfo=timezone.utc),
            thumbnail_url='/get_violation_thumbnail?image_path=frame.jpg',
        )
        cursor = violation_services._encode_violation_cursor(item)
        self.assertEqual(
            violation_services._decode_violation_cursor(cursor),
            (item.detection_time, item.id),
        )

    def test_thumbnail_and_image_size_work_without_blocking_the_event_loop(
        self,
    ) -> None:
        """Pillow work remains isolated from the asynchronous request path."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'frame.png'
            thumbnail = root / 'thumbnail.jpg'
            Image.new('RGBA', (30, 15), color=(1, 2, 3, 100)).save(source)
            _generate_thumbnail_sync(source, thumbnail)
            with Image.open(thumbnail) as image:
                self.assertEqual(image.mode, 'RGB')
            self.assertEqual(
                asyncio.run(image_size_for_violation('frame.png', root)),
                (30, 15),
            )
