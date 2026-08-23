from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime
from datetime import timezone
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image
from pydantic import ValidationError

from examples.violation_records import media_service
from examples.violation_records import routers as violation_routers
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

    def test_media_helpers_handle_cache_headers_and_invalid_evidence(
        self,
    ) -> None:
        """Media helpers create cache paths and reject invalid files.

        Invalid images never enter the thumbnail cache.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / 'evidence' / 'frame.png'
            nested.parent.mkdir()
            Image.new('RGB', (12, 6), color=(1, 2, 3)).save(nested)
            cache_path = media_service._thumbnail_cache_path(nested, root)

            self.assertEqual(
                cache_path, root /
                '_thumbnails/evidence/frame.jpg',
            )
            self.assertTrue(media_service._has_recognized_image_header(nested))
            asyncio.run(media_service.ensure_thumbnail(nested, root))
            cached_mtime = cache_path.stat().st_mtime
            _generate_thumbnail_sync(nested, cache_path)
            self.assertEqual(cache_path.stat().st_mtime, cached_mtime)

            blank = root / 'empty.bin'
            blank.write_bytes(b'')
            self.assertFalse(media_service._has_recognized_image_header(blank))
            avif = root / 'image.avif'
            avif.write_bytes(b'\x00\x00\x00\x00ftypavif')
            self.assertTrue(media_service._has_recognized_image_header(avif))
            with self.assertRaises(HTTPException):
                _generate_thumbnail_sync(blank, root / 'blank.jpg')
            self.assertIsNone(
                asyncio.run(image_size_for_violation('../unsafe.png', root)),
            )


class TestViolationRouterForwarders(unittest.IsolatedAsyncioTestCase):
    """Verify routes only forward validated values to application services."""

    async def test_routes_forward_to_their_single_owner_services(self) -> None:
        """Each endpoint delegates once without repeating domain behaviour."""
        db = MagicMock()
        credentials = MagicMock()
        request = MagicMock()
        expected = MagicMock()
        with patch.object(
            violation_routers.violation_query_service,
            'get_my_sites',
            new=AsyncMock(return_value=expected),
        ):
            self.assertIs(
                await violation_routers.get_my_sites(db, credentials),
                expected,
            )
        with patch.object(
            violation_routers.violation_query_service,
            'get_violation_filter_options',
            new=AsyncMock(return_value=expected),
        ):
            self.assertIs(
                await violation_routers.get_violation_filter_options(
                    1,
                    None,
                    db,
                    credentials,
                ),
                expected,
            )
        with patch.object(
            violation_routers.violation_query_service,
            'get_violations',
            new=AsyncMock(return_value=expected),
        ):
            self.assertIs(
                await violation_routers.get_violations(
                    request,
                    db=db,
                    credentials=credentials,
                ),
                expected,
            )
        with patch.object(
            violation_routers.violation_services,
            'get_violation_analytics',
            new=AsyncMock(return_value=expected),
        ):
            self.assertIs(
                await violation_routers.get_violation_analytics(
                    datetime(2026, 1, 1, tzinfo=timezone.utc),
                    datetime(2026, 1, 2, tzinfo=timezone.utc),
                    db=db,
                    credentials=credentials,
                ),
                expected,
            )
        with patch.object(
            violation_routers.violation_review_service,
            'get_next_review_violation',
            new=AsyncMock(return_value=expected),
        ):
            self.assertIs(
                await violation_routers.get_next_review_violation(
                    request,
                    db=db,
                    credentials=credentials,
                ),
                expected,
            )
        with patch.object(
            violation_routers.violation_review_service,
            'get_violation_review_audit_log',
            new=AsyncMock(return_value=expected),
        ):
            self.assertIs(
                await violation_routers.get_violation_review_audit_log(
                    1,
                    db,
                    credentials,
                ),
                expected,
            )
        with patch.object(
            violation_routers.violation_review_service,
            'get_single_violation',
            new=AsyncMock(return_value=expected),
        ):
            self.assertIs(
                await violation_routers.get_single_violation(
                    1,
                    request,
                    db,
                    credentials,
                ),
                expected,
            )
        with patch.object(
            violation_routers.violation_review_service,
            'submit_violation_feedback',
            new=AsyncMock(return_value=expected),
        ):
            self.assertIs(
                await violation_routers.submit_violation_feedback(
                    1,
                    MagicMock(),
                    db,
                    credentials,
                ),
                expected,
            )
        with patch.object(
            violation_routers.violation_review_service,
            'review_violation',
            new=AsyncMock(return_value=expected),
        ):
            self.assertIs(
                await violation_routers.review_violation(
                    1,
                    MagicMock(),
                    request,
                    db,
                    credentials,
                ),
                expected,
            )
        with patch.object(
            violation_routers.violation_media_service,
            'get_violation_image',
            new=AsyncMock(return_value=expected),
        ):
            self.assertIs(
                await violation_routers.get_violation_image(
                    'frame.jpg',
                    db,
                    credentials,
                ),
                expected,
            )
        with patch.object(
            violation_routers.violation_media_service,
            'get_violation_thumbnail',
            new=AsyncMock(return_value=expected),
        ):
            self.assertIs(
                await violation_routers.get_violation_thumbnail(
                    'frame.jpg',
                    db,
                    credentials,
                ),
                expected,
            )
        with patch.object(
            violation_routers.violation_upload_service,
            'upload_violation',
            new=AsyncMock(return_value=expected),
        ):
            self.assertIs(
                await violation_routers.upload_violation(
                    'Site A',
                    'Camera A',
                    None,
                    None,
                    None,
                    None,
                    None,
                    MagicMock(),
                    db,
                    credentials,
                ),
                expected,
            )
