from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image

from examples.violation_records import routers
from examples.violation_records.schemas import FeedbackDetectionItem
from examples.violation_records.schemas import ViolationFeedbackItem
from examples.violation_records.schemas import ViolationItem


def _feedback(
    feedback_id: int,
    feedback_type: str = 'false_positive',
    **values: object,
) -> ViolationFeedbackItem:
    defaults = {
        'id': feedback_id,
        'type': feedback_type,
        'status': 'pending',
        'submitted_at': datetime(2026, 7, 24, tzinfo=timezone.utc),
    }
    defaults.update(values)
    return ViolationFeedbackItem(**defaults)


class TestViolationRouterHelpers(unittest.TestCase):
    def test_detection_and_warning_json_decoding(self) -> None:
        self.assertIsNone(routers._decode_detection_items(None))
        self.assertIsNone(routers._decode_detection_items('{not json'))
        self.assertEqual(routers._decode_detection_items('[1, 2]'), [1, 2])
        self.assertEqual(
            routers._decode_detection_items('{"detection_items": [3]}'), [3],
        )
        self.assertEqual(
            routers._decode_detection_items('{"items": [4]}'), [4],
        )
        self.assertIsNone(routers._decode_detection_items('{"items": "bad"}'))

        self.assertIsNone(routers._warning_text_from_json(None))
        self.assertEqual(
            routers._warning_text_from_json(
                '{not json',
            ), '{not json',
        )
        self.assertEqual(
            routers._warning_text_from_json(
                '"plain warning"',
            ), 'plain warning',
        )
        self.assertEqual(
            routers._warning_text_from_json(
                '{"near_vehicle": {"count": 2}, "helmet": true, "skip": 0}',
            ),
            'near_vehicle: 2, helmet',
        )
        self.assertEqual(
            routers._warning_text_from_json(
                '["one", "two"]',
            ), 'one, two',
        )
        self.assertIsNone(routers._warning_text_from_json('123'))

    def test_media_url_and_detection_bbox_helpers(self) -> None:
        request = SimpleNamespace(
            url_for=lambda endpoint: f'https://api.test/{endpoint}',
        )
        image_url, thumbnail_url = routers._image_urls(
            'dir/image name.jpg', request,
        )
        self.assertIn('image_path=dir%2Fimage+name.jpg', image_url)
        self.assertIn('get_violation_thumbnail', thumbnail_url)
        self.assertEqual(
            routers._media_endpoint_url(
                'get_violation_image', 'file.jpg', None,
            ),
            '/get_violation_image?image_path=file.jpg',
        )

        self.assertEqual(
            routers._bbox_from_dict({'x1': 1, 'y1': 2, 'x2': 3, 'y2': 4}),
            [1.0, 2.0, 3.0, 4.0],
        )
        self.assertEqual(
            routers._bbox_from_dict({'x': 1, 'y': 2, 'w': 3, 'h': 4}),
            [1.0, 2.0, 4.0, 6.0],
        )
        self.assertIsNone(routers._bbox_from_dict({'x': 1}))
        self.assertEqual(
            routers._bbox_from_detection_item({'bbox': [1, 2, 3, 4]}),
            [1.0, 2.0, 3.0, 4.0],
        )
        self.assertEqual(
            routers._bbox_from_detection_item((1, 2, 3, 4, 0.9)),
            [1.0, 2.0, 3.0, 4.0],
        )
        self.assertIsNone(routers._bbox_from_detection_item({'box': ['bad']}))

    def test_feedback_detection_normalisation_and_ids(self) -> None:
        item = {
            'id': 'tracked',
            'class_name': 'worker',
            'confidence': 0.9,
            'bbox': {'x': 1, 'y': 2, 'width': 3, 'height': 4},
            'track_id': 99,
        }
        normalized = routers._feedback_detection_from_item(item, 0)
        self.assertEqual(normalized.id, 'tracked')
        self.assertEqual(normalized.label, 'worker')
        self.assertEqual(normalized.bbox, [1.0, 2.0, 4.0, 6.0])
        self.assertEqual(
            routers._feedback_detection_id_candidates(item, 0),
            {'det_0', 'tracked', '99'},
        )

        list_item = [1, 2, 3, 4, 0.75, 5, 12]
        normalized_list = routers._feedback_detection_from_item(list_item, 1)
        self.assertEqual(normalized_list.id, 'det_1')
        self.assertEqual(normalized_list.label, 'class-5')
        self.assertEqual(
            routers._feedback_detection_id_candidates(list_item, 1),
            {'det_1', '12'},
        )
        self.assertIsNone(routers._feedback_detection_from_item('invalid', 2))
        self.assertEqual(
            [
                item.id for item in routers._feedback_detections_from_json(
                    '[{"id":"one","bbox":[1,2,3,4]}, "bad"]',
                )
            ],
            ['one'],
        )
        self.assertEqual(
            routers._feedback_detection_ids_from_json(
                '[{"id":"one"}, [1,2,3,4,0.9,2,8]]',
            ),
            {'det_0', 'one', 'det_1', '8'},
        )
        self.assertIsNone(
            routers._feedback_detection_ids_from_json('bad json'),
        )

    def test_bbox_and_overlay_helpers(self) -> None:
        self.assertEqual(routers._clamp_ratio(-0.1), 0.0)
        self.assertEqual(routers._clamp_ratio(2), 1.0)
        self.assertIsNone(routers._bbox_to_normalized(None, (100, 100)))
        self.assertIsNone(routers._bbox_to_normalized(['bad'] * 4, (100, 100)))
        self.assertIsNone(
            routers._bbox_to_normalized(
                [3, 2, 1, 4], (100, 100),
            ),
        )
        self.assertIsNone(routers._bbox_to_normalized([1, 2, 3, 4], None))
        self.assertIsNone(routers._bbox_to_normalized([1, 2, 3, 4], (0, 100)))
        normalized_ratio = routers._bbox_to_normalized(
            [0.1, 0.2, 0.5, 0.7], None,
        )
        normalized_pixels = routers._bbox_to_normalized(
            [10, 20, 50, 70], (100, 100),
        )
        self.assertAlmostEqual(normalized_ratio.x, 0.1)
        self.assertAlmostEqual(normalized_ratio.y, 0.2)
        self.assertAlmostEqual(normalized_ratio.w, 0.4)
        self.assertAlmostEqual(normalized_ratio.h, 0.5)
        self.assertAlmostEqual(normalized_pixels.x, 0.1)
        self.assertAlmostEqual(normalized_pixels.y, 0.2)
        self.assertAlmostEqual(normalized_pixels.w, 0.4)
        self.assertAlmostEqual(normalized_pixels.h, 0.5)
        self.assertFalse(routers._bbox_nearly_equal(None, [1, 2, 3, 4]))
        self.assertTrue(
            routers._bbox_nearly_equal([1, 2, 3, 4], [1, 2, 3, 4.0000001]),
        )

        detection = FeedbackDetectionItem(
            id='det_0', label='worker', confidence=0.8, bbox=[10, 10, 30, 30],
        )
        false_positive = _feedback(
            1,
            target_detection_id='det_0',
            note='not a worker',
            original_bbox=[10, 10, 30, 30],
        )
        false_negative = _feedback(
            2,
            'false_negative',
            corrected_label='vehicle',
            corrected_bbox=[40, 40, 80, 80],
            note='missed vehicle',
        )
        overlays = routers._overlay_objects_from_feedback(
            [detection], [false_positive, false_negative], (100, 100),
        )
        self.assertEqual(
            [overlay.object_id for overlay in overlays], [
                'det_0', 'feedback_2',
            ],
        )
        self.assertTrue(overlays[0].is_flagged)
        self.assertEqual(overlays[1].label, 'vehicle')
        self.assertIs(
            routers._feedback_for_detection(
                detection, [false_positive],
            ), false_positive,
        )
        self.assertEqual(
            routers._overlay_objects_from_feedback(
                [],
                [_feedback(3, 'false_negative')],
                (100, 100),
            ),
            [],
        )

    def test_row_cursor_and_analytics_helper_branches(self) -> None:
        basic_row = tuple(range(routers._violation_column_count + 1))
        row, total = routers._split_violation_row_total(basic_row)
        self.assertEqual(len(row), routers._violation_column_count)
        self.assertEqual(total, routers._violation_column_count)
        self.assertEqual(
            routers._split_violation_row_total(
                (1, 2),
            ), ((1, 2), None),
        )

        mapping_row = SimpleNamespace(_mapping={'total_count': 9})
        self.assertEqual(
            routers._split_violation_row_total(
                mapping_row,
            ), (mapping_row, 9),
        )
        self.assertEqual(
            routers._scalar_value(
                SimpleNamespace(name='Site A'),
            ), 'Site A',
        )
        self.assertEqual(routers._scalar_value(7), 7)
        with patch.object(routers, 'STATIC_DIR', Path('/tmp/static')):
            self.assertEqual(
                routers._path_candidates_for_db(
                    Path('2026/image.jpg'), Path('/tmp/static/2026/image.jpg'),
                ),
                [
                    '2026/image.jpg', 'static/2026/image.jpg',
                    '/tmp/static/2026/image.jpg',
                ],
            )

        timestamp = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
        item = ViolationItem(
            id=7,
            site_name='Site',
            stream_name='Cam',
            detection_time=timestamp,
            image_path='image.jpg',
            created_at=timestamp,
        )
        self.assertEqual(routers._cursor_payload(item), (timestamp, 7))
        self.assertEqual(
            routers._cursor_payload(
                SimpleNamespace(
                    detection_time=timestamp, id=8,
                ),
            ),
            (timestamp, 8),
        )
        cursor = routers._encode_violation_cursor(
            (7, 'Site', 'Cam', timestamp),
        )
        self.assertEqual(
            routers._decode_violation_cursor(
                cursor,
            ), (timestamp, 7),
        )
        with self.assertRaises(HTTPException):
            routers._decode_violation_cursor('bad')

        self.assertEqual(routers._empty_analytics_response().summary.total, 0)
        self.assertEqual(
            routers._normalise_utc(datetime(2026, 7, 24)).tzinfo,
            timezone.utc,
        )
        self.assertEqual(
            routers._format_bucket(
                timestamp, 'hour',
            ), '2026-07-24T12:00:00Z',
        )
        self.assertEqual(routers._format_bucket(timestamp, 'week'), '2026-W30')
        self.assertEqual(
            routers._format_bucket(
                timestamp, 'day',
            ), '2026-07-24',
        )
        self.assertEqual(routers._format_bucket('raw', 'day'), 'raw')

    def test_analytics_database_expressions_and_type_validation(self) -> None:
        for dialect in ['postgresql', 'mysql', 'mariadb', 'sqlite', 'unknown']:
            db = SimpleNamespace(
                bind=SimpleNamespace(dialect=SimpleNamespace(name=dialect)),
            )
            for bucket in ['hour', 'day', 'week']:
                self.assertTrue(
                    str(routers._analytics_bucket_expr(bucket, db)),
                )
            self.assertTrue(str(routers._analytics_hour_expr(db)))
            self.assertTrue(str(routers._type_condition('near_vehicle', db)))

        self.assertEqual(
            routers._canonical_violation_type('no_helmet'),
            'no_safety_helmet',
        )
        with self.assertRaises(HTTPException) as invalid_type:
            routers._canonical_violation_type('unknown')
        self.assertEqual(invalid_type.exception.status_code, 422)

    def test_invalid_detection_values_and_feedback_bbox_matching(self) -> None:
        """Malformed detector values are ignored without breaking feedback UI."""
        self.assertIsNone(
            routers._bbox_from_detection_item({'bbox': [1, 2, 'bad', 4]}),
        )
        malformed_confidence = routers._feedback_detection_from_item(
            [1, 2, 3, 4, 'bad-confidence', 1],
            0,
        )
        self.assertIsNotNone(malformed_confidence)
        self.assertEqual(malformed_confidence.bbox, [1.0, 2.0, 3.0, 4.0])

        detection = FeedbackDetectionItem(
            id='det_1',
            bbox=[1, 2, 3, 4],
        )
        bbox_feedback = _feedback(
            3,
            target_detection_id='different-id',
            original_bbox=[1, 2, 3, 4],
        )
        self.assertIs(
            routers._feedback_for_detection(detection, [bbox_feedback]),
            bbox_feedback,
        )
        self.assertIsNone(routers._feedback_for_detection(detection, []))

    def test_row_total_helpers_cover_mapping_and_scalar_edge_cases(self) -> None:
        """Window-count rows handle tuple and non-sized database result shapes."""

        class MappingRow(tuple):
            @property
            def _mapping(self) -> dict[str, int]:
                return {'total_count': 11}

        full_row = MappingRow(range(routers._violation_column_count + 1))
        truncated, total = routers._split_violation_row_total(full_row)
        self.assertEqual(len(truncated), routers._violation_column_count)
        self.assertEqual(total, 11)

        shorter_row = MappingRow((1, 2))
        self.assertEqual(
            routers._split_violation_row_total(shorter_row),
            (shorter_row, 11),
        )
        self.assertEqual(
            routers._split_violation_row_total(SimpleNamespace()),
            (SimpleNamespace(), None),
        )

    def test_thumbnail_generation_handles_cached_rgba_and_invalid_images(self) -> None:
        """Thumbnails reuse current output, convert alpha images, and reject junk."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.png'
            thumbnail = root / 'thumbnail.jpg'
            Image.new('RGBA', (20, 10), color=(1, 2, 3, 100)).save(source)

            routers._generate_thumbnail_sync(source, thumbnail)
            with Image.open(thumbnail) as created:
                self.assertEqual(created.mode, 'RGB')

            routers._generate_thumbnail_sync(source, thumbnail)

            invalid_source = root / 'invalid.png'
            invalid_source.write_bytes(b'not an image')
            with self.assertRaises(HTTPException) as invalid_image:
                routers._generate_thumbnail_sync(
                    invalid_source, root / 'bad.jpg',
                )

        self.assertEqual(invalid_image.exception.status_code, 400)

    def test_image_size_returns_dimensions_or_none(self) -> None:
        """Detail overlays use available image dimensions and tolerate bad paths."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / 'frame.jpg'
            Image.new('RGB', (30, 15), color='red').save(image_path)
            with patch.object(routers, 'STATIC_DIR', root):
                self.assertEqual(
                    routers._image_size_for_violation('frame.jpg'),
                    (30, 15),
                )
                self.assertIsNone(
                    routers._image_size_for_violation('../outside.jpg'),
                )


class TestViolationMediaAccessCoverage(unittest.IsolatedAsyncioTestCase):
    async def test_media_authorization_rejects_users_without_sites(self) -> None:
        """Existing images remain inaccessible when no effective site is assigned."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'frame.jpg').write_bytes(b'image bytes')
            with (
                patch.object(routers, 'STATIC_DIR', root),
                patch.object(
                    routers,
                    'get_user_sites_cached',
                    new=AsyncMock(return_value=[]),
                ),
            ):
                with self.assertRaises(HTTPException) as denied:
                    await routers._authorize_violation_media_access(
                        'frame.jpg',
                        'user',
                        AsyncMock(),
                    )

        self.assertEqual(denied.exception.status_code, 403)


class TestViolationRouteGuardsCoverage(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.db = MagicMock()
        self.db.execute = AsyncMock()
        self.credentials = SimpleNamespace(subject={'username': 'reviewer'})
        self.missing_credentials = SimpleNamespace(subject={})

    async def test_range_and_stream_filter_reject_invalid_requests(self) -> None:
        """Analytics dates and stream filters expose clear client errors."""
        timestamp = datetime(2026, 7, 24, tzinfo=timezone.utc)
        with self.assertRaises(HTTPException) as invalid_range:
            routers._validate_analytics_range(timestamp, timestamp)
        self.assertEqual(invalid_range.exception.status_code, 422)

        self.db.execute.return_value = SimpleNamespace(first=lambda: None)
        user = SimpleNamespace(role='admin', group_id=1)
        with self.assertRaises(HTTPException) as forbidden_stream:
            await routers._resolve_stream_filter(
                '10',
                None,
                ['Roadwork'],
                user,
                self.db,
            )
        self.assertEqual(forbidden_stream.exception.status_code, 403)

    async def test_filter_options_reject_missing_identity_and_site_scope(self) -> None:
        """Camera filter options require both identity and accessible site ID."""
        with self.assertRaises(HTTPException) as missing_identity:
            await routers.get_violation_filter_options(
                1,
                None,
                self.db,
                self.missing_credentials,
            )
        self.assertEqual(missing_identity.exception.status_code, 401)

        with patch.object(
            routers,
            'load_user_with_effective_sites',
            new=AsyncMock(
                return_value=(SimpleNamespace(role='admin', group_id=1), []),
            ),
        ):
            with self.assertRaises(HTTPException) as inaccessible_site:
                await routers.get_violation_filter_options(
                    1,
                    None,
                    self.db,
                    self.credentials,
                )
        self.assertEqual(inaccessible_site.exception.status_code, 403)

    async def test_violation_list_applies_cursor_filter(self) -> None:
        """A valid cursor adds keyset filtering before an empty page response."""
        cursor = routers._encode_violation_cursor(
            (7, 'Roadwork', 'Cam 1', datetime(2026, 7, 24, tzinfo=timezone.utc)),
        )
        self.db.execute.return_value = SimpleNamespace(all=lambda: [])
        with patch.object(
            routers,
            'get_user_sites_cached',
            new=AsyncMock(return_value=['Roadwork']),
        ):
            result = await routers.get_violations(
                SimpleNamespace(),
                flagged=False,
                review_status=None,
                limit=1,
                offset=0,
                cursor=cursor,
                db=self.db,
                credentials=self.credentials,
            )

        self.assertEqual(result.total, 0)
        self.assertEqual(result.items, [])

    async def test_analytics_rejects_missing_identity_and_handles_no_sites(self) -> None:
        """Analytics does not disclose records to anonymous or unscoped callers."""
        start = datetime(2026, 7, 23, tzinfo=timezone.utc)
        end = datetime(2026, 7, 24, tzinfo=timezone.utc)
        with self.assertRaises(HTTPException) as missing_identity:
            await routers.get_violation_analytics(
                start,
                end,
                db=self.db,
                credentials=self.missing_credentials,
            )
        self.assertEqual(missing_identity.exception.status_code, 401)

        with patch.object(
            routers,
            'require_violation_analytics_access',
            new=AsyncMock(return_value=(SimpleNamespace(role='admin'), [])),
        ):
            result = await routers.get_violation_analytics(
                start,
                end,
                db=self.db,
                credentials=self.credentials,
            )
        self.assertEqual(result.summary.total, 0)

    async def test_review_queue_guards_reject_invalid_scopes(self) -> None:
        """Review queue applies identity, scope, current item, and site checks."""
        with self.assertRaises(HTTPException) as missing_identity:
            await routers.get_next_review_violation(
                SimpleNamespace(),
                db=self.db,
                credentials=self.missing_credentials,
            )
        self.assertEqual(missing_identity.exception.status_code, 401)

        with patch.object(
            routers,
            '_load_review_scope',
            new=AsyncMock(return_value=(SimpleNamespace(), [])),
        ):
            self.assertIsNone(
                await routers.get_next_review_violation(
                    SimpleNamespace(),
                    db=self.db,
                    credentials=self.credentials,
                ),
            )

        self.db.execute.return_value = SimpleNamespace(scalar=lambda: 'Other')
        with patch.object(
            routers,
            '_load_review_scope',
            new=AsyncMock(return_value=(SimpleNamespace(), ['Roadwork'])),
        ):
            with self.assertRaises(HTTPException) as inaccessible_site:
                await routers.get_next_review_violation(
                    SimpleNamespace(),
                    current_id=8,
                    site_id=1,
                    db=self.db,
                    credentials=self.credentials,
                )
        self.assertEqual(inaccessible_site.exception.status_code, 403)

    async def test_audit_log_guards_reject_invalid_scopes(self) -> None:
        """Audit history requires identity, review scope, and record access."""
        with self.assertRaises(HTTPException) as missing_identity:
            await routers.get_violation_review_audit_log(
                1,
                self.db,
                self.missing_credentials,
            )
        self.assertEqual(missing_identity.exception.status_code, 401)

        with patch.object(
            routers,
            '_load_review_scope',
            new=AsyncMock(return_value=(SimpleNamespace(), [])),
        ):
            with self.assertRaises(HTTPException) as empty_scope:
                await routers.get_violation_review_audit_log(
                    1,
                    self.db,
                    self.credentials,
                )
        self.assertEqual(empty_scope.exception.status_code, 403)

        self.db.execute.return_value = SimpleNamespace(scalar=lambda: None)
        with patch.object(
            routers,
            '_load_review_scope',
            new=AsyncMock(return_value=(SimpleNamespace(), ['Roadwork'])),
        ):
            with self.assertRaises(HTTPException) as inaccessible_record:
                await routers.get_violation_review_audit_log(
                    1,
                    self.db,
                    self.credentials,
                )
        self.assertEqual(inaccessible_record.exception.status_code, 403)

    async def test_detail_feedback_review_and_thumbnail_guard_empty_scopes(self) -> None:
        """All mutation and media routes reject callers outside their scope."""
        with patch.object(
            routers,
            'get_user_sites_cached',
            new=AsyncMock(return_value=[]),
        ):
            with self.assertRaises(HTTPException) as detail_denied:
                await routers.get_single_violation(
                    1,
                    SimpleNamespace(),
                    self.db,
                    self.credentials,
                )
            with self.assertRaises(HTTPException) as feedback_denied:
                await routers.submit_violation_feedback(
                    1,
                    MagicMock(),
                    self.db,
                    self.credentials,
                )
        self.assertEqual(detail_denied.exception.status_code, 403)
        self.assertEqual(feedback_denied.exception.status_code, 403)

        with patch.object(
            routers,
            '_load_review_scope',
            new=AsyncMock(return_value=(SimpleNamespace(), [])),
        ):
            with self.assertRaises(HTTPException) as review_denied:
                await routers.review_violation(
                    1,
                    MagicMock(),
                    SimpleNamespace(),
                    self.db,
                    self.credentials,
                )
        self.assertEqual(review_denied.exception.status_code, 403)

        with self.assertRaises(HTTPException) as thumbnail_denied:
            await routers.get_violation_thumbnail(
                'frame.jpg',
                self.db,
                self.missing_credentials,
            )
        self.assertEqual(thumbnail_denied.exception.status_code, 401)

    async def test_review_queue_adds_authorized_site_condition(self) -> None:
        """A valid review site is retained when selecting the next record."""
        site_result = SimpleNamespace(scalar=lambda: 'Roadwork')
        next_result = SimpleNamespace(first=lambda: None)
        self.db.execute.side_effect = [site_result, next_result]
        with patch.object(
            routers,
            '_load_review_scope',
            new=AsyncMock(return_value=(SimpleNamespace(), ['Roadwork'])),
        ):
            result = await routers.get_next_review_violation(
                SimpleNamespace(),
                site_id=1,
                db=self.db,
                credentials=self.credentials,
            )

        self.assertIsNone(result)

    async def test_feedback_rejects_missing_user_after_record_authorization(
        self,
    ) -> None:
        """Feedback creation returns 404 when the authenticated user vanished."""
        violation = SimpleNamespace(id=1, detections_json=None)
        self.db.execute.side_effect = [
            SimpleNamespace(scalar_one_or_none=lambda: violation),
            SimpleNamespace(scalar=lambda: None),
        ]
        payload = routers.ViolationFeedbackCreate(type='false_positive')
        with patch.object(
            routers,
            'get_user_sites_cached',
            new=AsyncMock(return_value=['Roadwork']),
        ):
            with self.assertRaises(HTTPException) as missing_user:
                await routers.submit_violation_feedback(
                    1,
                    payload,
                    self.db,
                    self.credentials,
                )

        self.assertEqual(missing_user.exception.status_code, 404)

    async def test_feedback_transaction_failure_returns_safe_server_error(
        self,
    ) -> None:
        """Feedback persistence failure attempts rollback and hides DB details."""
        violation = SimpleNamespace(id=1, detections_json=None)
        self.db.execute.side_effect = [
            SimpleNamespace(scalar_one_or_none=lambda: violation),
            SimpleNamespace(scalar=lambda: 7),
        ]
        self.db.add = MagicMock()
        self.db.commit = AsyncMock(
            side_effect=RuntimeError('database unavailable'),
        )
        self.db.rollback = AsyncMock(
            side_effect=RuntimeError('rollback unavailable'),
        )
        payload = routers.ViolationFeedbackCreate(type='false_positive')
        with patch.object(
            routers,
            'get_user_sites_cached',
            new=AsyncMock(return_value=['Roadwork']),
        ):
            with self.assertRaises(HTTPException) as failed:
                await routers.submit_violation_feedback(
                    1,
                    payload,
                    self.db,
                    self.credentials,
                )

        self.assertEqual(failed.exception.status_code, 500)
        self.db.rollback.assert_awaited_once()

    async def test_review_rejects_missing_identity_and_unflagged_record(self) -> None:
        """Review updates require a username and an already flagged record."""
        with self.assertRaises(HTTPException) as missing_identity:
            await routers.review_violation(
                1,
                MagicMock(),
                SimpleNamespace(),
                self.db,
                self.missing_credentials,
            )
        self.assertEqual(missing_identity.exception.status_code, 401)

        unflagged = SimpleNamespace(is_flagged=False)
        self.db.execute.return_value = SimpleNamespace(
            scalar_one_or_none=lambda: unflagged,
        )
        with patch.object(
            routers,
            '_load_review_scope',
            new=AsyncMock(
                return_value=(SimpleNamespace(id=7), ['Roadwork']),
            ),
        ):
            with self.assertRaises(HTTPException) as unflagged_record:
                await routers.review_violation(
                    1,
                    MagicMock(),
                    SimpleNamespace(),
                    self.db,
                    self.credentials,
                )

        self.assertEqual(unflagged_record.exception.status_code, 404)

    async def test_review_transaction_failure_returns_safe_server_error(self) -> None:
        """Review persistence failure attempts rollback even if rollback fails."""
        violation = SimpleNamespace(
            id=1,
            is_flagged=True,
            review_status='pending',
            flag_reason='false_positive',
        )
        self.db.execute.return_value = SimpleNamespace(
            scalar_one_or_none=lambda: violation,
        )
        self.db.add = MagicMock()
        self.db.commit = AsyncMock(
            side_effect=RuntimeError('database unavailable'),
        )
        self.db.rollback = AsyncMock(
            side_effect=RuntimeError('rollback unavailable'),
        )
        payload = SimpleNamespace(
            review_status='resolved',
            review_note='confirmed',
        )
        with patch.object(
            routers,
            '_load_review_scope',
            new=AsyncMock(
                return_value=(SimpleNamespace(id=7), ['Roadwork']),
            ),
        ):
            with self.assertRaises(HTTPException) as failed:
                await routers.review_violation(
                    1,
                    payload,
                    SimpleNamespace(),
                    self.db,
                    self.credentials,
                )

        self.assertEqual(failed.exception.status_code, 500)
        self.db.rollback.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
