from __future__ import annotations

import unittest
from datetime import datetime
from datetime import timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import Violation
from examples.auth.models import ViolationFeedback
from examples.auth.models import ViolationReviewAuditLog
from examples.db_management.schemas.auth import AccessTokenSubject
from examples.violation_records import violation_services
from examples.violation_records.schemas import FeedbackDetectionItem
from examples.violation_records.schemas import ViolationFeedbackItem
from examples.violation_records.schemas import ViolationListItem


def _credentials(username: str = 'admin') -> JwtAuthorizationCredentials:
    """Build credentials for a focused violation-service unit test.

    Args:
        username: Authenticated username exposed by the access-token subject.

    Returns:
        Minimal validated credentials used by the service functions.
    """
    return JwtAuthorizationCredentials(
        subject=cast(
            AccessTokenSubject,
            {
                'username': username,
                'user_id': 1,
                'role': 'admin',
                'jti': 'violation-service-test',
                'features': [],
            },
        ),
    )


class _Request:
    """Provide the request URL API used by violation response helpers."""

    def url_for(self, endpoint_name: str) -> str:
        """Return a deterministic protected-media endpoint.

        Args:
            endpoint_name: FastAPI route name requested by the helper.

        Returns:
            Absolute route URL for the test application.
        """
        return f'https://api.example.test/{endpoint_name}'


class TestViolationServiceHelpers(unittest.TestCase):
    """Verify pure formatting, filtering, and cursor helper behaviour."""

    def test_detail_and_overlay_helpers_normalise_feedback(self) -> None:
        """Detail payloads expose detector rows and feedback overlays once."""
        timestamp = datetime(2026, 8, 23, tzinfo=timezone.utc)
        violation = SimpleNamespace(
            id=7,
            site='Site A',
            stream_name='Camera A',
            detection_time=timestamp,
            image_path='2026-08-23/evidence.jpg',
            created_at=timestamp,
            detections_json='[[10, 20, 60, 80, 0.9, 3, 42]]',
            warnings_json='{"warning_no_hardhat": {"count": 2}}',
            cone_polygon_json=None,
            pole_polygon_json=None,
            is_flagged=True,
            flag_reason='wrong_class',
            flagged_by=1,
            flagged_at=timestamp,
            review_status='pending',
            review_note=None,
            reviewed_by=None,
            reviewed_at=None,
        )
        feedback = ViolationFeedbackItem(
            id=9,
            type='wrong_class',
            target_detection_id='det_0',
            corrected_label='worker',
            original_bbox=[10, 20, 60, 80],
            status='pending',
            submitted_at=timestamp,
        )

        item = violation_services._violation_to_detail_item(
            cast(Violation, violation),
            _Request(),
            'Needs review',
        )
        overlays = violation_services._overlay_objects_from_feedback(
            item.detections,
            [feedback],
            (100, 100),
        )

        self.assertEqual(item.warning_text, 'warning_no_hardhat: 2')
        assert item.detections is not None
        self.assertEqual(item.detections[0].id, 'det_0')
        self.assertEqual(
            item.image_url, (
                'https://api.example.test/get_violation_image?'
                'image_path=2026-08-23%2Fevidence.jpg'
            ),
        )
        self.assertEqual(len(overlays), 1)
        self.assertEqual(overlays[0].bbox.x, 0.1)
        self.assertTrue(overlays[0].is_flagged)

    def test_overlay_helpers_keep_false_negative_and_skip_invalid_boxes(
        self,
    ) -> None:
        """False negatives add overlays while malformed boxes are ignored."""
        timestamp = datetime(2026, 8, 23, tzinfo=timezone.utc)
        feedbacks = [
            ViolationFeedbackItem(
                id=1,
                type='false_negative',
                corrected_label='helmet',
                corrected_bbox=[10, 20, 50, 80],
                status='pending',
                submitted_at=timestamp,
            ),
            ViolationFeedbackItem(
                id=2,
                type='wrong_class',
                original_bbox=[40, 20, 10, 80],
                status='pending',
                submitted_at=timestamp,
            ),
        ]
        detections = [
            FeedbackDetectionItem(
                id='det_0',
                label='class-3',
                confidence=0.8,
                bbox=[80, 70, 20, 90],
            ),
        ]

        overlays = violation_services._overlay_objects_from_feedback(
            detections,
            feedbacks,
            (100, 100),
        )

        self.assertEqual(
            [overlay.object_id for overlay in overlays], [
                'feedback_1',
            ],
        )
        self.assertEqual(overlays[0].bbox.w, 0.4)
        self.assertIsNone(
            violation_services._bbox_to_normalized([2, 2, 1, 3], (10, 10)),
        )
        self.assertIsNone(
            violation_services._bbox_to_normalized([2, 2, 3, 3], None),
        )

    def test_cursor_and_optional_conditions_validate_request_values(
        self,
    ) -> None:
        """Keyset cursors round-trip and invalid type filters fail early."""
        item = ViolationListItem(
            id=12,
            site_name='Site A',
            stream_name='Camera A',
            detection_time=datetime(2026, 8, 23, tzinfo=timezone.utc),
            thumbnail_url='https://api.example.test/thumbnail',
        )

        cursor = violation_services._encode_violation_cursor(item)
        decoded_time, decoded_id = violation_services._decode_violation_cursor(
            cursor,
        )

        self.assertEqual(decoded_time, item.detection_time)
        self.assertEqual(decoded_id, 12)
        self.assertIsNotNone(
            violation_services._optional_violation_conditions(
                'no_safety_helmet',
                None,
                item.detection_time,
                item.detection_time,
                cursor,
            ),
        )
        with self.assertRaises(HTTPException) as raised:
            violation_services._optional_violation_conditions(
                'not-a-real-type',
                None,
                None,
                None,
                None,
            )
        self.assertEqual(raised.exception.status_code, 422)


class TestViolationAnalytics(unittest.IsolatedAsyncioTestCase):
    """Verify PostgreSQL aggregate result normalisation and validation."""

    async def test_analytics_maps_and_sorts_one_query_result(self) -> None:
        """A single aggregate query produces complete sorted dashboard data."""
        aggregate_rows = [
            ('summary', None, None, 9, 3),
            ('trend', '2026-08-23', None, 4, 0),
            ('trend', '2026-08-22', None, 5, 0),
            ('site', '2', 'Site B', 3, 0),
            ('site', '1', 'Site A', 6, 0),
            ('hour', '18', None, 4, 0),
            ('hour', '8', None, 5, 0),
            ('type', 'no_safety_helmet', '未戴安全帽', 6, 0),
            ('type', 'near_vehicle', '人員靠近車輛', 3, 0),
            ('type', 'no_safety_vest', '未穿安全背心', 0, 0),
        ]
        result = SimpleNamespace(all=lambda: aggregate_rows)
        db = SimpleNamespace(execute=AsyncMock(return_value=result))
        user = SimpleNamespace(role='admin', group_id=1)
        sites = [
            SimpleNamespace(name='Site A'),
            SimpleNamespace(name='Site B'),
        ]
        start = datetime(2026, 8, 20, tzinfo=timezone.utc)
        end = datetime(2026, 8, 24, tzinfo=timezone.utc)

        with patch.object(
            violation_services._user_service,
            'load_user_with_effective_sites',
            new=AsyncMock(return_value=(user, sites)),
        ):
            analytics = await violation_services.get_violation_analytics(
                start,
                end,
                cast(AsyncSession, db),
                _credentials(),
                bucket='day',
            )

        self.assertEqual(analytics.summary.total, 9)
        self.assertEqual(analytics.summary.today, 3)
        assert analytics.summary.top_site is not None
        assert analytics.summary.top_type is not None
        self.assertEqual(analytics.summary.top_site.site_name, 'Site A')
        self.assertEqual(analytics.summary.top_type.type, 'no_safety_helmet')
        self.assertEqual(
            [item.bucket for item in analytics.trend],
            ['2026-08-22', '2026-08-23'],
        )
        self.assertEqual(
            [item.hour for item in analytics.by_hour],
            [8, 18],
        )
        self.assertEqual(len(analytics.by_type), 2)
        db.execute.assert_awaited_once()

    async def test_analytics_rejects_invalid_scope_and_range(self) -> None:
        """Unauthorised users and invalid date ranges receive client errors."""
        start = datetime(2024, 2, 29, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, tzinfo=timezone.utc)
        db = SimpleNamespace(execute=AsyncMock())

        with self.assertRaises(HTTPException) as missing_identity:
            await violation_services.get_violation_analytics(
                start,
                end,
                cast(AsyncSession, db),
                JwtAuthorizationCredentials(
                    subject=cast(
                        AccessTokenSubject, {
                            'username': '',
                            'user_id': 1,
                            'role': 'viewer',
                            'jti': 'empty-user',
                            'features': [],
                        },
                    ),
                ),
            )
        self.assertEqual(missing_identity.exception.status_code, 401)

        with patch.object(
            violation_services._user_service,
            'load_user_with_effective_sites',
            new=AsyncMock(
                return_value=(
                    SimpleNamespace(role='viewer'),
                    [SimpleNamespace(name='Site A')],
                ),
            ),
        ):
            with self.assertRaises(HTTPException) as forbidden:
                await violation_services.get_violation_analytics(
                    start,
                    end,
                    cast(AsyncSession, db),
                    _credentials(),
                )
        self.assertEqual(forbidden.exception.status_code, 403)

        with patch.object(
            violation_services._user_service,
            'load_user_with_effective_sites',
            new=AsyncMock(
                return_value=(
                    SimpleNamespace(role='admin'),
                    [SimpleNamespace(name='Site A')],
                ),
            ),
        ):
            with self.assertRaises(HTTPException) as invalid_range:
                await violation_services.get_violation_analytics(
                    end,
                    start,
                    cast(AsyncSession, db),
                    _credentials(),
                )
        self.assertEqual(invalid_range.exception.status_code, 422)


class TestViolationServiceRemainingBranches(unittest.IsolatedAsyncioTestCase):
    """Verify remaining conversion, filter, and scope helper branches."""

    def test_nullable_conversions_and_invalid_cursor_are_safe(self) -> None:
        """Absent values, empty warnings, and invalid cursors stay harmless."""
        self.assertIsNone(violation_services._decode_detection_items(None))
        self.assertIsNone(violation_services._warning_text_from_json(None))
        self.assertIsNone(
            violation_services._warning_text_from_json(
                '{"warning_no_hardhat": {"count": 0}}',
            ),
        )
        self.assertIsNone(
            violation_services._feedback_detection_ids_from_json(None),
        )
        self.assertIsNone(violation_services._bbox_to_normalized(None, (1, 1)))
        self.assertIsNone(
            violation_services._bbox_to_normalized([2, 0, 1, 1], (1, 1)),
        )
        self.assertFalse(violation_services._bbox_nearly_equal(None, [0, 0]))
        self.assertTrue(
            violation_services._bbox_nearly_equal(
                [0.1, 0.2],
                [0.1000001, 0.2],
            ),
        )
        with self.assertRaises(HTTPException) as invalid_cursor:
            violation_services._decode_violation_cursor('not-a-cursor')
        self.assertEqual(invalid_cursor.exception.status_code, 422)

    async def test_row_mapping_and_loaders_convert_orm_records(self) -> None:
        """Feedback and audit rows are converted through typed projections."""
        timestamp = datetime(2026, 8, 23, tzinfo=timezone.utc)
        feedback = SimpleNamespace(
            id=3,
            violation_id=4,
            feedback_type='wrong_class',
            note='Correct label',
            target_detection_id='det_0',
            original_label='class-2',
            corrected_label='worker',
            original_bbox=[1, 2, 3, 4],
            corrected_bbox=None,
            model_version='v1',
            confidence=0.9,
            status='pending',
            user_id=7,
            created_at=timestamp,
        )
        audit = SimpleNamespace(
            id=5,
            violation_id=4,
            reviewed_by=8,
            action='review_status_changed',
            old_status='pending',
            new_status='resolved',
            review_note='Reviewed',
            flagged_reason='wrong_class',
            reviewed_at=timestamp,
        )
        db = SimpleNamespace(
            execute=AsyncMock(
                side_effect=[
                    SimpleNamespace(
                        scalars=lambda: MagicMock(all=lambda: [audit]),
                    ),
                    SimpleNamespace(
                        scalars=lambda: MagicMock(all=lambda: [feedback]),
                    ),
                ],
            ),
        )

        feedback_item = violation_services._feedback_to_item(
            cast(ViolationFeedback, feedback),
        )
        feedback_response = violation_services._feedback_to_response(
            cast(ViolationFeedback, feedback),
        )
        audit_item = violation_services._review_audit_to_item(
            cast(ViolationReviewAuditLog, audit),
        )
        audits = await violation_services._load_review_audit_logs(
            cast(AsyncSession, db),
            4,
        )
        feedbacks = await violation_services._load_violation_feedbacks(
            cast(AsyncSession, db),
            4,
        )

        self.assertEqual(feedback_item.submitted_by, 7)
        self.assertEqual(feedback_response.violation_id, 4)
        self.assertEqual(audit_item.new_status, 'resolved')
        self.assertEqual(audits[0].id, 5)
        self.assertEqual(feedbacks[0].id, 3)

    async def test_filter_scope_helpers_validate_access(self) -> None:
        """Stream, review, and site helpers reject out-of-scope requests."""
        db = SimpleNamespace(
            execute=AsyncMock(
                return_value=SimpleNamespace(first=lambda: (12, 4, 'Site A')),
            ),
        )
        admin = SimpleNamespace(role='admin', group_id=4)

        resolved = await violation_services._resolve_stream_filter(
            ' 12 ',
            'Site A',
            ['Site A'],
            admin,
            cast(AsyncSession, db),
        )
        self.assertEqual(resolved, (12, 'Site A'))

        with self.assertRaises(HTTPException) as invalid_stream:
            await violation_services._resolve_stream_filter(
                'camera',
                None,
                ['Site A'],
                admin,
                cast(AsyncSession, db),
            )
        self.assertEqual(invalid_stream.exception.status_code, 422)

        with patch.object(
            violation_services._user_service,
            'load_user_with_effective_sites',
            new=AsyncMock(
                return_value=(
                    SimpleNamespace(role='viewer'),
                    [SimpleNamespace(name='Site A')],
                ),
            ),
        ):
            with self.assertRaises(HTTPException) as forbidden_reviewer:
                await violation_services._load_review_scope(
                    'viewer',
                    cast(AsyncSession, db),
                )
        self.assertEqual(forbidden_reviewer.exception.status_code, 403)

        with patch.object(
            violation_services,
            '_load_review_scope',
            new=AsyncMock(return_value=(admin, ['Site A'])),
        ):
            review_sites = await violation_services._violation_site_names(
                'admin',
                True,
                None,
                cast(AsyncSession, db),
            )
        self.assertEqual(review_sites, ['Site A'])

    async def test_filter_builders_compose_authorised_conditions(self) -> None:
        """Builder helpers combine scope, text, time, cursor, and type filters.

        This ensures all authorised conditions are composed once.
        """
        timestamp = datetime(2026, 8, 23, tzinfo=timezone.utc)
        db = SimpleNamespace(
            execute=AsyncMock(
                return_value=SimpleNamespace(
                    scalar_one_or_none=lambda: 'Site A',
                ),
            ),
        )
        cursor = violation_services._encode_violation_cursor(
            ViolationListItem(
                id=4,
                site_name='Site A',
                stream_name='Camera A',
                detection_time=timestamp,
                thumbnail_url='https://api.example.test/thumbnail',
            ),
        )

        with (
            patch.object(
                violation_services,
                '_stream_violation_conditions',
                new=AsyncMock(return_value=[]),
            ),
            patch.object(
                violation_services,
                '_search_util',
                return_value=SimpleNamespace(
                    expand_synonyms=lambda value: [value, 'helmet'],
                ),
            ),
        ):
            conditions = await violation_services._build_violation_conditions(
                'admin',
                ['Site A'],
                1,
                None,
                'no_safety_helmet',
                'hard hat',
                timestamp,
                timestamp,
                True,
                'pending',
                cursor,
                cast(AsyncSession, db),
            )

        self.assertGreater(len(conditions), 5)
        self.assertIsNone(violation_services._violation_keyword_condition(''))

    def test_conversion_and_list_helpers_cover_remaining_payload_shapes(
        self,
    ) -> None:
        """Conversions preserve ratios, list fields, and feedback overlays.

        Invalid feedback bounding boxes are omitted from the response.
        """
        timestamp = datetime(2026, 8, 23, tzinfo=timezone.utc)
        ratio_bbox = violation_services._bbox_to_normalized(
            [0.1, 0.2, 0.8, 0.9],
            None,
        )
        invalid_size_bbox = violation_services._bbox_to_normalized(
            [0, 0, 20, 20],
            (0, 1),
        )
        feedback = ViolationFeedbackItem(
            id=8,
            type='wrong_class',
            note='Reclassified',
            original_bbox=[10, 20, 60, 80],
            status='pending',
            submitted_at=timestamp,
        )
        false_negative = ViolationFeedbackItem(
            id=9,
            type='false_negative',
            corrected_bbox=[20, 20, 10, 30],
            status='pending',
            submitted_at=timestamp,
        )
        overlays = violation_services._overlay_objects_from_feedback(
            [
                FeedbackDetectionItem(
                    id='det_0',
                    label='worker',
                    confidence=0.9,
                    bbox=[10, 20, 60, 80],
                ),
            ],
            [feedback, false_negative],
            (100, 100),
        )
        list_item = violation_services._violation_to_list_item(
            (
                11,
                'Site A',
                'Camera A',
                timestamp,
                'evidence.jpg',
                '{"warning_no_hardhat": {"count": 2}}',
                True,
                'pending',
                'Needs review',
            ),
            _Request(),
        )

        assert ratio_bbox is not None
        self.assertAlmostEqual(ratio_bbox.w, 0.7)
        self.assertIsNone(invalid_size_bbox)
        self.assertTrue(overlays[0].is_flagged)
        self.assertEqual(overlays[0].flag_note, 'Reclassified')
        self.assertEqual(list_item.id, 11)
        self.assertEqual(list_item.warning_text, 'warning_no_hardhat: 2')

    async def test_scope_helpers_cover_empty_and_unauthorised_filters(
        self,
    ) -> None:
        """Scope helpers return empty filters and reject out-of-scope sites."""
        db = SimpleNamespace(
            execute=AsyncMock(
                return_value=SimpleNamespace(
                    scalar_one_or_none=lambda: 'Other Site',
                ),
            ),
        )
        with self.assertRaises(HTTPException) as forbidden_site:
            await violation_services._filtered_violation_site_name(
                3,
                ['Site A'],
                cast(AsyncSession, db),
            )
        self.assertEqual(forbidden_site.exception.status_code, 403)
        self.assertIsNone(
            await violation_services._filtered_violation_site_name(
                None,
                ['Site A'],
                cast(AsyncSession, db),
            ),
        )
        self.assertEqual(
            await violation_services._stream_violation_conditions(
                'admin',
                None,
                None,
                ['Site A'],
                cast(AsyncSession, db),
            ),
            [],
        )
        with (
            patch.object(
                violation_services._user_service,
                'load_user_with_effective_sites',
                new=AsyncMock(
                    return_value=(
                        SimpleNamespace(role='admin', group_id=1),
                        [SimpleNamespace(name='Site A')],
                    ),
                ),
            ),
            patch.object(
                violation_services,
                '_resolve_stream_filter',
                new=AsyncMock(return_value=(12, 'Site A')),
            ),
        ):
            stream_conditions = (
                await violation_services._stream_violation_conditions(
                    'admin',
                    '12',
                    None,
                    ['Site A'],
                    cast(AsyncSession, db),
                )
            )
        self.assertEqual(len(stream_conditions), 2)

        with patch.object(
            violation_services._user_service,
            'get_cached_effective_site_names',
            new=AsyncMock(return_value=['Site A']),
        ):
            site_names = await violation_services._violation_site_names(
                'viewer',
                None,
                None,
                cast(AsyncSession, db),
            )
        self.assertEqual(site_names, ['Site A'])

    async def test_analytics_handles_empty_scope_leap_year_and_bad_type(
        self,
    ) -> None:
        """Analytics avoids SQL for empty scopes and validates range/type.

        Leap-year boundaries still enforce the five-year maximum.
        """
        start = datetime(2024, 2, 29, tzinfo=timezone.utc)
        db = SimpleNamespace(execute=AsyncMock())
        admin = SimpleNamespace(role='admin', group_id=1)

        with patch.object(
            violation_services._user_service,
            'load_user_with_effective_sites',
            new=AsyncMock(return_value=(admin, [])),
        ):
            empty = await violation_services.get_violation_analytics(
                start,
                datetime(2024, 3, 1, tzinfo=timezone.utc),
                cast(AsyncSession, db),
                _credentials(),
            )
        self.assertEqual(empty.summary.total, 0)
        db.execute.assert_not_awaited()

        with patch.object(
            violation_services._user_service,
            'load_user_with_effective_sites',
            new=AsyncMock(
                return_value=(
                admin, [SimpleNamespace(name='Site A')],
                ),
            ),
        ):
            with self.assertRaises(HTTPException) as long_range:
                await violation_services.get_violation_analytics(
                    start,
                    datetime(2029, 3, 1, tzinfo=timezone.utc),
                    cast(AsyncSession, db),
                    _credentials(),
                )
            with self.assertRaises(HTTPException) as invalid_type:
                await violation_services.get_violation_analytics(
                    start,
                    datetime(2024, 3, 1, tzinfo=timezone.utc),
                    cast(AsyncSession, db),
                    _credentials(),
                    violation_type='not-a-type',
                )
        self.assertEqual(long_range.exception.status_code, 422)
        self.assertEqual(invalid_type.exception.status_code, 422)

    def test_search_utility_is_created_once_for_keyword_queries(self) -> None:
        """The CPU keyword utility is initialised only for non-empty queries.

        Repeated searches reuse the cached utility instance.
        """
        violation_services._search_util.cache_clear()
        search_utility = MagicMock()
        search_utility.expand_synonyms.return_value = ['hard hat']
        with patch.object(
            violation_services,
            'SearchUtils',
            return_value=search_utility,
        ):
            condition = violation_services._violation_keyword_condition(
                'hard hat',
            )
            repeated = violation_services._violation_keyword_condition(
                'hard hat',
            )
        violation_services._search_util.cache_clear()

        self.assertIsNotNone(condition)
        self.assertIsNotNone(repeated)
        search_utility.expand_synonyms.assert_called()

    async def test_review_scope_stream_scope_and_empty_analytics_results(
        self,
    ) -> None:
        """Review helpers enforce stream scope and support empty analytics.

        Site, stream, and type filters remain constrained to allowed sites.
        """
        reviewer = SimpleNamespace(role='admin', group_id=1)
        with patch.object(
            violation_services._user_service,
            'load_user_with_effective_sites',
            new=AsyncMock(
                return_value=(
                    reviewer,
                    [SimpleNamespace(name='Site A')],
                ),
            ),
        ):
            review_scope = await violation_services._load_review_scope(
                'admin',
                cast(AsyncSession, SimpleNamespace()),
            )
        loaded_user, review_sites = review_scope
        self.assertIs(loaded_user, reviewer)
        self.assertEqual(review_sites, ['Site A'])

        denied_stream_db = SimpleNamespace(
            execute=AsyncMock(
                return_value=SimpleNamespace(first=lambda: (12, 2, 'Site A')),
            ),
        )
        with self.assertRaises(HTTPException) as denied_stream:
            await violation_services._resolve_stream_filter(
                '12',
                None,
                ['Site A'],
                reviewer,
                cast(AsyncSession, denied_stream_db),
            )
        self.assertEqual(denied_stream.exception.status_code, 403)

        denied_site_db = SimpleNamespace(
            execute=AsyncMock(
                return_value=SimpleNamespace(
                    scalar_one_or_none=lambda: 'Site B',
                ),
            ),
        )
        with (
            patch.object(
                violation_services._user_service,
                'load_user_with_effective_sites',
                new=AsyncMock(
                    return_value=(
                        reviewer,
                        [SimpleNamespace(name='Site A')],
                    ),
                ),
            ),
            self.assertRaises(HTTPException) as denied_site,
        ):
            await violation_services.get_violation_analytics(
                datetime(2026, 8, 1, tzinfo=timezone.utc),
                datetime(2026, 8, 2, tzinfo=timezone.utc),
                cast(AsyncSession, denied_site_db),
                _credentials(),
                site_id=2,
            )
        self.assertEqual(denied_site.exception.status_code, 403)

        aggregate = SimpleNamespace(
            all=lambda: [('summary', None, None, 0, 0)],
        )
        site_db = SimpleNamespace(
            execute=AsyncMock(
                side_effect=[
                    SimpleNamespace(scalar_one_or_none=lambda: 'Site A'),
                    aggregate,
                ],
            ),
        )
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        end = datetime(2026, 8, 2, tzinfo=timezone.utc)
        with patch.object(
            violation_services._user_service,
            'load_user_with_effective_sites',
            new=AsyncMock(
                return_value=(
                    reviewer,
                    [SimpleNamespace(name='Site A')],
                ),
            ),
        ):
            empty = await violation_services.get_violation_analytics(
                start,
                end,
                cast(AsyncSession, site_db),
                _credentials(),
                site_id=1,
                violation_type='no_safety_helmet',
            )
        self.assertEqual(empty.summary.total, 0)

        stream_db = SimpleNamespace(
            execute=AsyncMock(
                side_effect=[
                    SimpleNamespace(first=lambda: (12, 1, 'Site A')),
                    aggregate,
                ],
            ),
        )
        with patch.object(
            violation_services._user_service,
            'load_user_with_effective_sites',
            new=AsyncMock(
                return_value=(
                    reviewer,
                    [SimpleNamespace(name='Site A')],
                ),
            ),
        ):
            stream_empty = await violation_services.get_violation_analytics(
                start,
                end,
                cast(AsyncSession, stream_db),
                _credentials(),
                stream_id='12',
            )
        self.assertEqual(stream_empty.summary.total, 0)


if __name__ == '__main__':
    unittest.main()
