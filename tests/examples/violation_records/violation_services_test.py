from __future__ import annotations

import json
import unittest
from datetime import datetime
from datetime import timezone
from functools import partial
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth import user_service
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import User
from examples.auth.models import Violation
from examples.auth.models import ViolationFeedback
from examples.auth.models import ViolationReviewAuditLog
from examples.db_management.schemas.auth import AccessTokenSubject
from examples.violation_records import violation_queries
from examples.violation_records import violation_responses
from examples.violation_records import violation_services
from examples.violation_records.schemas import FeedbackDetectionItem
from examples.violation_records.schemas import ViolationFeedbackItem
from examples.violation_records.schemas import ViolationListItem
from examples.violation_records.violation_queries import ViolationFilters


def _analytics_maps_and_sorts_one_all(aggregate_rows, /):
    return aggregate_rows


def _filter_scope_helpers_validate_access_first():
    return (12, 4, 'Site A')


def _filter_site():
    return 'Site A'


def _expand_filter_synonyms(value):
    return [value, 'helmet']


def _other_site():
    return 'Other Site'


def _review_scope_stream_scope_and_first():
    return (12, 2, 'Site A')


def _review_site():
    return 'Site B'


def _review_scope_stream_scope_and_all():
    return [('summary', None, None, 0, 0)]


def _review_scope_stream_scope_and_first2():
    return (12, 1, 'Site A')


class _RowMappingAndLoadersConvertOrmRecordsCallbacks:
    audit: Any
    feedback: Any

    def _call_make_magicmock(self):
        _callbacks = _RowMappingAndLoadersConvertOrmRecordsCallbacks2()
        _callbacks.parent = self
        return MagicMock(all=_callbacks._call_all)

    def _call_make_magicmock2(self):
        _callbacks = _RowMappingAndLoadersConvertOrmRecordsCallbacks3()
        _callbacks.parent = self
        return MagicMock(all=_callbacks._call_all)


class _RowMappingAndLoadersConvertOrmRecordsCallbacks2:
    parent: _RowMappingAndLoadersConvertOrmRecordsCallbacks

    def _call_all(self):
        return [self.parent.audit]


class _RowMappingAndLoadersConvertOrmRecordsCallbacks3:
    parent: _RowMappingAndLoadersConvertOrmRecordsCallbacks

    def _call_all(self):
        return [self.parent.feedback]


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

    def test_list_preserves_active_warning_keys_for_frontend_localisation(
        self,
    ) -> None:
        """Compact list JSON supplies the same warning keys as detail JSON."""
        active = {
            'warning_no_hardhat': {'count': 2},
            'warning_no_safety_vest': {'count': 1},
            'warning_close_to_vehicle': {'count': 3},
            'warning_close_to_machinery': {'count': 1},
            'warning_people_in_controlled_area': {'count': 4},
            'warning_people_in_utility_pole_controlled_area': {'count': 1},
            'detect_machinery_close_to_pole': {'count': 2},
        }
        cases = [
            (json.dumps(active), active),
            (
                json.dumps(
                    {
                        'warning_no_hardhat': {'count': 2},
                        'warning_no_safety_vest': {'count': 0},
                    },
                ),
                {'warning_no_hardhat': {'count': 2}},
            ),
            ('{"warning_no_hardhat": {"count": 0}}', {}),
            ('{}', {}),
            (None, {}),
        ]
        for stored, expected in cases:
            with self.subTest(stored=stored):
                item = violation_responses.violation_to_list_item(
                    (
                        7,
                        'Site A',
                        'Camera A',
                        datetime(2026, 10, 11, tzinfo=timezone.utc),
                        'evidence.jpg',
                        stored,
                        False,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ),
                    _Request(),
                )
                payload = item.model_dump(mode='json')
                self.assertEqual(payload['warnings'], expected)
                self.assertEqual(
                    item.warning_text,
                    violation_responses.warning_text_from_json(stored),
                )
                self.assertNotIn('detection_items', payload)
                self.assertNotIn('image_path', payload)

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

        item = violation_responses.violation_to_detail_item(
            cast(Violation, violation),
            _Request(),
            'Needs review',
        )
        overlays = violation_responses.overlay_objects_from_feedback(
            item.detections,
            [feedback],
            (100, 100),
        )

        self.assertEqual(item.warning_text, 'warning_no_hardhat: 2')
        assert item.detections is not None
        self.assertEqual(item.detections[0].id, 'det_0')
        self.assertEqual(
            item.image_url,
            (
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

        overlays = violation_responses.overlay_objects_from_feedback(
            detections,
            feedbacks,
            (100, 100),
        )

        self.assertEqual(
            [overlay.object_id for overlay in overlays],
            [
                'feedback_1',
            ],
        )
        self.assertEqual(overlays[0].bbox.w, 0.4)
        self.assertIsNone(
            violation_responses.bbox_to_normalized([2, 2, 1, 3], (10, 10)),
        )
        self.assertIsNone(
            violation_responses.bbox_to_normalized([2, 2, 3, 3], None),
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

        cursor = violation_queries.encode_violation_cursor(item)
        decoded_time, decoded_id = violation_queries.decode_violation_cursor(
            cursor,
        )

        self.assertEqual(decoded_time, item.detection_time)
        self.assertEqual(decoded_id, 12)
        self.assertIsNotNone(
            violation_queries.optional_violation_conditions(
                violation_queries.ViolationFilters(
                    violation_type='no_safety_helmet',
                    keyword=None,
                    start_time=item.detection_time,
                    end_time=item.detection_time,
                    cursor=cursor,
                ),
            ),
        )
        with self.assertRaises(HTTPException) as raised:
            violation_queries.optional_violation_conditions(
                violation_queries.ViolationFilters(
                    violation_type='not-a-real-type',
                    keyword=None,
                    start_time=None,
                    end_time=None,
                    cursor=None,
                ),
            )
        self.assertEqual(raised.exception.status_code, 422)


class TestViolationAnalytics(unittest.IsolatedAsyncioTestCase):
    """Verify PostgreSQL aggregate result normalisation and validation."""

    async def test_analytics_missing_local_account_still_returns_401(self):
        with patch.object(
            violation_services,
            'load_violation_scope',
            AsyncMock(side_effect=HTTPException(404, 'User not found')),
        ):
            with self.assertRaises(HTTPException) as raised:
                await violation_services.get_violation_analytics(
                    ViolationFilters(),
                    AsyncMock(),
                    _credentials(),
                )
        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(raised.exception.detail, 'Invalid user')

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
        result = SimpleNamespace(
            all=partial(_analytics_maps_and_sorts_one_all, aggregate_rows),
        )
        db = SimpleNamespace(execute=AsyncMock(return_value=result))
        user = SimpleNamespace(role='admin', group_id=1)
        sites = [
            SimpleNamespace(name='Site A'),
            SimpleNamespace(name='Site B'),
        ]
        start = datetime(2026, 8, 20, tzinfo=timezone.utc)
        end = datetime(2026, 8, 24, tzinfo=timezone.utc)

        with patch.object(
            user_service,
            'load_user_with_effective_sites',
            new=AsyncMock(return_value=(user, sites)),
        ):
            analytics = await violation_services.get_violation_analytics(
                ViolationFilters(start_time=start, end_time=end),
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
                ViolationFilters(start_time=start, end_time=end),
                cast(AsyncSession, db),
                JwtAuthorizationCredentials(
                    subject=cast(
                        AccessTokenSubject,
                        {
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
            user_service,
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
                    ViolationFilters(start_time=start, end_time=end),
                    cast(AsyncSession, db),
                    _credentials(),
                )
        self.assertEqual(forbidden.exception.status_code, 403)

        with patch.object(
            user_service,
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
                    ViolationFilters(start_time=end, end_time=start),
                    cast(AsyncSession, db),
                    _credentials(),
                )
        self.assertEqual(invalid_range.exception.status_code, 422)


class TestViolationServiceRemainingBranches(unittest.IsolatedAsyncioTestCase):
    """Verify remaining conversion, filter, and scope helper branches."""

    def test_nullable_conversions_and_invalid_cursor_are_safe(self) -> None:
        """Absent values, empty warnings, and invalid cursors stay harmless."""
        self.assertIsNone(violation_responses.decode_detection_items(None))
        self.assertIsNone(violation_responses.warning_text_from_json(None))
        self.assertIsNone(
            violation_responses.warning_text_from_json(
                '{"warning_no_hardhat": {"count": 0}}',
            ),
        )
        self.assertIsNone(
            violation_responses.feedback_detection_ids_from_json(None),
        )
        self.assertIsNone(violation_responses.bbox_to_normalized(None, (1, 1)))
        self.assertIsNone(
            violation_responses.bbox_to_normalized([2, 0, 1, 1], (1, 1)),
        )
        self.assertFalse(violation_responses.bbox_nearly_equal(None, [0, 0]))
        self.assertTrue(
            violation_responses.bbox_nearly_equal(
                [0.1, 0.2],
                [0.1000001, 0.2],
            ),
        )
        with self.assertRaises(HTTPException) as invalid_cursor:
            violation_queries.decode_violation_cursor('not-a-cursor')
        self.assertEqual(invalid_cursor.exception.status_code, 422)

    async def test_row_mapping_and_loaders_convert_orm_records(self) -> None:
        """Feedback and audit rows are converted through typed projections."""
        _callbacks = _RowMappingAndLoadersConvertOrmRecordsCallbacks()
        timestamp = datetime(2026, 8, 23, tzinfo=timezone.utc)
        _callbacks.feedback = SimpleNamespace(
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
        _callbacks.audit = SimpleNamespace(
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
                        scalars=_callbacks._call_make_magicmock,
                    ),
                    SimpleNamespace(
                        scalars=_callbacks._call_make_magicmock2,
                    ),
                ],
            ),
        )

        feedback_item = violation_responses.feedback_to_item(
            cast(ViolationFeedback, _callbacks.feedback),
        )
        feedback_response = violation_responses.feedback_to_response(
            cast(ViolationFeedback, _callbacks.feedback),
        )
        audit_item = violation_responses.review_audit_to_item(
            cast(ViolationReviewAuditLog, _callbacks.audit),
        )
        audits = await violation_queries.load_review_audit_logs(
            cast(AsyncSession, db),
            4,
        )
        feedbacks = await violation_queries.load_violation_feedbacks(
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
                return_value=SimpleNamespace(
                    first=_filter_scope_helpers_validate_access_first,
                ),
            ),
        )
        admin = SimpleNamespace(role='admin', group_id=4)

        resolved = await violation_queries.resolve_stream_filter(
            ' 12 ',
            'Site A',
            ['Site A'],
            admin,
            cast(AsyncSession, db),
        )
        self.assertEqual(resolved, (12, 'Site A'))

        with self.assertRaises(HTTPException) as invalid_stream:
            await violation_queries.resolve_stream_filter(
                'camera',
                None,
                ['Site A'],
                admin,
                cast(AsyncSession, db),
            )
        self.assertEqual(invalid_stream.exception.status_code, 422)

        with patch.object(
            user_service,
            'load_user_with_effective_sites',
            new=AsyncMock(
                return_value=(
                    SimpleNamespace(role='viewer'),
                    [SimpleNamespace(name='Site A')],
                ),
            ),
        ):
            with self.assertRaises(HTTPException) as forbidden_reviewer:
                await violation_queries.load_review_scope(
                    'viewer',
                    cast(AsyncSession, db),
                )
        self.assertEqual(forbidden_reviewer.exception.status_code, 403)

        with patch.object(
            violation_queries,
            'load_review_scope',
            new=AsyncMock(return_value=(admin, ['Site A'])),
        ):
            review_sites = await violation_queries.load_violation_scope(
                'admin',
                violation_queries.ViolationFilters(
                    flagged=True,
                    review_status=None,
                ),
                cast(AsyncSession, db),
            )
        self.assertEqual(review_sites.site_names, ['Site A'])

    async def test_filter_builders_compose_authorised_conditions(self) -> None:
        """Builder helpers combine scope, text, time, cursor, and type filters.

        This ensures all authorised conditions are composed once.
        """
        timestamp = datetime(2026, 8, 23, tzinfo=timezone.utc)
        db = SimpleNamespace(
            execute=AsyncMock(
                return_value=SimpleNamespace(
                    scalar_one_or_none=_filter_site,
                ),
            ),
        )
        cursor = violation_queries.encode_violation_cursor(
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
                violation_queries,
                'stream_violation_conditions',
                new=AsyncMock(return_value=[]),
            ),
            patch.object(
                violation_queries,
                'search_util',
                return_value=SimpleNamespace(
                    expand_synonyms=_expand_filter_synonyms,
                ),
            ),
        ):
            conditions = await violation_queries.build_violation_conditions(
                violation_queries.ViolationScope(
                    User(role='admin', group_id=1),
                    ['Site A'],
                ),
                violation_queries.ViolationFilters(
                    site_id=1,
                    stream_id=None,
                    violation_type='no_safety_helmet',
                    keyword='hard hat',
                    start_time=timestamp,
                    end_time=timestamp,
                    flagged=True,
                    review_status='pending',
                    cursor=cursor,
                ),
                cast(AsyncSession, db),
            )

        self.assertGreater(len(conditions), 5)
        self.assertIsNone(violation_queries.violation_keyword_condition(''))

    def test_conversion_and_list_helpers_cover_remaining_payload_shapes(
        self,
    ) -> None:
        """Conversions preserve ratios, list fields, and feedback overlays.

        Invalid feedback bounding boxes are omitted from the response.
        """
        timestamp = datetime(2026, 8, 23, tzinfo=timezone.utc)
        ratio_bbox = violation_responses.bbox_to_normalized(
            [0.1, 0.2, 0.8, 0.9],
            None,
        )
        invalid_size_bbox = violation_responses.bbox_to_normalized(
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
        overlays = violation_responses.overlay_objects_from_feedback(
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
        list_item = violation_responses.violation_to_list_item(
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
                None,
                None,
                None,
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
                    scalar_one_or_none=_other_site,
                ),
            ),
        )
        with self.assertRaises(HTTPException) as forbidden_site:
            await violation_queries.filtered_violation_site_name(
                3,
                ['Site A'],
                cast(AsyncSession, db),
            )
        self.assertEqual(forbidden_site.exception.status_code, 403)
        self.assertIsNone(
            await violation_queries.filtered_violation_site_name(
                None,
                ['Site A'],
                cast(AsyncSession, db),
            ),
        )
        self.assertEqual(
            await violation_queries.stream_violation_conditions(
                violation_queries.ViolationScope(
                    User(role='admin', group_id=1),
                    ['Site A'],
                ),
                None,
                None,
                cast(AsyncSession, db),
            ),
            [],
        )
        with (
            patch.object(
                user_service,
                'load_user_with_effective_sites',
                new=AsyncMock(
                    return_value=(
                        User(role='admin', group_id=1),
                        [SimpleNamespace(name='Site A')],
                    ),
                ),
            ),
            patch.object(
                violation_queries,
                'resolve_stream_filter',
                new=AsyncMock(return_value=(12, 'Site A')),
            ),
        ):
            stream_conditions = (
                await violation_queries.stream_violation_conditions(
                    violation_queries.ViolationScope(
                        User(role='admin', group_id=1),
                        ['Site A'],
                    ),
                    '12',
                    None,
                    cast(AsyncSession, db),
                )
            )
        self.assertEqual(len(stream_conditions), 2)

        with patch.object(
            user_service,
            'load_user_with_effective_sites',
            new=AsyncMock(
                return_value=(
                    SimpleNamespace(role='viewer'),
                    [SimpleNamespace(name='Site A')],
                ),
            ),
        ):
            site_names = await violation_queries.load_violation_scope(
                'viewer',
                violation_queries.ViolationFilters(
                    flagged=None,
                    review_status=None,
                ),
                cast(AsyncSession, db),
            )
        self.assertEqual(site_names.site_names, ['Site A'])

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
            user_service,
            'load_user_with_effective_sites',
            new=AsyncMock(return_value=(admin, [])),
        ):
            empty = await violation_services.get_violation_analytics(
                ViolationFilters(
                    start_time=start,
                    end_time=datetime(2024, 3, 1, tzinfo=timezone.utc),
                ),
                cast(AsyncSession, db),
                _credentials(),
            )
        self.assertEqual(empty.summary.total, 0)
        db.execute.assert_not_awaited()

        with patch.object(
            user_service,
            'load_user_with_effective_sites',
            new=AsyncMock(
                return_value=(
                    admin,
                    [SimpleNamespace(name='Site A')],
                ),
            ),
        ):
            with self.assertRaises(HTTPException) as long_range:
                await violation_services.get_violation_analytics(
                    ViolationFilters(
                        start_time=start,
                        end_time=datetime(2029, 3, 1, tzinfo=timezone.utc),
                    ),
                    cast(AsyncSession, db),
                    _credentials(),
                )
            with self.assertRaises(HTTPException) as invalid_type:
                await violation_services.get_violation_analytics(
                    ViolationFilters(
                        start_time=start,
                        end_time=datetime(2024, 3, 1, tzinfo=timezone.utc),
                        violation_type='not-a-type',
                    ),
                    cast(AsyncSession, db),
                    _credentials(),
                )
        self.assertEqual(long_range.exception.status_code, 422)
        self.assertEqual(invalid_type.exception.status_code, 422)

    def test_search_utility_is_created_once_for_keyword_queries(self) -> None:
        """The CPU keyword utility is initialised only for non-empty queries.

        Repeated searches reuse the cached utility instance.
        """
        violation_queries.search_util.cache_clear()
        search_utility = MagicMock()
        search_utility.expand_synonyms.return_value = ['hard hat']
        with patch.object(
            violation_queries,
            'SearchUtils',
            return_value=search_utility,
        ):
            condition = violation_queries.violation_keyword_condition(
                'hard hat',
            )
            repeated = violation_queries.violation_keyword_condition(
                'hard hat',
            )
        violation_queries.search_util.cache_clear()

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
            user_service,
            'load_user_with_effective_sites',
            new=AsyncMock(
                return_value=(
                    reviewer,
                    [SimpleNamespace(name='Site A')],
                ),
            ),
        ):
            review_scope = await violation_queries.load_review_scope(
                'admin',
                cast(AsyncSession, SimpleNamespace()),
            )
        loaded_user, review_sites = review_scope
        self.assertIs(loaded_user, reviewer)
        self.assertEqual(review_sites, ['Site A'])

        denied_stream_db = SimpleNamespace(
            execute=AsyncMock(
                return_value=SimpleNamespace(
                    first=_review_scope_stream_scope_and_first,
                ),
            ),
        )
        with self.assertRaises(HTTPException) as denied_stream:
            await violation_queries.resolve_stream_filter(
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
                    scalar_one_or_none=_review_site,
                ),
            ),
        )
        with (
            patch.object(
                user_service,
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
                ViolationFilters(
                    start_time=datetime(2026, 8, 1, tzinfo=timezone.utc),
                    end_time=datetime(2026, 8, 2, tzinfo=timezone.utc),
                    site_id=2,
                ),
                cast(AsyncSession, denied_site_db),
                _credentials(),
            )
        self.assertEqual(denied_site.exception.status_code, 403)

        aggregate = SimpleNamespace(
            all=_review_scope_stream_scope_and_all,
        )
        site_db = SimpleNamespace(
            execute=AsyncMock(
                side_effect=[
                    SimpleNamespace(
                        scalar_one_or_none=_filter_site,
                    ),
                    aggregate,
                ],
            ),
        )
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        end = datetime(2026, 8, 2, tzinfo=timezone.utc)
        with patch.object(
            user_service,
            'load_user_with_effective_sites',
            new=AsyncMock(
                return_value=(
                    reviewer,
                    [SimpleNamespace(name='Site A')],
                ),
            ),
        ):
            empty = await violation_services.get_violation_analytics(
                ViolationFilters(
                    start_time=start,
                    end_time=end,
                    site_id=1,
                    violation_type='no_safety_helmet',
                ),
                cast(AsyncSession, site_db),
                _credentials(),
            )
        self.assertEqual(empty.summary.total, 0)

        stream_db = SimpleNamespace(
            execute=AsyncMock(
                side_effect=[
                    SimpleNamespace(
                        first=_review_scope_stream_scope_and_first2,
                    ),
                    aggregate,
                ],
            ),
        )
        with patch.object(
            user_service,
            'load_user_with_effective_sites',
            new=AsyncMock(
                return_value=(
                    reviewer,
                    [SimpleNamespace(name='Site A')],
                ),
            ),
        ):
            stream_empty = await violation_services.get_violation_analytics(
                ViolationFilters(
                    start_time=start,
                    end_time=end,
                    stream_id='12',
                ),
                cast(AsyncSession, stream_db),
                _credentials(),
            )
        self.assertEqual(stream_empty.summary.total, 0)


if __name__ == '__main__':
    unittest.main()
