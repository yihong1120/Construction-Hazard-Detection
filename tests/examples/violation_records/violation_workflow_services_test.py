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
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import Violation
from examples.db_management.schemas.auth import AccessTokenSubject
from examples.violation_records import violation_query_service
from examples.violation_records import violation_review_service
from examples.violation_records.schemas import ViolationFeedbackCreate
from examples.violation_records.schemas import ViolationItem
from examples.violation_records.schemas import ViolationListItem
from examples.violation_records.schemas import ViolationReviewUpdate


def _credentials() -> JwtAuthorizationCredentials:
    """Build credentials for a violation workflow service test.

    Returns:
        Minimal access-token credentials for an administrator.
    """
    return JwtAuthorizationCredentials(
        subject=cast(
            AccessTokenSubject,
            {
                'username': 'reviewer',
                'user_id': 8,
                'role': 'admin',
                'jti': 'violation-workflow-test',
                'features': [],
            },
        ),
    )


def _detail_item() -> ViolationItem:
    """Build a compact valid detail response for delegation assertions.

    Returns:
        Valid response model for one reviewable violation.
    """
    timestamp = datetime(2026, 8, 23, tzinfo=timezone.utc)
    return ViolationItem(
        id=4,
        site_name='Site A',
        stream_name='Camera A',
        detection_time=timestamp,
        image_path='evidence.jpg',
        created_at=timestamp,
        is_flagged=True,
        review_status='pending',
    )


class TestViolationQueryService(unittest.IsolatedAsyncioTestCase):
    """Verify query services project only their authorised data."""

    async def test_sites_and_filter_options_map_authorised_rows(self) -> None:
        """Site records and camera options retain only public attributes."""
        timestamp = datetime(2026, 8, 23, tzinfo=timezone.utc)
        sites = [
            SimpleNamespace(
                id=1,
                name='Site A',
                created_at=timestamp,
                updated_at=timestamp,
            ),
        ]
        site = SimpleNamespace(id=1, name='Site A')
        user = SimpleNamespace(role='super_admin', group_id=None)
        camera_result = SimpleNamespace(all=lambda: [(12, 'Camera A')])
        db = SimpleNamespace(
            scalar=AsyncMock(return_value=site),
            execute=AsyncMock(return_value=camera_result),
        )

        with (
            patch.object(
                violation_query_service.user_service,
                'load_user_with_effective_sites',
                new=AsyncMock(return_value=(user, sites)),
            ),
            patch.object(
                violation_query_service.user_service,
                'load_user_access_context',
                new=AsyncMock(return_value=(user, ['Site A'], None)),
            ),
        ):
            my_sites = await violation_query_service.get_my_sites(
                cast(AsyncSession, db),
                _credentials(),
            )
            options = (
                await violation_query_service.get_violation_filter_options(
                    1,
                    None,
                    cast(AsyncSession, db),
                    _credentials(),
                )
            )

        self.assertEqual(my_sites[0].name, 'Site A')
        self.assertEqual(options.cameras[0].stream_id, '12')
        self.assertEqual(options.cameras[0].name, 'Camera A')
        self.assertGreater(len(options.violation_types), 1)

    async def test_filter_options_without_a_site_use_authorised_scope(
        self,
    ) -> None:
        """The initial filter state exposes cameras from accessible sites."""
        user = SimpleNamespace(role='super_admin', group_id=None)
        camera_result = SimpleNamespace(all=lambda: [(12, 'Camera A')])
        db = SimpleNamespace(
            scalar=AsyncMock(),
            execute=AsyncMock(return_value=camera_result),
        )

        with patch.object(
            violation_query_service.user_service,
            'load_user_access_context',
            new=AsyncMock(return_value=(user, ['Site A'], None)),
        ):
            options = (
                await violation_query_service.get_violation_filter_options(
                    None,
                    None,
                    cast(AsyncSession, db),
                    _credentials(),
                )
            )

        self.assertEqual(options.cameras[0].stream_id, '12')
        db.scalar.assert_not_awaited()

    async def test_get_violations_builds_a_keyset_page(self) -> None:
        """The list service maps only the requested page and next cursor."""
        timestamp = datetime(2026, 8, 23, tzinfo=timezone.utc)
        first = ViolationListItem(
            id=5,
            site_name='Site A',
            stream_name='Camera A',
            detection_time=timestamp,
            thumbnail_url='https://api.example.test/thumbnail/5',
        )
        second = ViolationListItem(
            id=4,
            site_name='Site A',
            stream_name='Camera A',
            detection_time=timestamp,
            thumbnail_url='https://api.example.test/thumbnail/4',
        )
        result = SimpleNamespace(
            all=lambda: [
                (
                    5, 'Site A', 'Camera A', timestamp, '5.jpg', None, False,
                    None, None,
                ),
                (
                    4, 'Site A', 'Camera A', timestamp, '4.jpg', None, False,
                    None, None,
                ),
            ],
        )
        db = SimpleNamespace(execute=AsyncMock(return_value=result))

        with (
            patch.object(
                violation_query_service,
                '_violation_site_names',
                new=AsyncMock(return_value=['Site A']),
            ),
            patch.object(
                violation_query_service,
                '_build_violation_conditions',
                new=AsyncMock(return_value=[Violation.id == 5]),
            ),
            patch.object(
                violation_query_service,
                '_violation_to_list_item',
                side_effect=[first, second],
            ),
        ):
            page = await violation_query_service.get_violations(
                MagicMock(),
                cast(AsyncSession, db),
                _credentials(),
                limit=1,
            )

        self.assertTrue(page.has_more)
        self.assertEqual([item.id for item in page.items], [5])
        self.assertIsNotNone(page.next_cursor)
        db.execute.assert_awaited_once()


class TestViolationReviewService(unittest.IsolatedAsyncioTestCase):
    """Verify feedback and review state changes use one transaction."""

    async def test_submit_feedback_flags_the_violation_and_returns_response(
        self,
    ) -> None:
        """Valid targeted feedback stages one flagged feedback record."""
        violation = SimpleNamespace(
            id=4,
            site='Site A',
            detections_json='[[1, 2, 3, 4, 0.8, 1, 9]]',
            is_flagged=False,
            flag_reason=None,
            flagged_by=None,
            flagged_at=None,
            review_status=None,
        )
        db = SimpleNamespace(
            scalar=AsyncMock(side_effect=[violation, 8]),
            add=MagicMock(),
            commit=AsyncMock(),
            refresh=AsyncMock(
                side_effect=lambda feedback: setattr(feedback, 'id', 21),
            ),
            rollback=AsyncMock(),
        )
        payload = ViolationFeedbackCreate(
            type='wrong_class',
            target_detection_id='det_0',
            corrected_label='worker',
            note='Detector label needs correction',
        )

        with patch.object(
            violation_review_service.user_service,
            'get_cached_effective_site_names',
            new=AsyncMock(return_value=['Site A']),
        ):
            response = (
                await violation_review_service.submit_violation_feedback(
                    4,
                    payload,
                    cast(AsyncSession, db),
                    _credentials(),
                )
            )

        self.assertEqual(response.id, 21)
        self.assertEqual(response.violation_id, 4)
        self.assertTrue(violation.is_flagged)
        self.assertEqual(violation.review_status, 'pending')
        db.commit.assert_awaited_once()
        db.refresh.assert_awaited_once()

    async def test_review_and_next_item_use_authorised_scope(self) -> None:
        """Review writes an audit event and next-item lookup returns detail."""
        violation = SimpleNamespace(
            id=4,
            site='Site A',
            is_flagged=True,
            review_status='pending',
            review_note=None,
            reviewed_by=None,
            reviewed_at=None,
            flag_reason='wrong_class',
            flagged_at=datetime(2026, 8, 23, tzinfo=timezone.utc),
            detection_time=datetime(2026, 8, 23, tzinfo=timezone.utc),
        )
        reviewer = SimpleNamespace(id=8, role='admin')
        db = SimpleNamespace(
            scalar=AsyncMock(return_value=violation),
            add=MagicMock(),
            commit=AsyncMock(),
            refresh=AsyncMock(),
            rollback=AsyncMock(),
        )

        with (
            patch.object(
                violation_review_service,
                '_load_review_scope',
                new=AsyncMock(return_value=(reviewer, ['Site A'])),
            ),
            patch.object(
                violation_review_service,
                '_detail_response',
                new=AsyncMock(return_value=_detail_item()),
            ),
            patch.object(
                violation_review_service,
                '_violation_to_detail_item',
                return_value=_detail_item(),
            ),
        ):
            next_item = (
                await violation_review_service.get_next_review_violation(
                    MagicMock(),
                    cast(AsyncSession, db),
                    _credentials(),
                )
            )
            reviewed = await violation_review_service.review_violation(
                4,
                ViolationReviewUpdate(
                    review_status='resolved',
                    review_note='Resolved after review',
                ),
                MagicMock(),
                cast(AsyncSession, db),
                _credentials(),
            )

        assert next_item is not None
        self.assertEqual(next_item.id, 4)
        self.assertEqual(reviewed.review_status, 'pending')
        self.assertEqual(violation.review_status, 'resolved')
        self.assertEqual(db.add.call_count, 1)
        db.commit.assert_awaited_once()

    async def test_audit_detail_and_single_record_load_authorised_data(
        self,
    ) -> None:
        """Audit, detail, and single-record reads share one scope policy."""
        violation = SimpleNamespace(
            id=4,
            site='Site A',
            image_path='evidence.jpg',
            is_flagged=True,
        )
        reviewer = SimpleNamespace(id=8, role='admin')
        db = SimpleNamespace(scalar=AsyncMock(side_effect=[4, violation]))
        detail = _detail_item()

        with (
            patch.object(
                violation_review_service,
                '_load_review_scope',
                new=AsyncMock(return_value=(reviewer, ['Site A'])),
            ),
            patch.object(
                violation_review_service,
                '_load_review_audit_logs',
                new=AsyncMock(return_value=[]),
            ) as load_audit,
            patch.object(
                violation_review_service,
                '_load_violation_feedbacks',
                new=AsyncMock(return_value=[]),
            ),
            patch.object(
                violation_review_service,
                '_violation_to_detail_item',
                return_value=detail,
            ),
            patch.object(
                violation_review_service,
                '_overlay_objects_from_feedback',
                return_value=[],
            ),
            patch.object(
                violation_review_service,
                'image_size_for_violation',
                new=AsyncMock(return_value=(100, 100)),
            ),
            patch.object(
                violation_review_service.user_service,
                'get_cached_effective_site_names',
                new=AsyncMock(return_value=['Site A']),
            ),
        ):
            audit = (
                await violation_review_service.get_violation_review_audit_log(
                    4,
                    cast(AsyncSession, db),
                    _credentials(),
                )
            )
            detailed = await violation_review_service._detail_response(
                cast(Violation, violation),
                MagicMock(),
                cast(AsyncSession, db),
                include_audit=True,
            )
            single = await violation_review_service.get_single_violation(
                4,
                MagicMock(),
                cast(AsyncSession, db),
                _credentials(),
            )

        self.assertEqual(audit, [])
        self.assertIs(detailed, detail)
        self.assertIs(single, detail)
        self.assertGreaterEqual(load_audit.await_count, 2)

    async def test_review_rejects_unflagged_records_and_rolls_back_errors(
        self,
    ) -> None:
        """Review rejects unflagged records and rolls back failed commits."""
        reviewer = SimpleNamespace(id=8, role='admin')
        unflagged = SimpleNamespace(id=4, site='Site A', is_flagged=False)
        failing = SimpleNamespace(
            id=5,
            site='Site A',
            is_flagged=True,
            review_status='pending',
            review_note=None,
            reviewed_by=None,
            reviewed_at=None,
            flag_reason='wrong_class',
        )
        db = SimpleNamespace(
            scalar=AsyncMock(side_effect=[unflagged, failing]),
            add=MagicMock(),
            commit=AsyncMock(side_effect=SQLAlchemyError('database offline')),
            refresh=AsyncMock(),
            rollback=AsyncMock(),
        )
        update = ViolationReviewUpdate(review_status='resolved')

        with patch.object(
            violation_review_service,
            '_load_review_scope',
            new=AsyncMock(return_value=(reviewer, ['Site A'])),
        ):
            with self.assertRaises(HTTPException) as unflagged_error:
                await violation_review_service.review_violation(
                    4,
                    update,
                    MagicMock(),
                    cast(AsyncSession, db),
                    _credentials(),
                )
            with self.assertRaises(HTTPException) as failed_commit:
                await violation_review_service.review_violation(
                    5,
                    update,
                    MagicMock(),
                    cast(AsyncSession, db),
                    _credentials(),
                )

        self.assertEqual(unflagged_error.exception.status_code, 404)
        self.assertEqual(failed_commit.exception.status_code, 500)
        db.rollback.assert_awaited_once()

    async def test_review_rejects_missing_or_unflagged_records(self) -> None:
        """Review denies records outside scope and unflagged records."""
        db = SimpleNamespace(scalar=AsyncMock(return_value=None))
        reviewer = SimpleNamespace(id=8, role='admin')

        with patch.object(
            violation_review_service,
            '_load_review_scope',
            new=AsyncMock(return_value=(reviewer, ['Site A'])),
        ):
            with self.assertRaises(HTTPException) as missing:
                await violation_review_service.review_violation(
                    4,
                    ViolationReviewUpdate(review_status='resolved'),
                    MagicMock(),
                    cast(AsyncSession, db),
                    _credentials(),
                )

        self.assertEqual(missing.exception.status_code, 403)

    async def test_review_readers_and_feedback_reject_invalid_records(
        self,
    ) -> None:
        """Read and feedback workflows fail safely for invalid record state."""
        reviewer = SimpleNamespace(id=8, role='admin')
        violation = SimpleNamespace(
            id=4,
            site='Site A',
            detections_json='[[1, 2, 3, 4, 0.8, 1, 9]]',
        )
        db = SimpleNamespace(scalar=AsyncMock())
        payload = ViolationFeedbackCreate(
            type='wrong_class',
            target_detection_id='det_0',
            corrected_label='worker',
        )

        with patch.object(
            violation_review_service,
            '_load_review_scope',
            new=AsyncMock(return_value=(reviewer, [])),
        ):
            no_pending = (
                await violation_review_service.get_next_review_violation(
                    MagicMock(),
                    cast(AsyncSession, db),
                    _credentials(),
                )
            )
        self.assertIsNone(no_pending)

        with patch.object(
            violation_review_service,
            '_load_review_scope',
            new=AsyncMock(return_value=(reviewer, ['Site A'])),
        ):
            db.scalar = AsyncMock(side_effect=['Site A', None])
            scoped_none = (
                await violation_review_service.get_next_review_violation(
                    MagicMock(),
                    cast(AsyncSession, db),
                    _credentials(),
                    site_id=1,
                    current_id=4,
                )
            )
        self.assertIsNone(scoped_none)

        with patch.object(
            violation_review_service,
            '_load_review_scope',
            new=AsyncMock(return_value=(reviewer, ['Site A'])),
        ):
            db.scalar = AsyncMock(return_value=None)
            with self.assertRaises(HTTPException) as audit_denied:
                await violation_review_service.get_violation_review_audit_log(
                    4,
                    cast(AsyncSession, db),
                    _credentials(),
                )
        self.assertEqual(audit_denied.exception.status_code, 403)

        with patch.object(
            violation_review_service.user_service,
            'get_cached_effective_site_names',
            new=AsyncMock(return_value=['Site A']),
        ):
            db.scalar = AsyncMock(return_value=None)
            with self.assertRaises(HTTPException) as single_denied:
                await violation_review_service.get_single_violation(
                    4,
                    MagicMock(),
                    cast(AsyncSession, db),
                    _credentials(),
                )
            with self.assertRaises(HTTPException) as feedback_denied:
                await violation_review_service.submit_violation_feedback(
                    4,
                    payload,
                    cast(AsyncSession, db),
                    _credentials(),
                )
        self.assertEqual(single_denied.exception.status_code, 403)
        self.assertEqual(feedback_denied.exception.status_code, 403)

        invalid_target = ViolationFeedbackCreate(
            type='wrong_class',
            target_detection_id='det_99',
            corrected_label='worker',
        )
        with patch.object(
            violation_review_service.user_service,
            'get_cached_effective_site_names',
            new=AsyncMock(return_value=['Site A']),
        ):
            db.scalar = AsyncMock(return_value=violation)
            with self.assertRaises(HTTPException) as invalid_target_error:
                await violation_review_service.submit_violation_feedback(
                    4,
                    invalid_target,
                    cast(AsyncSession, db),
                    _credentials(),
                )
        self.assertEqual(invalid_target_error.exception.status_code, 422)


if __name__ == '__main__':
    unittest.main()
