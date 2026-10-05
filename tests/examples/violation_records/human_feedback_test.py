from __future__ import annotations

from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from examples.violation_records import record_policy
from examples.violation_records import violation_review_service as service
from examples.violation_records.schemas import ViolationFeedbackCreate


class HumanFeedbackTest(IsolatedAsyncioTestCase):
    def setUp(self):
        self.record = SimpleNamespace(
            id=155176,
            site='Site',
            detections_json='[]',
            model_version='v1',
            model_classes=[
                {
                    'id': 0,
                    'code': 'person',
                    'display_name': '人員',
                    'color': '#FF9800',
                },
            ],
        )
        self.user = SimpleNamespace(id=8, tenant_id='tenant', role='user')
        self.credentials = SimpleNamespace(
            subject={'username': 'user', 'user_id': 8, 'tenant_id': 'tenant'},
        )

        async def refresh(row):
            row.id = 123

        self.db = SimpleNamespace(
            scalar=AsyncMock(side_effect=[self.record, 8]),
            get=AsyncMock(return_value=self.user),
            add=MagicMock(),
            commit=AsyncMock(),
            refresh=AsyncMock(side_effect=refresh),
            rollback=AsyncMock(),
        )
        for name, replacement in [
            (
                'get_effective_site_names',
                AsyncMock(return_value=['Site']),
            ),
        ]:
            patcher = patch.object(service.user_service, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.access = AsyncMock()
        patcher = patch.object(service, 'authorize_violation', self.access)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_snapshot_accepts_annotation_without_client_model_fields(
        self,
    ):
        payload = ViolationFeedbackCreate.model_validate(
            {
                'type': 'false_negative',
                'corrected_label': 'person',
                'corrected_bbox': [10, 20, 100, 200],
                'color': '#000000',
                'model_id': 'client-claim',
                'model_version': 'client-claim',
            },
        )
        result = await service.submit_violation_feedback(
            155176, payload, self.db, self.credentials,
        )
        row = self.db.add.call_args.args[0]
        self.assertIsNone(row.model_version)
        self.assertEqual(self.record.model_classes[0]['color'], '#FF9800')
        self.assertNotIn('color', result.model_dump())
        self.assertNotIn('model_version', result.model_dump())
        self.assertNotIn('model_id', result.model_dump())
        self.assertEqual(result.corrected_label, 'person')
        self.assertEqual(result.corrected_bbox, [10, 20, 100, 200])
        self.db.commit.assert_awaited_once()
        self.access.assert_awaited_once_with(self.db, self.user, self.record)

    async def test_policy_rechecked_at_submission(self):
        payload = ViolationFeedbackCreate(
            type='false_negative',
            corrected_label='person',
            corrected_bbox=[0, 0, 1, 1],
        )
        # Opening a form does not grant permanent policy approval.
        self.record.model_classes = []
        with self.assertRaises(HTTPException) as raised:
            await service.submit_violation_feedback(
                155176, payload, self.db, self.credentials,
            )
        self.assertEqual(raised.exception.status_code, 422)
        self.db.add.assert_not_called()
        self.db.commit.assert_not_awaited()

    async def test_unauthorized_record_still_rejected(self):
        self.access.side_effect = HTTPException(403)
        payload = ViolationFeedbackCreate(
            type='false_negative',
            corrected_label='person',
            corrected_bbox=[0, 0, 1, 1],
        )
        with self.assertRaises(HTTPException) as raised:
            await service.submit_violation_feedback(
                155176, payload, self.db, self.credentials,
            )
        self.assertEqual(raised.exception.status_code, 403)
        self.db.commit.assert_not_awaited()

    async def test_model_changes_do_not_change_feedback_options(self):
        with patch.object(
            record_policy,
            'require_violation',
            AsyncMock(return_value=self.record),
        ):
            before = await record_policy.record_options(
                None, self.user, 155176, 'feedback',
            )
            # Current catalog changes must not be read for this record.
            from examples.shared import class_colors

            self.assertNotEqual(class_colors.COLORS['person'], '#000000')
            with patch.dict(class_colors.COLORS, person='#000000'):
                self.assertEqual(
                    record_policy.historical_classes(self.record)[0].color,
                    '#FF9800',
                )
                after = await record_policy.record_options(
                    None, self.user, 155176, 'feedback',
                )
        self.assertEqual(before, after)
        self.assertTrue(before['options'])
        self.assertEqual(
            len({v['id'] for v in before['options']}), len(before['options']),
        )

    def test_missing_or_invalid_annotation_still_rejected(self):
        for fields in (
            {},
            {'corrected_label': 'person'},
            {'corrected_label': 'person', 'corrected_bbox': [5, 5, 1, 1]},
        ):
            with self.assertRaises(ValidationError):
                ViolationFeedbackCreate(type='false_negative', **fields)
