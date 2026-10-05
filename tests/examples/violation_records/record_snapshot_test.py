from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase

from fastapi import HTTPException

from examples.streaming_web.overlay_labels import _color_for_class
from examples.violation_records.feedback_policy import validate_feedback_label
from examples.violation_records.record_policy import historical_classes
from examples.violation_records.schemas import FeedbackDetectionItem
from examples.violation_records.schemas import ViolationFeedbackCreate
from examples.violation_records.violation_services import (
    _overlay_objects_from_feedback,
)
from tests.model_fixtures import CLASSES


class RecordSnapshotTest(TestCase):
    def test_snapshot_name_color_and_feedback_use_record_metadata(self):
        record = SimpleNamespace(
            model_classes=CLASSES, model_id='model', model_version='v1',
        )
        metadata = [c.model_dump() for c in historical_classes(record)]
        self.assertEqual(len(metadata), len(CLASSES))
        for number in range(len(CLASSES)):
            self.assertEqual(
                metadata[number]['color'],
                '#%02X%02X%02X' % _color_for_class(metadata[number]['code']),
            )
        items = _overlay_objects_from_feedback(
            [
                FeedbackDetectionItem(
                    id='det_0', label='person', bbox=[0, 0, 0.5, 0.5],
                ),
            ],
            [],
            None,
            metadata,
        )
        self.assertEqual(
            (items[0].display_name, items[0].color), ('人員', '#FF9800'),
        )
        validate_feedback_label(
            ViolationFeedbackCreate(
                type='false_negative',
                corrected_label='person',
                corrected_bbox=[0, 0, 10, 10],
            ),
            record,
        )
        with self.assertRaises(HTTPException):
            validate_feedback_label(
                ViolationFeedbackCreate(
                    type='false_negative',
                    corrected_label='forged',
                    corrected_bbox=[0, 0, 10, 10],
                ),
                record,
            )
        self.assertEqual(record.model_classes, CLASSES)

    def test_snapshot_precedence_and_versioned_missing_snapshot(self):
        custom = {
            'id': 5,
            'code': 'forklift',
            'display_name': '堆高機',
            'color': '#123456',
        }
        record = SimpleNamespace(model_classes=[custom])
        self.assertEqual(historical_classes(record)[0].model_dump(), custom)
        record.model_classes = []
        self.assertEqual(historical_classes(record), [])
        record.model_classes = None
        record.model_version = 'different'
        self.assertEqual(historical_classes(record), [])


def test_missing_snapshot_never_uses_a_fixed_class_table():
    for record in (
        SimpleNamespace(model_classes=None, model_id=None, model_version=None),
        SimpleNamespace(
            model_classes=None,
            model_id='model',
            model_version='v1',
        ),
    ):
        assert historical_classes(record) == []
        with TestCase().assertRaises(HTTPException):
            validate_feedback_label(
                ViolationFeedbackCreate(
                    type='false_negative', corrected_label='person',
                    corrected_bbox=[0, 0, 10, 10],
                ), record,
            )
