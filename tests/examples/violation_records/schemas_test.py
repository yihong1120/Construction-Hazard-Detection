from __future__ import annotations

import unittest
from datetime import datetime

from pydantic import ValidationError

from examples.violation_records.schemas import SiteOut
from examples.violation_records.schemas import UploadViolationResponse
from examples.violation_records.schemas import ViolationFeedbackCreate
from examples.violation_records.schemas import ViolationFeedbackItem
from examples.violation_records.schemas import ViolationFeedbackResponse
from examples.violation_records.schemas import ViolationFilterOptions
from examples.violation_records.schemas import ViolationItem
from examples.violation_records.schemas import ViolationList
from examples.violation_records.schemas import ViolationReviewUpdate


class TestSchemas(unittest.TestCase):
    """
    Test suite for the Pydantic models defined in schemas.py.
    """

    def test_site_out_success(self) -> None:
        """
        Ensure SiteOut can be instantiated with valid data.
        """
        data = {
            'id': 1,
            'name': 'Test Site',
            'created_at': datetime(2023, 1, 1, 10, 30, 0),
            'updated_at': datetime(2023, 1, 2, 11, 0, 0),
        }
        site = SiteOut(**data)
        self.assertEqual(site.id, 1)
        self.assertEqual(site.name, 'Test Site')
        self.assertEqual(site.created_at, data['created_at'])
        self.assertEqual(site.updated_at, data['updated_at'])

    def test_site_out_missing_field(self) -> None:
        """
        If a required field is missing (e.g., 'name'),
        a ValidationError is expected.
        """
        data = {
            'id': 1,
            # 'name' is missing
            'created_at': datetime(2023, 1, 1),
            'updated_at': datetime(2023, 1, 2),
        }
        with self.assertRaises(ValidationError):
            SiteOut(**data)

    def test_violation_filter_options_success(self) -> None:
        """Camera and type options retain their public API shape."""
        options = ViolationFilterOptions(
            cameras=[{'stream_id': '10', 'name': 'Cam A'}],
            violation_types=[{
                'code': 'near_vehicle',
                'label': '人員靠近車輛',
            }],
        )

        self.assertEqual(options.cameras[0].stream_id, '10')
        self.assertEqual(options.violation_types[0].code, 'near_vehicle')

    def test_violation_item_success(self) -> None:
        """
        Ensure ViolationItem can be instantiated with valid data.
        """
        data = {
            'id': 123,
            'site_name': 'Example Site',
            'stream_name': 'CamX',
            'detection_time': datetime(2023, 5, 1, 12, 0, 0),
            'detected_at': datetime(2023, 5, 1, 12, 0, 0),
            'image_path': 'path/to/image.jpg',
            'image_url': '/get_violation_image?image_path=path%2Fto%2Fimage.jpg',
            'thumbnail_url': (
                '/get_violation_thumbnail?image_path=path%2Fto%2Fimage.jpg'
            ),
            'created_at': datetime(2023, 5, 1, 13, 0, 0),
            'detection_items': 'some detections',
            'warnings': None,
            'cone_polygons': '[]',
            'pole_polygons': None,
            'detections': [
                {
                    'id': 'det_0',
                    'label': 'class-5',
                    'confidence': 0.93,
                    'bbox': [40, 40, 180, 210],
                },
            ],
            'feedbacks': [
                {
                    'id': 1,
                    'type': 'false_positive',
                    'note': '測試',
                    'target_detection_id': 'det_0',
                    'status': 'pending',
                    'submitted_by': 9,
                    'submitted_at': datetime(2026, 6, 26, 1, 0, 0),
                },
            ],
            'overlay_objects': [
                {
                    'object_id': 'det_0',
                    'label': 'class-5',
                    'confidence': 0.93,
                    'bbox': {'x': 0.1, 'y': 0.1, 'w': 0.35, 'h': 0.425},
                    'is_flagged': True,
                    'flag_reason': 'false_positive',
                    'flag_note': '測試',
                },
            ],
            'feedback_note': '測試',
        }
        violation = ViolationItem(**data)
        self.assertEqual(violation.id, 123)
        self.assertEqual(violation.site_name, 'Example Site')
        self.assertEqual(violation.stream_name, 'CamX')
        self.assertEqual(violation.detection_time, data['detection_time'])
        self.assertEqual(violation.detected_at, data['detected_at'])
        self.assertEqual(violation.image_path, 'path/to/image.jpg')
        self.assertIn('get_violation_image', violation.image_url)
        self.assertIn('get_violation_thumbnail', violation.thumbnail_url)
        self.assertEqual(violation.created_at, data['created_at'])
        self.assertEqual(violation.detection_items, 'some detections')
        self.assertIsNone(violation.warnings)
        self.assertEqual(violation.cone_polygons, '[]')
        self.assertIsNone(violation.pole_polygons)
        self.assertEqual(violation.feedback_note, '測試')
        self.assertEqual(violation.detections[0].id, 'det_0')
        assert violation.overlay_objects is not None
        self.assertTrue(violation.overlay_objects[0].is_flagged)
        self.assertEqual(violation.overlay_objects[0].bbox.x, 0.1)
        assert violation.feedbacks is not None
        self.assertEqual(violation.feedbacks[0].note, '測試')

    def test_violation_item_invalid_field_type(self) -> None:
        """
        If a field type is incorrect, e.g. 'id' is a string instead of int,
        a ValidationError should be raised.
        """
        data = {
            'id': 'bad_id_type',
            'site_name': 'Example Site',
            'stream_name': 'CamX',
            'detection_time': datetime(2023, 5, 1),
            'image_path': 'path/to/img.jpg',
            'created_at': datetime(2023, 5, 1, 14, 0, 0),
        }
        with self.assertRaises(ValidationError):
            ViolationItem(**data)

    def test_violation_list_success(self) -> None:
        """
        Ensure ViolationList can be instantiated with a 'total' count and
        a list of ViolationItem objects.
        """
        violation_data = {
            'id': 2,
            'site_name': 'Another Site',
            'stream_name': 'CamY',
            'detection_time': datetime(2023, 5, 2, 9, 0, 0),
            'image_path': 'path/to/another.jpg',
            'created_at': datetime(2023, 5, 2, 9, 15, 0),
        }
        violations_list_data = {
            'total': 1,
            'items': [
                violation_data,
            ],
        }
        result = ViolationList(**violations_list_data)
        self.assertEqual(result.total, 1)
        self.assertEqual(len(result.items), 1)
        first_item = result.items[0]
        self.assertEqual(first_item.id, 2)
        self.assertEqual(first_item.stream_name, 'CamY')

    def test_violation_list_empty_items(self) -> None:
        """
        If 'items' is an empty list, the schema should still work.
        """
        data = {
            'total': 0,
            'items': [],
        }
        result = ViolationList(**data)
        self.assertEqual(result.total, 0)
        self.assertEqual(result.items, [])

    def test_upload_violation_response_success(self) -> None:
        """
        Ensure UploadViolationResponse can be instantiated with valid data.
        """
        data = {
            'message': 'Violation uploaded successfully.',
            'violation_id': 999,
        }
        response = UploadViolationResponse(**data)
        self.assertEqual(response.message, data['message'])
        self.assertEqual(response.violation_id, 999)

    def test_upload_violation_response_missing_field(self) -> None:
        """
        If a required field is missing (e.g., 'violation_id'),
        a ValidationError is expected.
        """
        data = {
            'message': 'OK',
            # violation_id missing
        }
        with self.assertRaises(ValidationError):
            UploadViolationResponse(**data)

    def test_violation_feedback_create_false_positive_success(self) -> None:
        """False-positive feedback can target a persisted detection box."""
        payload = ViolationFeedbackCreate(
            type='false_positive',
            target_detection_id='det_0',
            original_label='class-5',
            original_bbox=[40, 40, 180, 210],
            confidence=0.93,
        )

        self.assertEqual(payload.type, 'false_positive')
        self.assertEqual(payload.original_bbox, [40.0, 40.0, 180.0, 210.0])
        self.assertEqual(payload.confidence, 0.93)

    def test_violation_feedback_create_false_positive_note_only(self) -> None:
        """False-positive feedback may be record-level with just a note."""
        payload = ViolationFeedbackCreate(
            type='false_positive',
            note='測試',
        )

        self.assertEqual(payload.type, 'false_positive')
        self.assertEqual(payload.note, '測試')

    def test_violation_feedback_create_invalid_bbox(self) -> None:
        """BBox validation rejects missing co-ordinates."""
        with self.assertRaises(ValidationError):
            ViolationFeedbackCreate(
                type='false_positive',
                original_bbox=[40, 40, 180],
            )

    def test_violation_feedback_create_false_negative_requires_annotation(
        self,
    ) -> None:
        """Missed detections need the corrected label and bbox."""
        with self.assertRaises(ValidationError):
            ViolationFeedbackCreate(type='false_negative')

    def test_violation_feedback_response_success(self) -> None:
        """Stored feedback response exposes pending review state."""
        response = ViolationFeedbackResponse(
            id=1,
            violation_id=77,
            type='false_positive',
            target_detection_id='det_0',
            status='pending',
            created_at=datetime(2026, 6, 24, 10, 0, 0),
        )

        self.assertEqual(response.id, 1)
        self.assertEqual(response.violation_id, 77)
        self.assertEqual(response.status, 'pending')

    def test_violation_feedback_item_success(self) -> None:
        """Detail feedback items expose note and target detection metadata."""
        feedback = ViolationFeedbackItem(
            id=1,
            type='false_positive',
            note='測試',
            target_detection_id='det_0',
            status='pending',
            submitted_by=9,
            submitted_at=datetime(2026, 6, 26, 1, 0, 0),
        )

        self.assertEqual(feedback.note, '測試')
        self.assertEqual(feedback.target_detection_id, 'det_0')

    def test_violation_review_update_success(self) -> None:
        """Review updates accept the supported review states."""
        payload = ViolationReviewUpdate(
            review_status='resolved',
            review_note='Confirmed violation',
        )

        self.assertEqual(payload.review_status, 'resolved')
        self.assertEqual(payload.review_note, 'Confirmed violation')


if __name__ == '__main__':
    unittest.main()

"""
pytest --cov=examples.violation_records.schemas \
       --cov-report=term-missing \
       tests/examples/violation_records/schemas_test.py
"""
