from __future__ import annotations

import unittest
from datetime import datetime

from pydantic import ValidationError

from examples.violation_records.schemas import SiteOut
from examples.violation_records.schemas import UploadViolationResponse
from examples.violation_records.schemas import ViolationItem
from examples.violation_records.schemas import ViolationList


class TestSchemas(unittest.TestCase):
    """
    Test suite for the Pydantic models defined in schemas.py.
    """

    def test_site_out_success(self):
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

    def test_site_out_missing_field(self):
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

    def test_violation_item_success(self):
        """
        Ensure ViolationItem can be instantiated with valid data.
        """
        data = {
            'id': 123,
            'site_name': 'Example Site',
            'stream_name': 'CamX',
            'detection_time': datetime(2023, 5, 1, 12, 0, 0),
            'image_path': 'path/to/image.jpg',
            'created_at': datetime(2023, 5, 1, 13, 0, 0),
            'detection_items': 'some detections',
            'warnings': None,
            'cone_polygons': '[]',
            'pole_polygons': None,
        }
        violation = ViolationItem(**data)
        self.assertEqual(violation.id, 123)
        self.assertEqual(violation.site_name, 'Example Site')
        self.assertEqual(violation.stream_name, 'CamX')
        self.assertEqual(violation.detection_time, data['detection_time'])
        self.assertEqual(violation.image_path, 'path/to/image.jpg')
        self.assertEqual(violation.created_at, data['created_at'])
        self.assertEqual(violation.detection_items, 'some detections')
        self.assertIsNone(violation.warnings)
        self.assertEqual(violation.cone_polygons, '[]')
        self.assertIsNone(violation.pole_polygons)

    def test_violation_item_invalid_field_type(self):
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

    def test_violation_list_success(self):
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

    def test_violation_list_empty_items(self):
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

    def test_upload_violation_response_success(self):
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

    def test_upload_violation_response_missing_field(self):
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


if __name__ == '__main__':
    unittest.main()

"""
pytest --cov=examples.violation_records.schemas \
       --cov-report=term-missing \
       tests/examples/violation_records/schemas_test.py
"""
