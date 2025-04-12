from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import MagicMock

from fastapi import UploadFile
from pydantic import ValidationError

from examples.YOLO_server_api.backend.schemas import DetectionRequest
from examples.YOLO_server_api.backend.schemas import ModelFileUpdate
from examples.YOLO_server_api.backend.schemas import UpdateModelRequest


class TestDetectionRequest(unittest.TestCase):
    """
    Tests for the DetectionRequest model in schemas.py
    """

    def test_creation_direct(self):
        """
        Tests creating a DetectionRequest object directly
        with a mock UploadFile and a model name.
        """
        mock_file = MagicMock(spec=UploadFile)
        data = {'model': 'yolo11n', 'image': mock_file}
        detection_request = DetectionRequest(**data)

        self.assertEqual(detection_request.model, 'yolo11n')
        self.assertIs(detection_request.image, mock_file)

    def test_creation_missing_fields(self):
        """
        Tests ValidationError is raised if required fields are missing.
        """
        # 'image' and 'model' are both required
        with self.assertRaises(ValidationError):
            DetectionRequest()

    def test_as_form_method_manual_args(self):
        """
        Tests the as_form classmethod by manually passing
        the arguments that would normally come from FastAPI.
        """
        mock_file = MagicMock(spec=UploadFile)
        detection_request = DetectionRequest.as_form(
            model='mock_model',
            image=mock_file,
        )

        self.assertEqual(detection_request.model, 'mock_model')
        self.assertIs(detection_request.image, mock_file)


class TestModelFileUpdate(unittest.TestCase):
    """
    Tests for the ModelFileUpdate model in schemas.py
    """

    def test_creation_direct(self):
        """
        Tests creating a ModelFileUpdate object directly
        with a mock UploadFile and a model name.
        """
        mock_file = MagicMock(spec=UploadFile)
        model_file_update = ModelFileUpdate(model='yolov5', file=mock_file)

        self.assertEqual(model_file_update.model, 'yolov5')
        self.assertIs(model_file_update.file, mock_file)

    def test_creation_missing_fields(self):
        """
        Tests ValidationError is raised if required fields are missing.
        """
        with self.assertRaises(ValidationError):
            ModelFileUpdate()

    def test_as_form_method_manual_args(self):
        """
        Tests the as_form classmethod by manually passing
        the arguments that would normally come from FastAPI.
        """
        mock_file = MagicMock(spec=UploadFile)
        model_file_update = ModelFileUpdate.as_form(
            model='mock_model',
            file=mock_file,
        )

        self.assertEqual(model_file_update.model, 'mock_model')
        self.assertIs(model_file_update.file, mock_file)


class TestUpdateModelRequest(unittest.TestCase):
    """
    Tests for the UpdateModelRequest model in schemas.py
    """

    def test_valid_creation(self):
        """
        Tests creating an UpdateModelRequest with valid data.
        """
        data = {
            'model': 'yolov5',
            'last_update_time': '2023-10-01T12:30:00',
        }
        req = UpdateModelRequest(**data)
        self.assertEqual(req.model, 'yolov5')
        self.assertEqual(req.last_update_time, '2023-10-01T12:30:00')

    def test_missing_fields(self):
        """
        Tests that missing required fields raise ValidationError.
        """
        with self.assertRaises(ValidationError):
            UpdateModelRequest()

    def test_last_update_as_datetime_valid(self):
        """
        Tests the last_update_as_datetime method with a valid ISO string.
        """
        data = {
            'model': 'yolov5',
            'last_update_time': '2023-10-01T12:30:00',
        }
        req = UpdateModelRequest(**data)
        dt = req.last_update_as_datetime()
        self.assertIsNotNone(dt)
        self.assertIsInstance(dt, datetime)
        self.assertEqual(dt.year, 2023)
        self.assertEqual(dt.month, 10)
        self.assertEqual(dt.day, 1)
        self.assertEqual(dt.hour, 12)
        self.assertEqual(dt.minute, 30)

    def test_last_update_as_datetime_invalid(self):
        """
        Tests last_update_as_datetime with an invalid ISO date string.
        """
        data = {
            'model': 'yolov5',
            'last_update_time': 'invalid-datetime',
        }
        req = UpdateModelRequest(**data)
        dt = req.last_update_as_datetime()
        self.assertIsNone(dt)


if __name__ == '__main__':
    unittest.main()

"""
pytest \
    --cov=examples.YOLO_server_api.backend.schemas \
    --cov-report=term-missing \
    tests/examples/YOLO_server_api/backend/schemas_test.py
"""
