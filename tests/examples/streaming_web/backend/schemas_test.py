from __future__ import annotations

import unittest

from pydantic import ValidationError

from examples.streaming_web.backend.schemas import FramePostResponse
from examples.streaming_web.backend.schemas import LabelListResponse


class TestLabelListResponse(unittest.TestCase):
    """
    Tests for the LabelListResponse Pydantic model.
    """

    def test_valid_labels(self) -> None:
        """
        Test creating a LabelListResponse with a valid list of labels.
        """
        labels: list[str] = ['site1', 'site2', 'cameraA']
        response = LabelListResponse(labels=labels)

        self.assertEqual(response.labels, labels)
        self.assertIsInstance(response.labels, list)
        for label in response.labels:
            self.assertIsInstance(label, str)

    def test_empty_labels(self) -> None:
        """
        Test creating a LabelListResponse with an empty list of labels.
        """
        response = LabelListResponse(labels=[])
        self.assertIsInstance(response.labels, list)
        self.assertEqual(len(response.labels), 0)

    def test_invalid_labels_type(self) -> None:
        """
        Test creating a LabelListResponse with a non-string label.
        Expecting a ValidationError with a relevant message.
        """
        invalid_data: dict[str, list[str | int]] = {
            'labels': ['valid_label', 123],
        }  # 123 is not a string

        with self.assertRaises(ValidationError) as context:
            LabelListResponse(**invalid_data)

        self.assertIn('Input should be a valid string', str(context.exception))


class TestFramePostResponse(unittest.TestCase):
    """
    Tests for the FramePostResponse Pydantic model.
    """

    def test_valid_frame_post_response(self) -> None:
        """
        Test creating a FramePostResponse with valid data.
        """
        data: dict[str, str] = {
            'status': 'ok',
            'message': 'Frame stored successfully.',
        }
        response = FramePostResponse(**data)

        self.assertEqual(response.status, data['status'])
        self.assertEqual(response.message, data['message'])
        self.assertIsInstance(response.status, str)
        self.assertIsInstance(response.message, str)

    def test_missing_required_fields(self) -> None:
        """
        Test that missing required fields raises a ValidationError.
        """
        invalid_data: dict[str, str] = {}
        with self.assertRaises(ValidationError) as context:
            FramePostResponse(**invalid_data)

        self.assertIn('Field required', str(context.exception))

    def test_invalid_status_type(self) -> None:
        """
        Test creating a FramePostResponse with an invalid type for 'status'.
        Expecting ValidationError with a relevant message.
        """
        invalid_data: dict[str, int | str] = {
            'status': 200,  # Should be a string
            'message': 'Frame was stored.',
        }
        with self.assertRaises(ValidationError) as context:
            FramePostResponse(**invalid_data)

        self.assertIn('Input should be a valid string', str(context.exception))


if __name__ == '__main__':
    unittest.main()

"""
pytest \
    --cov=examples.streaming_web.backend.schemas \
    --cov-report=term-missing \
    tests/examples/streaming_web/backend/schemas_test.py
"""
