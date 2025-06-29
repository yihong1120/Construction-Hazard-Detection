from __future__ import annotations

import unittest
from datetime import datetime

from pydantic import ValidationError

from examples.db_management.schemas.stream_config import StreamConfigCreate
from examples.db_management.schemas.stream_config import StreamConfigRead
from examples.db_management.schemas.stream_config import StreamConfigUpdate


class TestStreamConfigCreate(unittest.TestCase):
    """Unit tests for StreamConfigCreate schema."""

    def test_valid(self) -> None:
        """Test creating a valid StreamConfigCreate instance with defaults."""
        data: StreamConfigCreate = StreamConfigCreate(
            site_id=1, stream_name='stream1', video_url='http://video.url',
        )
        self.assertEqual(data.site_id, 1)
        self.assertEqual(data.stream_name, 'stream1')
        self.assertEqual(data.video_url, 'http://video.url')
        self.assertEqual(data.model_key, 'yolo11x')
        self.assertTrue(data.detect_with_server)
        self.assertEqual(data.work_start_hour, 7)
        self.assertEqual(data.work_end_hour, 18)
        self.assertFalse(data.detect_no_safety_vest_or_helmet)
        self.assertFalse(data.store_in_redis)
        self.assertIsNone(data.expire_date)

    def test_optional_fields(self) -> None:
        """Test creating StreamConfigCreate with optional fields provided."""
        data: StreamConfigCreate = StreamConfigCreate(
            site_id=2,
            stream_name='stream2',
            video_url='http://video.url',
            model_key='custom_model',
            detect_with_server=False,
            work_start_hour=8,
            work_end_hour=20,
            detect_no_safety_vest_or_helmet=True,
            store_in_redis=True,
            expire_date=None,
        )
        self.assertEqual(data.model_key, 'custom_model')
        self.assertFalse(data.detect_with_server)
        self.assertEqual(data.work_start_hour, 8)
        self.assertEqual(data.work_end_hour, 20)
        self.assertTrue(data.detect_no_safety_vest_or_helmet)
        self.assertTrue(data.store_in_redis)

    def test_missing_required_fields(self) -> None:
        """Test validation error when required fields are missing."""
        with self.assertRaises(ValidationError):
            StreamConfigCreate()


class TestStreamConfigUpdate(unittest.TestCase):
    """Unit tests for StreamConfigUpdate schema."""

    def test_all_optional_fields(self) -> None:
        """
        Test creating StreamConfigUpdate instance with all fields optional.
        """
        data: StreamConfigUpdate = StreamConfigUpdate()
        self.assertIsNone(data.stream_name)
        self.assertIsNone(data.video_url)
        self.assertIsNone(data.model_key)
        self.assertIsNone(data.detect_with_server)
        self.assertIsNone(data.work_start_hour)

    def test_partial_fields_provided(self) -> None:
        """Test creating StreamConfigUpdate with partial fields provided."""
        data: StreamConfigUpdate = StreamConfigUpdate(
            stream_name='new_stream', work_end_hour=20,
        )
        self.assertEqual(data.stream_name, 'new_stream')
        self.assertEqual(data.work_end_hour, 20)
        self.assertIsNone(data.model_key)


class TestStreamConfigRead(unittest.TestCase):
    """Unit tests for StreamConfigRead schema."""

    def test_valid_instance(self) -> None:
        """Test creating a valid StreamConfigRead instance."""
        now: datetime = datetime.now()
        data: StreamConfigRead = StreamConfigRead(
            id=1,
            stream_name='stream1',
            video_url='http://video.url',
            model_key='model1',
            detect_with_server=True,
            store_in_redis=False,
            work_start_hour=7,
            work_end_hour=18,
            detect_no_safety_vest_or_helmet=True,
            detect_near_machinery_or_vehicle=False,
            detect_in_restricted_area=False,
            detect_in_utility_pole_restricted_area=False,
            detect_machinery_close_to_pole=False,
            expire_date=None,
            total_stream_in_group=2,
            max_allowed_streams=10,
            updated_at=now,
        )
        self.assertEqual(data.id, 1)
        self.assertEqual(data.stream_name, 'stream1')
        self.assertEqual(data.model_key, 'model1')
        self.assertTrue(data.detect_with_server)
        self.assertFalse(data.store_in_redis)
        self.assertEqual(data.total_stream_in_group, 2)
        self.assertEqual(data.max_allowed_streams, 10)
        self.assertEqual(data.updated_at, now)

    def test_missing_required_fields(self) -> None:
        """Test validation error when required fields are missing."""
        with self.assertRaises(ValidationError):
            StreamConfigRead()


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.db_management.schemas.stream_config \
       --cov-report=term-missing \
       tests/examples/db_management/schemas/stream_config_test.py
'''
