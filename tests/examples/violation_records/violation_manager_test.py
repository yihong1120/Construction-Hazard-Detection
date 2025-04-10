from __future__ import annotations

import unittest
import uuid
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.auth.models import Violation
from examples.violation_records.violation_manager import ViolationManager


class TestViolationManager(unittest.IsolatedAsyncioTestCase):
    """
    Test cases for the ViolationManager class.
    """

    async def asyncSetUp(self) -> None:
        """
        Set up for each test. This method is called before each test_* method
        in an asynchronous test case.
        """
        # Create an instance of ViolationManager.
        self.manager = ViolationManager(base_dir='static')

        # Prepare a timestamp for testing.
        self.mock_detection_time = datetime(2025, 4, 9, 10, 30, 0)

        # Example bytes for an image.
        self.mock_image_bytes = b'fake_image_bytes'

        # Example strings for JSON fields.
        self.mock_warnings_json = '{"warning": true}'
        self.mock_detections_json = '{"detections": []}'
        self.mock_cone_polygon_json = '{"cones": []}'
        self.mock_pole_polygon_json = '{"poles": []}'

        # Mock SQLAlchemy session methods.
        self.mock_db = AsyncMock()
        self.mock_db.add = MagicMock()
        self.mock_db.commit = AsyncMock()
        self.mock_db.refresh = AsyncMock()

    @patch('examples.violation_records.violation_manager.uuid.uuid4')
    @patch('examples.violation_records.violation_manager.Path.mkdir')
    @patch('examples.violation_records.violation_manager.aiofiles.open')
    async def test_save_violation_success(
        self,
        mock_aiofiles_open: MagicMock,
        mock_mkdir: MagicMock,
        mock_uuid: MagicMock,
    ) -> None:
        """
        Test the save_violation method when all operations succeed.

        Args:
            mock_aiofiles_open (MagicMock): Mock for aiofiles.open.
            mock_mkdir (MagicMock): Mock for Path.mkdir.
            mock_uuid (MagicMock): Mock for uuid.uuid4.
        """
        # Arrange
        mock_uuid.return_value = uuid.UUID('12345678123456781234567812345678')

        # We want aiofiles.open(...) to return an async context manager.
        mock_file_handle = AsyncMock()
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_file_handle
        mock_context_manager.__aexit__.return_value = False
        mock_aiofiles_open.return_value = mock_context_manager

        # Create a fake Violation object that will be returned by db.refresh
        fake_violation = Violation(id=999)
        self.mock_db.refresh.side_effect = lambda obj: setattr(
            obj, 'id', fake_violation.id,
        )

        # Act
        new_violation_id = await self.manager.save_violation(
            db=self.mock_db,
            site='Test Site',
            stream_name='Camera1',
            detection_time=self.mock_detection_time,
            image_bytes=self.mock_image_bytes,
            warnings_json=self.mock_warnings_json,
            detections_json=self.mock_detections_json,
            cone_polygon_json=self.mock_cone_polygon_json,
            pole_polygon_json=self.mock_pole_polygon_json,
        )

        # Assert
        # 1) Directory creation
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # 2) File opened with correct path & mode
        pos_args, kw_args = mock_aiofiles_open.call_args
        opened_path = pos_args[0]
        opened_mode = kw_args['mode']

        self.assertIn('static/2025-04-09', str(opened_path))
        # *** Here's the fix: checking for the hyphenated version ***
        self.assertIn(
            '12345678-1234-5678-1234-567812345678.png',
            str(opened_path),
        )
        self.assertEqual(opened_mode, 'wb')

        # 3) File write with correct bytes
        mock_file_handle.write.assert_awaited_once_with(self.mock_image_bytes)

        # 4) Database calls
        self.mock_db.add.assert_called_once()
        self.mock_db.commit.assert_awaited_once()
        self.mock_db.refresh.assert_awaited_once()

        # 5) Returned ID matches
        self.assertEqual(new_violation_id, 999)

    @patch('examples.violation_records.violation_manager.aiofiles.open')
    async def test_save_violation_failure(
        self,
        mock_aiofiles_open: MagicMock,
    ) -> None:
        """
        Test the save_violation method when file writing fails.

        Args:
            mock_aiofiles_open (MagicMock): Mock for aiofiles.open.
        """
        # Arrange
        mock_aiofiles_open.side_effect = Exception('File I/O error')

        # Act
        new_violation_id = await self.manager.save_violation(
            db=self.mock_db,
            site='Test Site',
            stream_name='Camera1',
            detection_time=self.mock_detection_time,
            image_bytes=self.mock_image_bytes,
        )

        # Assert
        self.assertIsNone(new_violation_id)
        self.mock_db.add.assert_not_called()
        self.mock_db.commit.assert_not_awaited()
        self.mock_db.refresh.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.violation_records.violation_manager \
    --cov-report=term-missing \
    tests/examples/violation_records/violation_manager_test.py
'''
