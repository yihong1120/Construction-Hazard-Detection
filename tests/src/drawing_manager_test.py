from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from shapely.geometry import Polygon

from src.drawing_manager import DrawingManager
from src.drawing_manager import main


class TestDrawingManager(unittest.TestCase):
    """
    Unit tests for the DrawingManager class.
    """

    def setUp(self) -> None:
        """
        Set up the initial state needed for tests.
        """
        self.drawer: DrawingManager = DrawingManager()
        self.frame: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)
        self.datas: list[list[float]] = [
            [50, 50, 150, 150, 0.95, 0],    # Hardhat
            [200, 200, 300, 300, 0.85, 5],  # Person
            [400, 400, 500, 500, 0.75, 9],  # Vehicle
            [100, 100, 120, 120, 0.9, 6],   # Safety Cone
            [250, 250, 270, 270, 0.8, 6],   # Safety Cone
            [450, 450, 470, 470, 0.7, 6],   # Safety Cone
            [500, 200, 520, 220, 0.7, 6],   # Safety Cone
            [150, 400, 170, 420, 0.7, 6],   # Safety Cone
        ]
        self.polygons: list[Polygon] = [
            Polygon([
                (100, 100), (250, 250), (450, 450),
                (500, 200), (150, 400),
            ]).convex_hull,
        ]

    def tearDown(self) -> None:
        """
        Clean up after each test.
        """
        del self.drawer
        del self.frame
        del self.datas
        del self.polygons

        # Remove the output directory
        output_dir: Path = Path('detected_frames/test_output')
        if output_dir.exists() and output_dir.is_dir():
            shutil.rmtree(output_dir)

        root_dir: Path = Path('detected_frames')
        if root_dir.exists() and root_dir.is_dir():
            shutil.rmtree(root_dir)

    def test_get_font_cache(self) -> None:
        """
        Test the font caching mechanism in the get_font method.
        """
        # First call to get_font should load and cache the font
        font_first_call = self.drawer.get_font('en')

        # Second call to get_font with the same language
        # should use the cached font
        font_second_call = self.drawer.get_font('en')

        # Assert that the returned font objects are the same,
        # indicating caching
        self.assertIs(
            font_first_call, font_second_call,
            'Font should be retrieved from cache',
        )

        # Check if the font is indeed cached
        font_path = 'assets/fonts/NotoSansTC-VariableFont_wght.ttf'
        self.assertIn(
            font_path, self.drawer.font_cache,
            'Font should be cached after first retrieval',
        )

    def test_get_font_fallback_to_default(self) -> None:
        """
        Test the get_font method when loading a custom font fails,
        falling back to the default font.
        """
        # Mock ImageFont.truetype to raise an OSError
        with patch('PIL.ImageFont.truetype', side_effect=OSError):
            # Call get_font, which should fall back to the default font
            fallback_font = self.drawer.get_font('en')

            # Check that the returned font is the default font
            self.assertIs(
                fallback_font, DrawingManager.default_font,
                'Should fall back to the default font when loading fails',
            )

            # Ensure that the default font is not None
            self.assertIsNotNone(
                DrawingManager.default_font,
                'Default font should be loaded and cached',
            )

    def test_draw_detections_on_frame_with_thai_language(self) -> None:
        """
        Test drawing detections on a frame with Thai language labels.
        """
        # Example detection data
        datas = [
            [50, 50, 150, 150, 0.95, 0],  # Hardhat
            [200, 200, 300, 300, 0.85, 5],  # Person
            [400, 400, 500, 500, 0.75, 9],  # Vehicle
        ]

        # Example polygon for safety cones
        polygons = [
            Polygon(
                [(100, 100), (250, 250), (450, 450), (500, 200), (150, 400)],
            ).convex_hull,
        ]

        # Draw detections on frame in Thai language
        frame_with_detections = self.drawer.draw_detections_on_frame(
            self.frame.copy(), polygons, datas, language='th',
        )

        # Check if the frame returned is a numpy array
        self.assertIsInstance(frame_with_detections, np.ndarray)

        # Check if the frame dimensions are the same
        self.assertEqual(frame_with_detections.shape, self.frame.shape)

    def test_draw_detections_on_frame(self) -> None:
        """
        Test drawing detections on a frame.
        """
        frame_with_detections = self.drawer.draw_detections_on_frame(
            self.frame.copy(), self.polygons, self.datas,
        )

        # Check if the frame returned is a numpy array
        self.assertIsInstance(frame_with_detections, np.ndarray)

        # Check if the frame dimensions are the same
        self.assertEqual(frame_with_detections.shape, self.frame.shape)

    def test_draw_detections_on_frame_with_invalid_label(self) -> None:
        """
        Test drawing on a frame with an invalid label.
        """
        invalid_data = [
            [10, 10, 50, 50, 0.99, 999],  # Invalid label ID
        ]
        frame_with_detections = self.drawer.draw_detections_on_frame(
            self.frame.copy(), [], invalid_data,
        )

        # Check if the frame returned is a numpy array
        self.assertIsInstance(frame_with_detections, np.ndarray)

        # Check if the frame dimensions are the same
        self.assertEqual(frame_with_detections.shape, self.frame.shape)

    def test_draw_detections_on_frame_with_no_detections(self) -> None:
        """
        Test drawing on a frame with no detections.
        """
        frame_with_detections = self.drawer.draw_detections_on_frame(
            self.frame.copy(), [], [],
        )

        # Check if the frame returned is a numpy array
        self.assertIsInstance(frame_with_detections, np.ndarray)

        # Check if the frame dimensions are the same
        self.assertEqual(frame_with_detections.shape, self.frame.shape)

        # Check if no objects are drawn (frame should be unchanged)
        np.testing.assert_array_equal(
            frame_with_detections, self.frame,
            'Frame should not be changed with no detections',
        )

    def test_draw_polygons(self) -> None:
        """
        Test drawing polygons on the frame.
        """
        frame_with_polygons = self.drawer.draw_polygons(
            self.frame.copy(), self.polygons,
        )

        # Check if the frame returned is a numpy array
        self.assertIsInstance(frame_with_polygons, np.ndarray)

        # Check if the frame dimensions are the same
        self.assertEqual(frame_with_polygons.shape, self.frame.shape)

    def test_save_frame(self) -> None:
        """
        Test saving a frame to disk.
        """
        frame_bytes: bytearray = bytearray(
            np.zeros((480, 640, 3), dtype=np.uint8).tobytes(),
        )
        output_filename: str = 'test_frame'

        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            self.drawer.save_frame(frame_bytes, output_filename)

            # Construct the expected file path
            expected_file: Path = Path('detected_frames/test_frame.png')

            # Assert calls to open and write
            mock_file.assert_called_once_with(expected_file, 'wb')
            mock_file().write.assert_called_once_with(frame_bytes)

    def test_main(self) -> None:
        """
        Test the main function to ensure the complete process is covered.
        """
        with patch('pathlib.Path.mkdir') as mock_mkdir, \
                patch('builtins.open', unittest.mock.mock_open()) as mock_file:

            main()

            # Assert the directory was created
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

            # Construct the expected file path
            expected_file: Path = Path('detected_frames/frame_001.png')

            # Assert calls to open and write
            mock_file.assert_called_once_with(expected_file, 'wb')
            mock_file().write.assert_called_once()


if __name__ == '__main__':
    unittest.main()
