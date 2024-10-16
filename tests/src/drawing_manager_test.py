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

    def test_draw_detections_on_frame_with_thai_language(self) -> None:
        """
        Test drawing detections on a frame with Thai language labels.
        """
        # Create a DrawingManager with Thai language
        drawer_thai = DrawingManager(language='th')

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

        # Draw detections on frame
        frame_with_detections = drawer_thai.draw_detections_on_frame(
            self.frame.copy(), polygons, datas,
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

        # Check if objects are drawn correctly
        for data in self.datas:
            x1, y1, x2, y2, _, label_id = data
            label_id = int(label_id)  # Ensure label_id is an int
            label: str = self.drawer.category_id_to_name.get(label_id, '')
            if label in self.drawer.colors:
                color: tuple[int, int, int] = self.drawer.colors[label]
                # Convert colour to BGR
                color_bgr: tuple[int, int, int] = color[::-1]

                # Ensure x1, y1, x2, y2 are integers
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Check if the object is drawn with the correct colour
                for y in range(
                    max(y1, 0),
                    min(y2, frame_with_detections.shape[0]),
                ):
                    actual_color_left: np.ndarray = (
                        frame_with_detections[y, x1]
                    )
                    actual_color_right: np.ndarray = (
                        frame_with_detections[y, x2 - 1]
                    )
                    self.assertTrue(
                        np.allclose(actual_color_left, color_bgr, atol=10),
                        (f"Expected colour {color_bgr}, "
                         f"but {actual_color_left}"),
                    )
                    self.assertTrue(
                        np.allclose(actual_color_right, color_bgr, atol=10),
                        (f"Expected colour {color_bgr}, "
                         f"but {actual_color_right}"),
                    )

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

    def test_draw_safety_cones_polygon_with_no_cones(self) -> None:
        """
        Test drawing a safety cones polygon with no cones.
        """
        frame_with_polygon = self.drawer.draw_safety_cones_polygon(
            self.frame.copy(), [],
        )

        # Check if the frame returned is a numpy array
        self.assertIsInstance(frame_with_polygon, np.ndarray)

        # Check if the frame dimensions are the same
        self.assertEqual(frame_with_polygon.shape, self.frame.shape)

        # Check if no polygon is drawn (frame should be unchanged)
        np.testing.assert_array_equal(
            frame_with_polygon, self.frame,
            'Frame should not be changed with no cones',
        )

    def test_draw_safety_cones_polygon(self) -> None:
        """
        Test drawing a safety cones polygon.
        """
        frame_with_polygon = self.drawer.draw_safety_cones_polygon(
            self.frame.copy(), self.polygons,
        )

        # Check if the frame returned is a numpy array
        self.assertIsInstance(frame_with_polygon, np.ndarray)

        # Check if the frame dimensions are the same
        self.assertEqual(frame_with_polygon.shape, self.frame.shape)

        # Check if safety cones are drawn correctly
        found_safety_cone: bool = False
        # Colour of the polygon border (pink)
        expected_color: tuple[int, int, int] = (255, 0, 255)

        for polygon in self.polygons:
            polygon_points: np.ndarray = np.array(
                polygon.exterior.coords, dtype=np.int32,
            )
            for point in polygon_points:
                x, y = point
                if (
                    0 <= y < frame_with_polygon.shape[0] and
                    0 <= x < frame_with_polygon.shape[1]
                ):
                    pixel: np.ndarray = frame_with_polygon[y, x]
                    if np.allclose(pixel, expected_color, atol=10):
                        found_safety_cone = True
                        break

        self.assertTrue(
            found_safety_cone,
            f"Colour {expected_color} not found"
            f"on the polygon border",
        )

    def test_draw_safety_cones_polygon_less_three_cones(self) -> None:
        """
        Test drawing a safety cones polygon with less than three cones.
        """
        datas: list[list[float]] = [
            [300, 50, 400, 150, 0.75, 6],    # Only one safety cone detection
        ]
        # Extract only the coordinates for creating the Polygon
        coords = [
            (
                (float(data[0]) + float(data[2])) / 2,
                (float(data[1]) + float(data[3])) / 2,
            )
            for data in datas
        ]
        frame_with_polygon = self.drawer.draw_safety_cones_polygon(
            self.frame.copy(), [Polygon(coords).convex_hull] if len(
                coords,
            ) >= 3 else [],
        )

        # Check if the frame returned is a numpy array
        self.assertIsInstance(frame_with_polygon, np.ndarray)

        # Check if the frame dimensions are the same
        self.assertEqual(frame_with_polygon.shape, self.frame.shape)

        # Check if no polygon is drawn (frame should be unchanged)
        np.testing.assert_array_equal(
            frame_with_polygon, self.frame,
            'Frame should not be changed with less than three cones',
        )

    def test_draw_safety_cones_polygon_with_many_cones(self) -> None:
        """
        Test drawing a safety cones polygon with many cones.
        """
        # Generate a large number of safety cones
        num_cones: int = 100
        datas: list[list[float]] = [
            [
                np.random.randint(0, 640), np.random.randint(0, 480),
                np.random.randint(0, 640), np.random.randint(0, 480),
                0.75, 6,
            ]
            for _ in range(num_cones)
        ]
        # Extract only the coordinates for creating the Polygon
        coords = [
            (
                (float(data[0]) + float(data[2])) / 2,
                (float(data[1]) + float(data[3])) / 2,
            )
            for data in datas
        ]
        frame_with_polygon = self.drawer.draw_safety_cones_polygon(
            self.frame.copy(), [Polygon(coords).convex_hull],
        )

        # Check if the frame returned is a numpy array
        self.assertIsInstance(frame_with_polygon, np.ndarray)

        # Check if the frame dimensions are the same
        self.assertEqual(frame_with_polygon.shape, self.frame.shape)

        # Check if the polygon is drawn correctly
        # (the frame should not be unchanged)
        self.assertFalse(
            np.array_equal(frame_with_polygon, self.frame),
            'Frame should be changed with many cones',
        )

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
