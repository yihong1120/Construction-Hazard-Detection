from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import mock_open
from unittest.mock import patch

import numpy as np

from examples.YOLO_data_augmentation.visualise_bounding_boxes import (
    BoundingBoxVisualiser,
)
from examples.YOLO_data_augmentation.visualise_bounding_boxes import main


class TestBoundingBoxVisualiser(unittest.TestCase):
    def setUp(self) -> None:
        self.image_path: str = (
            'tests/cv_dataset/images/'
            '-_jpeg.rf.3e98d2f5b90e0b1459e15f570a433459.jpg'
        )
        self.label_path: str = (
            'tests/cv_dataset/labels/'
            '-_jpeg.rf.3e98d2f5b90e0b1459e15f570a433459.txt'
        )
        self.class_names: list[str] = [
            'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
            'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle',
        ]
        self.visualiser: BoundingBoxVisualiser | None = None

    def tearDown(self) -> None:
        """
        Clean up resources after each test case.
        """
        if self.visualiser is not None:
            del self.visualiser
        # Reset the visualiser to None
        self.visualiser = None

    @patch(
        'examples.YOLO_data_augmentation.'
        'visualise_bounding_boxes.cv2.imread',
        return_value=None,
    )
    def test_image_loading_failure(self, mock_imread: Any) -> None:
        """
        Test if ValueError is raised when image could not be loaded.
        """
        with self.assertRaises(ValueError) as context:
            _ = BoundingBoxVisualiser(
                self.image_path, self.label_path, self.class_names,
            )

        self.assertEqual(
            str(context.exception),
            'Image could not be loaded. Please check the image path.',
        )

    @patch(
        'pathlib.Path.open',
        new_callable=mock_open,
        read_data='0 0.5 0.5 0.5 0.5\n',
    )
    @patch(
        'examples.YOLO_data_augmentation.'
        'visualise_bounding_boxes.cv2.imread',
    )
    def test_draw_bounding_boxes(
        self, mock_imread: Any, mock_file: Any,
    ) -> None:
        """
        Test drawing bounding boxes on an image.
        """
        # Mock the image as a numpy array with shape (100, 100, 3)
        mock_image: np.ndarray = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image

        with patch('pathlib.Path.open', mock_file):
            self.visualiser = BoundingBoxVisualiser(
                self.image_path, self.label_path, self.class_names,
            )

            self.visualiser.draw_bounding_boxes()

        # Assert that rectangle was called correctly
        mock_imread.assert_called_once_with(self.image_path)

        # Ensure 'r' mode is passed correctly
        mock_file.assert_called_once_with('r')

    @patch(
        'examples.YOLO_data_augmentation.'
        'visualise_bounding_boxes.cv2.imwrite',
    )
    @patch(
        'examples.YOLO_data_augmentation.'
        'visualise_bounding_boxes.cv2.imread',
    )
    def test_save_image(
        self, mock_imread: Any, mock_imwrite: Any,
    ) -> None:
        """
        Test saving the image with drawn bounding boxes.
        """
        # Mock the image as a numpy array with shape (100, 100, 3)
        mock_image: np.ndarray = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image

        self.visualiser = BoundingBoxVisualiser(
            self.image_path, self.label_path, self.class_names,
        )

        self.visualiser.draw_bounding_boxes()
        self.visualiser.save_or_display_image(
            output_path='output.jpg', save=True,
        )

        mock_imwrite.assert_called_once_with('output.jpg', mock_image)

    @patch(
        'examples.YOLO_data_augmentation.'
        'visualise_bounding_boxes.plt.show',
    )
    @patch(
        'examples.YOLO_data_augmentation.'
        'visualise_bounding_boxes.plt.imshow',
    )
    @patch(
        'examples.YOLO_data_augmentation.'
        'visualise_bounding_boxes.cv2.imread',
    )
    def test_display_image(
        self, mock_imread: Any, mock_imshow: Any, mock_show: Any,
    ) -> None:
        """
        Test displaying the image with drawn bounding boxes.
        """
        # Mock the image as a numpy array with shape (100, 100, 3)
        mock_image: np.ndarray = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image

        self.visualiser = BoundingBoxVisualiser(
            self.image_path, self.label_path, self.class_names,
        )

        self.visualiser.draw_bounding_boxes()
        self.visualiser.save_or_display_image(
            output_path='output.jpg', save=False,
        )

        mock_imshow.assert_called_once()
        mock_show.assert_called_once()

    @patch(
        'sys.argv',
        [
            'visualise_bounding_boxes.py', '--image',
            'image.jpg', '--label', 'label.txt', '--save',
        ],
    )
    @patch(
        'examples.YOLO_data_augmentation.visualise_bounding_boxes.'
        'BoundingBoxVisualiser.__init__',
        return_value=None,
    )
    @patch(
        'examples.YOLO_data_augmentation.visualise_bounding_boxes.'
        'BoundingBoxVisualiser.draw_bounding_boxes',
    )
    @patch(
        'examples.YOLO_data_augmentation.visualise_bounding_boxes.'
        'BoundingBoxVisualiser.save_or_display_image',
    )
    def test_main(
        self, mock_save: Any, mock_draw: Any, mock_init: Any,
    ) -> None:
        """
        Test the main function by simulating command-line arguments.
        """
        main()

        mock_init.assert_called_once_with(
            'image.jpg', 'label.txt', [
                'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
                'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle',
            ],
        )
        mock_draw.assert_called_once()
        mock_save.assert_called_once_with('visualised_image.jpg', True)


if __name__ == '__main__':
    unittest.main()
