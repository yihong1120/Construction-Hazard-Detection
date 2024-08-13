import unittest
from unittest.mock import patch, mock_open
from pathlib import Path
import numpy as np
from examples.YOLOv8_data_augmentation.visualise_bounding_boxes import BoundingBoxVisualiser

class TestBoundingBoxVisualiser(unittest.TestCase):
    def setUp(self):
        self.image_path = 'tests/cv_dataset/images/-_jpeg.rf.3e98d2f5b90e0b1459e15f570a433459.jpg'
        self.label_path = 'tests/cv_dataset/labels/-_jpeg.rf.3e98d2f5b90e0b1459e15f570a433459.txt'
        self.class_names = [
            'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
            'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
        ]

    @patch('examples.YOLOv8_data_augmentation.visualise_bounding_boxes.cv2.imread')
    @patch('builtins.open', new_callable=mock_open, read_data='0 0.5 0.5 0.5 0.5\n')
    def test_draw_bounding_boxes(self, mock_file, mock_imread):
        """
        Test drawing bounding boxes on an image.
        """
        # Mock the image as a numpy array with shape (100, 100, 3)
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image

        self.visualiser = BoundingBoxVisualiser(
            self.image_path, self.label_path, self.class_names
        )

        self.visualiser.draw_bounding_boxes()

        # Assert that rectangle was called correctly
        mock_imread.assert_called_once_with(self.image_path)
        # 使用 Path 对象来进行断言
        mock_file.assert_called_once_with(Path(self.label_path), 'r')

    @patch('examples.YOLOv8_data_augmentation.visualise_bounding_boxes.cv2.imwrite')
    @patch('examples.YOLOv8_data_augmentation.visualise_bounding_boxes.cv2.imread')
    def test_save_image(self, mock_imread, mock_imwrite):
        """
        Test saving the image with drawn bounding boxes.
        """
        # Mock the image as a numpy array with shape (100, 100, 3)
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image

        self.visualiser = BoundingBoxVisualiser(
            self.image_path, self.label_path, self.class_names
        )

        self.visualiser.draw_bounding_boxes()
        self.visualiser.save_or_display_image(output_path='output.jpg', save=True)

        mock_imwrite.assert_called_once_with('output.jpg', mock_image)

    @patch('examples.YOLOv8_data_augmentation.visualise_bounding_boxes.plt.show')
    @patch('examples.YOLOv8_data_augmentation.visualise_bounding_boxes.plt.imshow')
    @patch('examples.YOLOv8_data_augmentation.visualise_bounding_boxes.cv2.imread')
    def test_display_image(self, mock_imread, mock_imshow, mock_show):
        """
        Test displaying the image with drawn bounding boxes.
        """
        # Mock the image as a numpy array with shape (100, 100, 3)
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image

        self.visualiser = BoundingBoxVisualiser(
            self.image_path, self.label_path, self.class_names
        )

        self.visualiser.draw_bounding_boxes()
        self.visualiser.save_or_display_image(output_path='output.jpg', save=False)

        mock_imshow.assert_called_once()
        mock_show.assert_called_once()


if __name__ == '__main__':
    unittest.main()
