import os
import unittest
from unittest.mock import patch, MagicMock
from src.data_augmentation import DataAugmentation
from imgaug.augmentables.bbs import BoundingBox

class TestDataAugmentation(unittest.TestCase):
    """
    Test cases for data_augmentation.py.
    """

    def setUp(self):
        """
        Set up the test case with common test data.
        """
        self.train_path = 'tests/test_dataset'
        self.num_augmentations = 2
        self.data_augmentation = DataAugmentation(self.train_path, self.num_augmentations)

    @patch('src.data_augmentation.iaa.Sequential')
    def test_get_augmentation_sequence(self, mock_sequential):
        """
        Test the _get_augmentation_sequence method to ensure it creates an augmentation sequence.
        """
        seq = self.data_augmentation._get_augmentation_sequence()
        mock_sequential.assert_called_once()
        self.assertIsNotNone(seq, "The augmentation sequence should not be None")

    @patch('src.data_augmentation.imageio.imread')
    @patch('src.data_augmentation.imageio.imwrite')
    @patch('src.data_augmentation.DataAugmentation.read_label_file')
    @patch('src.data_augmentation.DataAugmentation.write_label_file')
    def test_augment_image(self, mock_write_label_file, mock_read_label_file, mock_imwrite, mock_imread):
        """
        Test the augment_image method to ensure it processes and augments a single image.
        """
        # Mock the image and label file reading and writing
        mock_imread.return_value = MagicMock()
        mock_read_label_file.return_value = []

        # Call the augment_image method
        image_path = 'tests/test_dataset/images/test.png'
        self.data_augmentation.augment_image(image_path, self.num_augmentations, self.data_augmentation.seq, '.png')

        # Assert that the image and label files are read and written the correct number of times
        self.assertEqual(mock_imread.call_count, 1, "The image should be read once")
        self.assertEqual(mock_imwrite.call_count, self.num_augmentations, "The image should be written num_augmentations times")
        self.assertEqual(mock_write_label_file.call_count, self.num_augmentations, "The label file should be written num_augmentations times")

    @patch('src.data_augmentation.glob.glob')
    @patch('src.data_augmentation.concurrent.futures.ThreadPoolExecutor')
    def test_augment_data(self, mock_executor, mock_glob):
        """
        Test the augment_data method to ensure it performs data augmentation on all images in the dataset.
        """
        # Mock the glob.glob to return a list of image paths
        mock_glob.return_value = ['tests/test_dataset/images/test.png']

        # Mock the ThreadPoolExecutor
        mock_executor.return_value.__enter__.return_value = MagicMock()
        mock_executor.return_value.__exit__.return_value = MagicMock()

        # Call the augment_data method
        self.data_augmentation.augment_data()

        # Assert that the ThreadPoolExecutor is used
        mock_executor.assert_called_once_with(max_workers=None)
        # Assert that the glob.glob is called to fetch image paths
        mock_glob.assert_called_once()

    def test_read_label_file(self):
        """
        Test the read_label_file method to ensure it correctly reads label files and converts annotations into BoundingBox objects.
        """
        # Create a sample label file content
        label_content = "0 0.5 0.5 1.0 1.0\n1 0.25 0.25 0.5 0.5"
        label_path = 'tests/test_dataset/labels/test.txt'
        image_shape = (100, 100, 3)  # Sample image shape

        # Write the sample label content to a file
        with open(label_path, 'w') as file:
            file.write(label_content)

        # Call the read_label_file method
        bounding_boxes = DataAugmentation.read_label_file(label_path, image_shape)

        # Define the expected BoundingBox objects
        expected_bounding_boxes = [
            BoundingBox(x1=0, y1=0, x2=100, y2=100, label=0),
            BoundingBox(x1=12.5, y1=12.5, x2=62.5, y2=62.5, label=1)
        ]

        # Clean up the sample label file
        os.remove(label_path)

        # Assert that the bounding boxes match the expected bounding boxes
        self.assertEqual(len(bounding_boxes), len(expected_bounding_boxes), "The number of bounding boxes should match")
        for bb, expected_bb in zip(bounding_boxes, expected_bounding_boxes):
            self.assertAlmostEqual(bb.x1, expected_bb.x1)
            self.assertAlmostEqual(bb.y1, expected_bb.y1)
            self.assertAlmostEqual(bb.x2, expected_bb.x2)
            self.assertAlmostEqual(bb.y2, expected_bb.y2)
            self.assertEqual(bb.label, expected_bb.label)

    def test_write_label_file(self):
        """
        Test the write_label_file method to ensure it correctly writes bounding box information back to a label file.
        """
        # Create a sample list of BoundingBox objects
        bounding_boxes = [
            BoundingBox(x1=0, y1=0, x2=100, y2=100, label=0),
            BoundingBox(x1=12.5, y1=12.5, x2=62.5, y2=62.5, label=1)
        ]
        label_path = 'tests/test_dataset/labels/test_aug.txt'
        image_width = 100
        image_height = 100

        # Call the write_label_file method
        DataAugmentation.write_label_file(bounding_boxes, label_path, image_width, image_height)

        # Read the content of the written label file
        with open(label_path, 'r') as file:
            content = file.read()

        # Define the expected label file content
        expected_content = "0 0.5 0.5 1.0 1.0\n1 0.375 0.375 0.5 0.5\n"

        # Clean up the sample label file
        os.remove(label_path)

        # Assert that the content matches the expected content
        self.assertEqual(content, expected_content, "The label file content should match the expected content")

if __name__ == '__main__':
    unittest.main()