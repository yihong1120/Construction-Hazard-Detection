from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from examples.YOLO_data_augmentation.data_augmentation_albumentations import DataAugmentation
from examples.YOLO_data_augmentation.data_augmentation_albumentations import main


class TestDataAugmentation(unittest.TestCase):
    """
    Unit tests for the DataAugmentation class.
    """

    def setUp(self) -> None:
        """
        Set up the test environment.
        """
        self.train_path = 'tests/cv_dataset'
        self.num_augmentations = 2
        self.augmenter = DataAugmentation(
            self.train_path, self.num_augmentations,
        )

    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'cv2.imread',
    )
    @patch('builtins.print')
    def test_augment_image_exception(
        self, mock_print: MagicMock, mock_imread: MagicMock,
    ) -> None:
        """
        Test augment_image when an exception occurs during image reading.

        Args:
            mock_print (MagicMock): Mocked print function.
            mock_imread (MagicMock): Mocked cv2.imread function.
        """
        # Mock image reading exception
        mock_imread.side_effect = Exception('Mocked exception')

        # Test augment_image
        self.augmenter.augment_image(
            Path('tests/cv_dataset/images/mock_image.jpg'),
        )

        # Check if the print method was called with the correct output
        mock_print.assert_any_call(
            'Error processing image: '
            'tests/cv_dataset/images/mock_image.jpg: Mocked exception',
        )

    @patch('builtins.print')
    def test_augment_image_none(self, mock_print: MagicMock) -> None:
        """
        Test augment_image method when the image is None.

        Args:
            mock_print (MagicMock): Mocked print function.
        """
        # Test when image is None
        self.augmenter.augment_image(None)
        mock_print.assert_any_call('Error processing image: None')

    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'cv2.imread',
    )
    @patch(
        'examples.YOLO_data_augmentation.'
        'data_augmentation_albumentations.cv2.cvtColor',
    )
    def test_augment_image(
        self, mock_cvtColor: MagicMock, mock_imread: MagicMock,
    ) -> None:
        """
        Test augment_image method with valid image data.

        Args:
            mock_cvtColor (MagicMock): Mocked cv2.cvtColor function.
            mock_imread (MagicMock): Mocked cv2.imread function.
        """
        # Mock image and label data
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_bboxes = [[0.5, 0.5, 0.2, 0.2]]
        mock_class_labels = [1]

        mock_imread.return_value = mock_image
        mock_cvtColor.return_value = mock_image

        with patch.object(
            self.augmenter,
            'read_label_file',
            return_value=(mock_class_labels, mock_bboxes),
        ):
            with patch.object(
                self.augmenter, 'write_label_file',
            ) as mock_write_label_file:
                with patch(
                    'examples.YOLO_data_augmentation.'
                    'data_augmentation_albumentations.cv2.imwrite',
                ) as mock_imwrite:
                    self.augmenter.augment_image(Path('image.jpg'))
                    self.assertTrue(mock_imread.called)
                    self.assertTrue(mock_cvtColor.called)
                    self.assertTrue(mock_imwrite.called)
                    self.assertTrue(mock_write_label_file.called)

    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'cv2.imread',
    )
    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'cv2.cvtColor',
    )
    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'cv2.imwrite',
    )
    def test_augment_image_with_alpha_channel(
        self,
        mock_imwrite: MagicMock,
        mock_cvtColor: MagicMock,
        mock_imread: MagicMock,
    ) -> None:
        """
        Test augment_image method with an image that has an alpha channel.

        Args:
            mock_imwrite (MagicMock): Mocked cv2.imwrite function.
            mock_cvtColor (MagicMock): Mocked cv2.cvtColor function.
            mock_imread (MagicMock): Mocked cv2.imread function.
        """
        # Mock image with alpha channel
        mock_image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        mock_imread.return_value = mock_image

        mock_bboxes = [[0.5, 0.5, 0.2, 0.2]]
        mock_class_labels = [1]

        with patch.object(
            self.augmenter,
            'read_label_file',
            return_value=(mock_class_labels, mock_bboxes),
        ):
            # Simulate cv2.cvtColor behaviour to remove alpha channel
            mock_cvtColor.side_effect = lambda img, code: img[
                :,
                :, :3,
            ] if img.shape[2] == 4 else img

            self.augmenter.augment_image(Path('image_with_alpha.jpg'))
            self.assertTrue(mock_imread.called)
            self.assertTrue(mock_cvtColor.called)

            # Ensure the image has 3 channels after removing the alpha channel
            processed_image = mock_cvtColor(mock_image, None)
            self.assertEqual(processed_image.shape[2], 3)

            self.assertTrue(mock_imwrite.called)

    def test_resize_small_image(self) -> None:
        """
        Test resize_image_and_bboxes method.
        """
        mock_image = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        mock_bboxes = [[0.5, 0.5, 0.2, 0.2]]
        class_labels = [1]
        image_path = Path('small_image.jpg')

        resized_image, resized_bboxes = self.augmenter.resize_image_and_bboxes(
            mock_image, mock_bboxes, class_labels, image_path,
        )
        self.assertEqual(resized_image.shape[0], 64)
        self.assertEqual(resized_image.shape[1], 64)

    def test_resize_large_image(self) -> None:
        """
        Test resize_image_and_bboxes method with a large image.
        """
        mock_image = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        mock_bboxes = [[0.5, 0.5, 0.2, 0.2]]
        mock_class_labels = [1]
        image_path = Path('large_image.jpg')

        with patch('builtins.print') as mock_print:
            resized_image, resized_bboxes = (
                self.augmenter.resize_image_and_bboxes(
                    mock_image,
                    mock_bboxes,
                    mock_class_labels,
                    image_path,
                )
            )
            self.assertEqual(resized_image.shape, (1920, 1920, 3))
            mock_print.assert_called_with(
                f"Resize {image_path} due to large size: {mock_image.shape}",
            )

    def test_random_crop_with_random_size(self) -> None:
        """
        Test random_crop_with_random_size method.
        """
        image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        cropped_image = self.augmenter.random_crop_with_random_size(image)
        self.assertTrue(400 <= cropped_image.shape[0] <= 800)
        self.assertTrue(400 <= cropped_image.shape[1] <= 800)

    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'ProcessPoolExecutor',
    )
    def test_augment_data(self, mock_executor: MagicMock) -> None:
        """
        Test augment_data method.

        Args:
            mock_executor (MagicMock): Mocked ProcessPoolExecutor.
        """
        mock_executor.return_value.__enter__.return_value.map = MagicMock()
        self.augmenter.augment_data(batch_size=2)
        self.assertTrue(mock_executor.called)

    def test_read_label_file(self) -> None:
        """
        Test read_label_file method.
        """
        label_content = '0 0.5 0.5 0.2 0.2\n'
        label_path = Path('label.txt')
        with patch(
            'builtins.open',
            unittest.mock.mock_open(read_data=label_content),
        ):
            class_labels, bboxes = self.augmenter.read_label_file(label_path)
            self.assertEqual(class_labels, [0])
            self.assertEqual(bboxes, [[0.5, 0.5, 0.2, 0.2]])

    def test_write_label_file(self) -> None:
        """
        Test write_label_file method.
        """
        bboxes_aug = [(0.5, 0.5, 0.2, 0.2)]
        class_labels_aug = [0]
        label_path = Path('label_aug.txt')
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            self.augmenter.write_label_file(
                bboxes_aug, class_labels_aug, label_path,
            )
            mock_file().write.assert_called_with('0 0.5 0.5 0.2 0.2\n')

    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'random.shuffle',
    )
    def test_shuffle_data(self, mock_shuffle: MagicMock) -> None:
        """
        Test shuffle_data method.

        Args:
            mock_shuffle (MagicMock): Mocked random.shuffle function.
        """
        image_dir = Path(self.train_path) / 'images'
        label_dir = Path(self.train_path) / 'labels'
        image_paths = [image_dir / f'image_{i}.jpg' for i in range(5)]
        label_paths = [label_dir / f'label_{i}.txt' for i in range(5)]

        with patch.object(
            Path, 'glob',
            side_effect=[image_paths, label_paths],
        ):
            with patch.object(Path, 'rename') as mock_rename:
                self.augmenter.shuffle_data()
                self.assertTrue(mock_shuffle.called)
                self.assertTrue(mock_rename.called)

    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'DataAugmentation',
    )
    @patch('argparse.ArgumentParser.parse_args')
    def test_main(
        self, mock_parse_args: MagicMock, MockDataAugmentation: MagicMock,
    ) -> None:
        """
        Test main function.

        Args:
            mock_parse_args (MagicMock):
                Mocked argparse.ArgumentParser.parse_args function.
            MockDataAugmentation (MagicMock): Mocked DataAugmentation class.
        """
        # Mock command line arguments
        mock_parse_args.return_value = argparse.Namespace(
            train_path='./dataset_aug/train',
            num_augmentations=10,
            batch_size=5,
        )

        # Mock DataAugmentation class
        mock_augmenter = MockDataAugmentation.return_value
        mock_augmenter.augment_data = MagicMock()
        mock_augmenter.shuffle_data = MagicMock()

        # Execute main function
        main()

        # Verify DataAugmentation class was correctly initialised
        MockDataAugmentation.assert_called_once_with('./dataset_aug/train', 10)

        # Verify augment_data and shuffle_data methods were called
        mock_augmenter.augment_data.assert_called_once_with(batch_size=5)
        mock_augmenter.shuffle_data.assert_called_once()

    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'DataAugmentation',
    )
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_exception(
        self, mock_parse_args: MagicMock, MockDataAugmentation: MagicMock,
    ) -> None:
        """
        Test main function when an exception occurs.

        Args:
            mock_parse_args (MagicMock):
                Mocked argparse.ArgumentParser.parse_args function.
            MockDataAugmentation (MagicMock): Mocked DataAugmentation class.
        """
        # Mock command line arguments
        mock_parse_args.return_value = argparse.Namespace(
            train_path='./dataset_aug/train',
            num_augmentations=10,
            batch_size=5,
        )

        # Mock DataAugmentation class to raise an exception
        mock_augmenter = MockDataAugmentation.return_value
        mock_augmenter.augment_data.side_effect = Exception('Test exception')

        with patch('builtins.print') as mock_print:
            # Execute main function
            main()

            # Verify print was called with the correct error message
            mock_print.assert_called_with('Error: Test exception')


if __name__ == '__main__':
    unittest.main()
