from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import mock_open
from unittest.mock import patch

import numpy as np

from examples.YOLOv8_data_augmentation.data_augmentation import (
    DataAugmentation,
)
from examples.YOLOv8_data_augmentation.data_augmentation import main


class TestDataAugmentation(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test environment before each test.
        """
        self.train_path: str = 'tests/cv_dataset'
        self.num_augmentations: int = 5
        self.augmenter: DataAugmentation = DataAugmentation(
            self.train_path, self.num_augmentations,
        )

    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.imageio.imread',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.imageio.imwrite',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.Path.glob',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'iaa.Sequential.__call__',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.Path.open',
        new_callable=mock_open,
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.Path.write_text',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.Path.write_bytes',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.Path.rename',
    )
    def test_augment_image(
        self,
        mock_rename: MagicMock,
        mock_write_bytes: MagicMock,
        mock_write_text: MagicMock,
        mock_open_file: MagicMock,
        mock_seq_call: MagicMock,
        mock_glob: MagicMock,
        mock_imwrite: MagicMock,
        mock_imread: MagicMock,
    ) -> None:
        """
        Test the augment_image method.
        """
        # Mock the image and label data with alpha channel
        mock_image = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        mock_imread.return_value = mock_image
        mock_augmented_image = np.random.randint(
            0, 256, (100, 100, 3), dtype=np.uint8,
        )
        mock_seq_call.return_value = (mock_augmented_image, MagicMock())

        # Mock the label file reading
        mock_glob.return_value = [
            Path('tests/cv_dataset/images/mock_image.jpg'),
        ]

        # Test augment_image
        self.augmenter.augment_image(
            Path('tests/cv_dataset/images/mock_image.jpg'),
        )

        mock_imread.assert_called_once_with(
            Path('tests/cv_dataset/images/mock_image.jpg'),
        )
        mock_imwrite.assert_called()
        mock_seq_call.assert_called()
        mock_open_file.assert_called()
        mock_write_text.assert_not_called()
        mock_write_bytes.assert_not_called()
        mock_rename.assert_not_called()

        # Check if the alpha channel was removed
        self.assertEqual(mock_augmented_image.shape[2], 3)

    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.glob',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'gc.collect',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'DataAugmentation.augment_image',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.write_text',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.write_bytes',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.rename',
    )
    def test_augment_data(
        self,
        mock_rename: MagicMock,
        mock_write_bytes: MagicMock,
        mock_write_text: MagicMock,
        mock_augment_image: MagicMock,
        mock_gc_collect: MagicMock,
        mock_glob: MagicMock,
    ) -> None:
        """
        Test the augment_data method.
        """
        mock_glob.return_value = [
            Path(f'tests/dataset/images/mock_image_{i:02d}.jpg')
            for i in range(10)
        ]

        self.augmenter.augment_data(batch_size=2)

        self.assertEqual(mock_augment_image.call_count, 10)
        self.assertEqual(mock_gc_collect.call_count, 5)
        mock_write_text.assert_not_called()
        mock_write_bytes.assert_not_called()
        mock_rename.assert_not_called()

    @patch(
        'builtins.open',
        new_callable=mock_open,
        read_data='5 0.5 0.5 0.75 0.75\n',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.exists',
        return_value=True,
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.open',
        new_callable=mock_open,
    )
    def test_read_label_file(
        self,
        mock_open_path: MagicMock,
        mock_exists: MagicMock,
        mock_file: MagicMock,
    ) -> None:
        """
        Test the read_label_file method.
        """
        mock_label_path = Path('tests/cv_dataset/labels/mock_label.txt')

        # Test reading the label file
        image_shape = (100, 100, 3)

        bbs = self.augmenter.read_label_file(mock_label_path, image_shape)
        self.assertEqual(len(bbs), 1)
        self.assertEqual(bbs[0].label, 5)
        self.assertAlmostEqual(bbs[0].x1, 12.5)
        self.assertAlmostEqual(bbs[0].y1, 12.5)
        self.assertAlmostEqual(bbs[0].x2, 87.5)
        self.assertAlmostEqual(bbs[0].y2, 87.5)

    @patch('builtins.open', new_callable=mock_open)
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.exists',
        return_value=True,
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.mkdir',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.parent',
        new_callable=MagicMock,
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.write_text',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.write_bytes',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.rename',
    )
    def test_write_label_file(
        self,
        mock_rename: MagicMock,
        mock_write_bytes: MagicMock,
        mock_write_text: MagicMock,
        mock_parent: MagicMock,
        mock_mkdir: MagicMock,
        mock_exists: MagicMock,
        mock_file: MagicMock,
    ) -> None:
        """
        Test the write_label_file method.
        """
        # Mock bounding box data
        bbs = MagicMock()
        bbs.bounding_boxes = [MagicMock(x1=10, y1=10, x2=50, y2=50, label=0)]

        # Mock the open method
        m = mock_open()

        # Revise the mock_open_path to return the mock_open object
        mock_label_path = MagicMock(spec=Path)
        mock_label_path.open = m

        # Utilise the write_label_file method
        image_shape = (100, 100, 3)
        self.augmenter.write_label_file(
            bbs, mock_label_path, image_shape[1], image_shape[0],
        )

        # Assert that the write method was called with the correct content
        m().write.assert_called_once_with('0 0.3 0.3 0.4 0.4\n')
        mock_write_text.assert_not_called()
        mock_write_bytes.assert_not_called()
        mock_rename.assert_not_called()

    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.glob',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'random.shuffle',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.rename',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.write_text',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.write_bytes',
    )
    def test_shuffle_data(
        self,
        mock_write_bytes: MagicMock,
        mock_write_text: MagicMock,
        mock_rename: MagicMock,
        mock_shuffle: MagicMock,
        mock_glob: MagicMock,
    ) -> None:
        """
        Test the shuffle_data method.
        """
        mock_glob.side_effect = [
            [
                Path(
                    f'tests/dataset/images/mock_image_{i:02d}.jpg',
                ) for i in range(5)
            ],
            [
                Path(
                    f'tests/dataset/labels/mock_label_{i:02d}.txt',
                ) for i in range(5)
            ],
        ]

        with patch(
            'examples.YOLOv8_data_augmentation.data_augmentation.'
            'uuid.uuid4',
            side_effect=[f'uuid_{i:02d}' for i in range(5)],
        ):
            self.augmenter.shuffle_data()

        mock_shuffle.assert_called_once()
        self.assertEqual(mock_rename.call_count, 10)
        mock_write_text.assert_not_called()
        mock_write_bytes.assert_not_called()

    @patch('time.sleep', return_value=None)  # Mock time.sleep to skip delay
    @patch.object(
        sys,
        'argv',
        [
            'main', '--train_path=tests/dataset',
            '--num_augmentations=2', '--batch_size=2',
        ],
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'DataAugmentation.shuffle_data',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'DataAugmentation.augment_data',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.write_text',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.write_bytes',
    )
    def test_main(
        self,
        mock_write_bytes: MagicMock,
        mock_write_text: MagicMock,
        mock_augment_data: MagicMock,
        mock_shuffle_data: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """
        Test the main function with command line arguments.
        """
        main()
        mock_augment_data.assert_called_once_with(batch_size=2)
        mock_shuffle_data.assert_called_once()
        mock_sleep.assert_called_once()
        mock_write_text.assert_not_called()
        mock_write_bytes.assert_not_called()

    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'imageio.imread',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'imageio.imwrite',
    )
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.Path.glob')
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'iaa.Sequential.__call__',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.Path.open',
        new_callable=unittest.mock.mock_open,
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.Path.'
        'write_text',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.Path.'
        'write_bytes',
    )
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.Path.rename')
    def test_augment_image_resize_small(
        self,
        mock_rename: MagicMock,
        mock_write_bytes: MagicMock,
        mock_write_text: MagicMock,
        mock_open_file: MagicMock,
        mock_seq_call: MagicMock,
        mock_glob: MagicMock,
        mock_imwrite: MagicMock,
        mock_imread: MagicMock,
    ) -> None:
        """
        Test the augment_image method for small images.
        """
        # Mock a small image
        mock_image = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image
        mock_augmented_image = np.random.randint(
            0, 256, (100, 100, 3), dtype=np.uint8,
        )
        mock_seq_call.return_value = (mock_augmented_image, MagicMock())

        # Mock the label file reading
        mock_glob.return_value = [
            Path('tests/cv_dataset/images/mock_image.jpg'),
        ]

        with patch('builtins.print') as mock_print:
            self.augmenter.augment_image(
                Path('tests/cv_dataset/images/mock_image.jpg'),
            )
            mock_print.assert_any_call(
                f"Resize tests/cv_dataset/images/mock_image.jpg "
                f"due to small size: {mock_image.shape}",
            )

        mock_imread.assert_called_once_with(
            Path('tests/cv_dataset/images/mock_image.jpg'),
        )
        mock_imwrite.assert_called()
        mock_seq_call.assert_called()
        mock_open_file.assert_called()
        mock_write_text.assert_not_called()
        mock_write_bytes.assert_not_called()
        mock_rename.assert_not_called()

    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'imageio.imread',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'imageio.imwrite',
    )
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.Path.glob')
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'iaa.Sequential.__call__',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.Path.open',
        new_callable=unittest.mock.mock_open,
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.write_text',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.write_bytes',
    )
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.Path.rename')
    def test_augment_image_resize_large(
        self,
        mock_rename: MagicMock,
        mock_write_bytes: MagicMock,
        mock_write_text: MagicMock,
        mock_open_file: MagicMock,
        mock_seq_call: MagicMock,
        mock_glob: MagicMock,
        mock_imwrite: MagicMock,
        mock_imread: MagicMock,
    ) -> None:
        """
        Test the augment_image method for large images.
        """
        # Mock a large image
        mock_image = np.random.randint(0, 256, (2000, 2000, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image
        mock_augmented_image = np.random.randint(
            0, 256, (1000, 1000, 3), dtype=np.uint8,
        )
        mock_seq_call.return_value = (mock_augmented_image, MagicMock())

        # Mock the label file reading
        mock_glob.return_value = [
            Path('tests/cv_dataset/images/mock_image.jpg'),
        ]

        with patch('builtins.print') as mock_print:
            self.augmenter.augment_image(
                Path('tests/cv_dataset/images/mock_image.jpg'),
            )
            mock_print.assert_any_call(
                f'Resize tests/cv_dataset/images/mock_image.jpg '
                f'due to large size: {mock_image.shape}',
            )

        mock_imread.assert_called_once_with(
            Path('tests/cv_dataset/images/mock_image.jpg'),
        )
        mock_imwrite.assert_called()
        mock_seq_call.assert_called()
        mock_open_file.assert_called()
        mock_write_text.assert_not_called()
        mock_write_bytes.assert_not_called()
        mock_rename.assert_not_called()

    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'imageio.imread',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'imageio.imwrite',
    )
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.Path.glob')
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'iaa.Sequential.__call__',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.Path.open',
        new_callable=mock_open,
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.write_text',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.write_bytes',
    )
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.Path.rename')
    def test_augment_image_none(
        self,
        mock_rename: MagicMock,
        mock_write_bytes: MagicMock,
        mock_write_text: MagicMock,
        mock_open_file: MagicMock,
        mock_seq_call: MagicMock,
        mock_glob: MagicMock,
        mock_imwrite: MagicMock,
        mock_imread: MagicMock,
    ) -> None:
        """
        Test the augment_image method when image is None.
        """
        # Mock image as None
        mock_imread.return_value = None

        # Mock the label file reading
        mock_glob.return_value = [
            Path('tests/cv_dataset/images/mock_image.jpg'),
        ]

        with patch('builtins.print') as mock_print:
            self.augmenter.augment_image(
                Path('tests/cv_dataset/images/mock_image.jpg'),
            )
            mock_print.assert_any_call(
                'Image is None or has no shape: '
                'tests/cv_dataset/images/mock_image.jpg',
            )

        mock_imread.assert_called_once_with(
            Path('tests/cv_dataset/images/mock_image.jpg'),
        )
        mock_imwrite.assert_not_called()
        mock_seq_call.assert_not_called()
        mock_open_file.assert_not_called()
        mock_write_text.assert_not_called()
        mock_write_bytes.assert_not_called()
        mock_rename.assert_not_called()

    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'imageio.imread',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'imageio.imwrite',
    )
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.Path.glob')
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.iaa.'
        'Sequential.__call__',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.Path.open',
        new_callable=mock_open,
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.write_text',
    )
    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'Path.write_bytes',
    )
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.Path.rename')
    def test_augment_image_no_shape(
        self,
        mock_rename: MagicMock,
        mock_write_bytes: MagicMock,
        mock_write_text: MagicMock,
        mock_open_file: MagicMock,
        mock_seq_call: MagicMock,
        mock_glob: MagicMock,
        mock_imwrite: MagicMock,
        mock_imread: MagicMock,
    ) -> None:
        """
        Test the augment_image method when image has no shape.
        """
        # Mock image with no shape
        mock_image = MagicMock()
        mock_image.shape = None
        mock_imread.return_value = mock_image

        # Mock the label file reading
        mock_glob.return_value = [
            Path('tests/cv_dataset/images/mock_image.jpg'),
        ]

        with patch('builtins.print') as mock_print:
            self.augmenter.augment_image(
                Path('tests/cv_dataset/images/mock_image.jpg'),
            )
            mock_print.assert_any_call(
                'Image is None or has no shape: '
                'tests/cv_dataset/images/mock_image.jpg',
            )

        mock_imread.assert_called_once_with(
            Path('tests/cv_dataset/images/mock_image.jpg'),
        )
        mock_imwrite.assert_not_called()
        mock_seq_call.assert_not_called()
        mock_open_file.assert_not_called()
        mock_write_text.assert_not_called()
        mock_write_bytes.assert_not_called()
        mock_rename.assert_not_called()

    @patch(
        'examples.YOLOv8_data_augmentation.data_augmentation.'
        'imageio.imread',
    )
    @patch('builtins.print')
    def test_augment_image_exception(
        self,
        mock_print: MagicMock,
        mock_imread: MagicMock,
    ) -> None:
        # Mock image reading exception
        mock_imread.side_effect = Exception('Mocked exception')

        # Test augment_image
        self.augmenter.augment_image(
            Path('tests/cv_dataset/images/mock_image.jpg'),
        )

        # Check if the print method was called with the correct output
        mock_print.assert_any_call(
            'Error processing image: tests/cv_dataset/images/mock_image.jpg:',
        )
        # Check if the exception message is in the print output
        self.assertTrue(
            any(
                'Mocked exception' in str(call)
                for call in mock_print.call_args_list
            ),
        )


if __name__ == '__main__':
    unittest.main()
