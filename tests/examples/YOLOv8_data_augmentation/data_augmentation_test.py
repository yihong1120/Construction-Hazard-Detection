import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import sys
from examples.YOLOv8_data_augmentation.data_augmentation import DataAugmentation, main

class TestDataAugmentation(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment before each test.
        """
        self.train_path = 'tests/dataset'
        self.num_augmentations = 5
        self.augmenter = DataAugmentation(self.train_path, self.num_augmentations)

    @patch('examples.YOLOv8_data_augmentation.data_augmentation.imageio.imread')
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.imageio.imwrite')
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.Path.glob')
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.iaa.Sequential.__call__')
    def test_augment_image(self, mock_seq_call, mock_glob, mock_imwrite, mock_imread):
        """
        Test the augment_image method.
        """
        # Mock the image and label data
        mock_imread.return_value = MagicMock(shape=(100, 100, 3))
        mock_seq_call.return_value = (MagicMock(shape=(100, 100, 3)), MagicMock())
        
        # Mock the label file reading
        mock_glob.return_value = [Path('tests/dataset/images/1.jpg')]
        
        # Test augment_image
        image_path = Path('tests/dataset/images/1.jpg')
        self.augmenter.augment_image(image_path)
        
        mock_imread.assert_called_once_with(image_path)
        mock_imwrite.assert_called()
        mock_seq_call.assert_called()

    @patch('examples.YOLOv8_data_augmentation.data_augmentation.Path.glob')
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.gc.collect')
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.DataAugmentation.augment_image')
    def test_augment_data(self, mock_augment_image, mock_gc_collect, mock_glob):
        """
        Test the augment_data method.
        """
        mock_glob.return_value = [Path(f'tests/dataset/images/img_{i:02d}.jpg') for i in range(10)]
        
        self.augmenter.augment_data(batch_size=2)
        
        self.assertEqual(mock_augment_image.call_count, 10)
        self.assertEqual(mock_gc_collect.call_count, 5)

    @patch('builtins.open', new_callable=mock_open, read_data='0 0.5 0.5 0.5 0.5\n')
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.Path.exists', return_value=True)
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.Path.open', new_callable=mock_open)
    def test_read_label_file(self, mock_open_path, mock_exists, mock_file):
        """
        Test the read_label_file method.
        """
        label_path = Path('tests/dataset/labels/1.txt')
        image_shape = (100, 100, 3)
        
        bbs = self.augmenter.read_label_file(label_path, image_shape)
        
        self.assertEqual(len(bbs), 1)
        self.assertEqual(bbs[0].label, 0)
        self.assertAlmostEqual(bbs[0].x1, 25.0)
        self.assertAlmostEqual(bbs[0].y1, 25.0)
        self.assertAlmostEqual(bbs[0].x2, 75.0)
        self.assertAlmostEqual(bbs[0].y2, 75.0)

    @patch('builtins.open', new_callable=mock_open)
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.Path.exists', return_value=True)
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.Path.mkdir')
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.Path.parent', new_callable=MagicMock)
    def test_write_label_file(self, mock_parent, mock_mkdir, mock_exists, mock_file):
        """
        Test the write_label_file method.
        """
        bbs = MagicMock()
        bbs.bounding_boxes = [MagicMock(x1=10, y1=10, x2=50, y2=50, label=0)]
        
        label_path = Path('tests/dataset/labels/1.txt')
        self.augmenter.write_label_file(bbs, label_path, 100, 100)
        
        mock_file().write.assert_called()

    @patch('examples.YOLOv8_data_augmentation.data_augmentation.Path.glob')
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.random.shuffle')
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.Path.rename')
    def test_shuffle_data(self, mock_rename, mock_shuffle, mock_glob):
        """
        Test the shuffle_data method.
        """
        mock_glob.side_effect = [
            [Path(f'tests/dataset/images/img_{i:02d}.jpg') for i in range(5)],
            [Path(f'tests/dataset/labels/img_{i:02d}.txt') for i in range(5)]
        ]
        
        with patch('examples.YOLOv8_data_augmentation.data_augmentation.uuid.uuid4', side_effect=[f'uuid_{i:02d}' for i in range(5)]):
            self.augmenter.shuffle_data()
        
        mock_shuffle.assert_called_once()
        self.assertEqual(mock_rename.call_count, 10)

    @patch('time.sleep', return_value=None)  # Mock time.sleep to skip delay
    @patch.object(sys, 'argv', ['main', '--train_path=tests/dataset', '--num_augmentations=2', '--batch_size=2'])
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.DataAugmentation.shuffle_data')
    @patch('examples.YOLOv8_data_augmentation.data_augmentation.DataAugmentation.augment_data')
    def test_main(self, mock_augment_data, mock_shuffle_data, mock_sleep):
        """
        Test the main function with command line arguments.
        """
        main()
        mock_augment_data.assert_called_once_with(batch_size=2)
        mock_shuffle_data.assert_called_once()
        mock_sleep.assert_called_once()

if __name__ == '__main__':
    unittest.main()
