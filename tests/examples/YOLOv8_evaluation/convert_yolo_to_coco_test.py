from __future__ import annotations

import argparse
import json
import unittest
from unittest.mock import MagicMock
from unittest.mock import mock_open
from unittest.mock import patch

from examples.YOLOv8_evaluation.convert_yolo_to_coco import COCOConverter
from examples.YOLOv8_evaluation.convert_yolo_to_coco import main


class TestCOCOConverter(unittest.TestCase):
    def setUp(self):
        self.categories = [
            'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
            'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle',
        ]
        self.converter = COCOConverter(self.categories)

    @patch('examples.YOLOv8_evaluation.convert_yolo_to_coco.os.listdir')
    @patch('examples.YOLOv8_evaluation.convert_yolo_to_coco.os.path.exists')
    @patch('examples.YOLOv8_evaluation.convert_yolo_to_coco.Image.open')
    @patch(
        'builtins.open',
        new_callable=mock_open,
        read_data='0 0.5 0.5 0.5 0.5\n',
    )
    @patch('builtins.print')  # Mock print to check warning messages
    def test_convert_annotations(
        self,
        mock_print,
        mock_file,
        mock_image_open,
        mock_exists,
        mock_listdir,
    ):
        """
        Test the conversion of YOLO annotations to COCO format,
        including the handling of non-existing images.
        """
        # Setup the mocks
        mock_listdir.return_value = ['image1.txt', 'image2.txt']
        # Simulate image1 exists, image2 does not exist
        mock_exists.side_effect = lambda path: path.endswith('image1.jpg')
        mock_image = MagicMock()
        mock_image.size = (800, 600)
        mock_image_open.return_value = mock_image

        # Run the conversion
        self.converter.convert_annotations('labels_dir', 'images_dir')

        # Check that the image metadata is
        # correctly added only for the existing image
        self.assertEqual(len(self.converter.coco_format['images']), 1)
        self.assertEqual(
            self.converter.coco_format['images'][0]['file_name'], 'image1.jpg',
        )

        # Check that the annotation is
        # correctly added only for the existing image
        self.assertEqual(len(self.converter.coco_format['annotations']), 1)
        annotation = self.converter.coco_format['annotations'][0]
        self.assertEqual(annotation['image_id'], 1)
        self.assertEqual(annotation['category_id'], 1)
        self.assertEqual(annotation['bbox'], [200.0, 150.0, 400.0, 300.0])

        # Check that a warning was printed for the non-existing image
        mock_print.assert_called_with(
            'Warning: images_dir/image2.jpg does not exist.',
        )

    @patch('builtins.open', new_callable=mock_open)
    def test_save_to_json(self, mock_file):
        """
        Test saving the COCO format data to a JSON file.
        """
        # Add some dummy data to the COCO format
        self.converter.coco_format['images'].append({
            'id': 1,
            'width': 800,
            'height': 600,
            'file_name': 'image1.jpg',
        })
        self.converter.coco_format['annotations'].append({
            'id': 1,
            'image_id': 1,
            'category_id': 1,
            'bbox': [200.0, 150.0, 400.0, 300.0],
            'area': 120000.0,
            'segmentation': [],
            'iscrowd': 0,
        })

        # Run the save to JSON method
        self.converter.save_to_json('output.json')

        # Check that the file was written with the correct data
        mock_file.assert_called_once_with('output.json', 'w')
        written_data = json.loads(
            ''.join(call.args[0] for call in mock_file().write.mock_calls),
        )
        self.assertIn('images', written_data)
        self.assertIn('annotations', written_data)
        self.assertEqual(len(written_data['images']), 1)
        self.assertEqual(len(written_data['annotations']), 1)

    @patch(
        'builtins.open',
        new_callable=mock_open,
        read_data='0 0.5 0.5 0.5 0.5\n',
    )
    def test_initialise_categories(self, mock_file):
        """
        Test the initialisation of categories in COCO format.
        """
        # Reset categories to avoid duplication
        self.converter.coco_format['categories'] = []
        self.converter.initialise_categories(self.categories)
        categories = self.converter.coco_format['categories']
        self.assertEqual(len(categories), len(self.categories))
        for i, category in enumerate(self.categories):
            self.assertEqual(categories[i]['name'], category)
            self.assertEqual(categories[i]['id'], i + 1)

    @patch(
        'examples.YOLOv8_evaluation.convert_yolo_to_coco.'
        'COCOConverter.convert_annotations',
    )
    @patch(
        'examples.YOLOv8_evaluation.convert_yolo_to_coco.'
        'COCOConverter.save_to_json',
    )
    @patch('argparse.ArgumentParser.parse_args')
    def test_main(
            self,
            mock_parse_args,
            mock_save_to_json,
            mock_convert_annotations,
        ):
            """
            Test the main function.
            """
            # Setup the mock arguments
            mock_parse_args.return_value = argparse.Namespace(
                labels_dir='dataset/valid/labels',
                images_dir='dataset/valid/images',
                output='dataset/coco_annotations.json',
            )

            # Mock open to avoid creating a real file
            with patch('builtins.open', mock_open()):
                # Run the main function
                main()

                # Check that convert_annotations and save_to_json were called
                mock_convert_annotations.assert_called_once_with(
                    'dataset/valid/labels', 'dataset/valid/images',
                )
                mock_save_to_json.assert_called_once_with(
                    'dataset/coco_annotations.json',
                )


if __name__ == '__main__':
    unittest.main()
