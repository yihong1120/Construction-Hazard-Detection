from __future__ import annotations

import argparse
import json
import os
from typing import Any

from PIL import Image


class COCOConverter:
    """
    Converts YOLO format annotations to COCO format.
    """

    def __init__(self, categories: list[str]):
        """
        Initialises COCO data structure and category mappings.

        Args:
            categories (List[str]): A list of category names.
        """
        self.coco_format: dict[str, list[Any]] = {
            'images': [],
            'annotations': [],
            'categories': [],
        }
        self.initialise_categories(categories)
        self.image_id = 1  # Unique ID for each image
        self.annotation_id = 1  # Unique ID for each annotation

    def initialise_categories(self, categories: list[str]):
        """Initialises categories for COCO format.

        Args:
            categories (List[str]): A list of category names.
        """
        for i, category in enumerate(categories):
            self.coco_format['categories'].append(
                {
                    'id': i + 1,
                    'name': category,
                    'supercategory': 'none',
                },
            )

    def convert_annotations(self, labels_dir: str, images_dir: str):
        """Reads YOLO formatted annotations and converts them to COCO format.

        Args:
            labels_dir (str): Directory containing YOLO labels.
            images_dir (str): Directory containing image files.
        """
        for filename in os.listdir(labels_dir):
            if filename.endswith('.txt'):
                image_name = filename.replace('.txt', '.jpg')
                image_path = os.path.join(images_dir, image_name)

                if not os.path.exists(image_path):
                    print(f"Warning: {image_path} does not exist.")
                    continue

                image = Image.open(image_path)
                width, height = image.size

                self.coco_format['images'].append(
                    {
                        'id': self.image_id,
                        'width': width,
                        'height': height,
                        'file_name': image_name,
                    },
                )

                with open(os.path.join(labels_dir, filename)) as file:
                    for line in file:
                        cls_id, x_center, y_center, bbox_width, bbox_height = (
                            (
                                float(x) if float(x) != int(
                                    float(x),
                                ) else int(float(x))
                            )
                            for x in line.strip().split()
                        )

                        x_min = (x_center - bbox_width / 2) * width
                        y_min = (y_center - bbox_height / 2) * height
                        bbox_width *= width
                        bbox_height *= height

                        self.coco_format['annotations'].append(
                            {
                                'id': self.annotation_id,
                                'image_id': self.image_id,
                                'category_id': int(cls_id) + 1,
                                'bbox': [
                                    x_min, y_min, bbox_width, bbox_height,
                                ],
                                'area': bbox_width * bbox_height,
                                'segmentation': [],
                                'iscrowd': 0,
                            },
                        )
                        self.annotation_id += 1
                self.image_id += 1

    def save_to_json(self, output_path: str):
        """Saves the COCO formatted data to a JSON file.

        Args:
            output_path (str): Path to save the JSON output.
        """
        with open(output_path, 'w') as json_file:
            json.dump(self.coco_format, json_file, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description='Convert YOLO format annotations to COCO format.',
    )
    parser.add_argument(
        '--labels_dir',
        type=str,
        required=True,
        help='Directory containing YOLO labels.',
    )
    parser.add_argument(
        '--images_dir',
        type=str,
        required=True,
        help='Directory containing image files.',
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file path for COCO formatted annotations.',
    )

    args = parser.parse_args()

    categories = [
        'Hardhat',
        'Mask',
        'NO-Hardhat',
        'NO-Mask',
        'NO-Safety Vest',
        'Person',
        'Safety Cone',
        'Safety Vest',
        'machinery',
        'vehicle',
    ]

    converter = COCOConverter(categories)
    converter.convert_annotations(args.labels_dir, args.images_dir)
    converter.save_to_json(args.output)
    print(f"COCO format annotations have been saved to {args.output}")


if __name__ == '__main__':
    main()

"""example usage
python convert_yolo_to_coco.py \
    --labels_dir tests/dataset/val/labels \
    --images_dir tests/dataset/val/images \
    --output tests/dataset/coco_annotations.json
"""
