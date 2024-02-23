import json
import os
import argparse
from PIL import Image
from typing import List, Dict

class COCOConverter:
    """Converts YOLO format annotations to COCO format."""

    def __init__(self, categories: List[str]):
        """
        Initialises COCO data structure and category mappings.
        
        Args:
            categories (List[str]): A list of category names.
        """
        self.coco_format = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.initialise_categories(categories)

    def initialise_categories(self, categories: List[str]):
        """Initialises categories for COCO format.
        
        Args:
            categories (List[str]): A list of category names.
        """
        for i, category in enumerate(categories):
            self.coco_format["categories"].append({
                "id": i + 1,
                "name": category,
                "supercategory": "none"
            })

    def convert_annotations(self, labels_dir: str, images_dir: str):
        """Reads YOLO formatted annotations and converts them to COCO format.
        
        Args:
            labels_dir (str): Directory containing YOLO labels.
            images_dir (str): Directory containing image files.
        """
        annotation_id = 1  # Unique ID for each annotation
        for filename in os.listdir(labels_dir):
            if filename.endswith('.txt'):
                image_name = filename.replace('.txt', '.jpg')
                image_path = os.path.join(images_dir, image_name)
                image = Image.open(image_path)
                width, height = image.size

                self.coco_format["images"].append({
                    "id": annotation_id,
                    "width": width,
                    "height": height,
                    "file_name": image_name
                })

                with open(os.path.join(labels_dir, filename), 'r') as file:
                    for line in file:
                        class_id, x_center, y_center, bbox_width, bbox_height = [
                            float(x) if float(x) != int(float(x)) else int(float(x)) 
                            for x in line.strip().split()
                        ]

                        x_min = (x_center - bbox_width / 2) * width
                        y_min = (y_center - bbox_height / 2) * height
                        bbox_width *= width
                        bbox_height *= height

                        self.coco_format["annotations"].append({
                            "id": annotation_id,
                            "image_id": annotation_id,
                            "category_id": class_id + 1,
                            "bbox": [x_min, y_min, bbox_width, bbox_height],
                            "area": bbox_width * bbox_height,
                            "segmentation": [],
                            "iscrowd": 0
                        })
                        annotation_id += 1

    def save_to_json(self, output_path: str):
        """Saves the COCO formatted data to a JSON file.
        
        Args:
            output_path (str): Path to save the JSON output.
        """
        with open(output_path, 'w') as json_file:
            json.dump(self.coco_format, json_file, indent=4)


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Convert YOLO format annotations to COCO format.')
    parser.add_argument('--labels_dir', type=str, required=True, help='Directory containing YOLO labels.')
    parser.add_argument('--images_dir', type=str, required=True, help='Directory containing image files.')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path for COCO formatted annotations.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    categories = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

    converter = COCOConverter(categories)
    converter.convert_annotations(args.labels_dir, args.images_dir)
    converter.save_to_json(args.output)
    print(f'COCO format annotations have been saved to {args.output}')

"""example usage
python convert_yolo_to_coco.py --labels_dir dataset/valid/labels --images_dir dataset/valid/images --output dataset/coco_annotations.json
"""