from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from matplotlib import pyplot as plt


class BoundingBoxVisualiser:
    """
    Class for visualising bounding boxes on images.
    """

    def __init__(
        self,
        image_path: str | Path,
        label_path: str | Path,
        class_names: list,
    ):
        """
        Initialises the BoundingBoxVisualiser with the specified image,
        label paths, and class names.

        Args:
            image_path: The path to the image file.
            label_path: The path to the label file.
            class_names: A list of class names.
        """
        self.image_path = Path(image_path)
        self.label_path = Path(label_path)
        self.class_names = class_names
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise ValueError(
                'Image could not be loaded. Please check the image path.',
            )

    def draw_bounding_boxes(self) -> None:
        """Draws bounding boxes on the image based on the label file."""
        height, width, _ = self.image.shape

        with open(self.label_path) as f:
            lines = f.readlines()

        for line in lines:
            class_id, x_centre, y_centre, bbox_width, bbox_height = map(
                float,
                line.split(),
            )

            # Convert from relative to absolute coordinates
            x_centre, bbox_width = x_centre * width, bbox_width * width
            y_centre, bbox_height = y_centre * height, bbox_height * height

            # Calculate the top left corner
            x1, y1 = int(
                x_centre - bbox_width / 2,
            ), int(y_centre - bbox_height / 2)
            x2, y2 = int(
                x_centre + bbox_width / 2,
            ), int(y_centre + bbox_height / 2)

            # Draw the rectangle and label
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = self.class_names[int(class_id)]
            cv2.putText(
                self.image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )

    def save_or_display_image(
        self, output_path: str | Path, save: bool,
    ) -> None:
        """
        Saves or displays the image based on the user's preference.

        Args:
            output_path: Path to save the image with drawn bounding boxes.
            save: A boolean indicating whether to save the image or display it.
        """
        if save:
            cv2.imwrite(str(output_path), self.image)
            print(f"Image saved to {output_path}")
        else:
            plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualise bounding boxes on images.',
    )
    parser.add_argument(
        '--image',
        help='The path to the image file.',
        required=True,
    )
    parser.add_argument(
        '--label',
        help='The path to the label file.',
        required=True,
    )
    parser.add_argument(
        '--output',
        help='The path where the image should be saved.',
        default='visualised_image.jpg',
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Flag whether to save the image instead of displaying it.',
    )

    args = parser.parse_args()

    # List of class names as specified in your data.yaml file
    class_names = [
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

    visualiser = BoundingBoxVisualiser(args.image, args.label, class_names)
    visualiser.draw_bounding_boxes()
    visualiser.save_or_display_image(args.output, args.save)


if __name__ == '__main__':
    main()


"""example
python visualise_bounding_boxes.py \
    --image './aug_4.jpg' \
    --label './aug_4.txt'
"""
