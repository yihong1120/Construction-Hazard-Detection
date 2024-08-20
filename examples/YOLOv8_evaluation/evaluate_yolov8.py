from __future__ import annotations

import argparse
from typing import Any

from ultralytics import YOLO  # Import the YOLO class


class ModelEvaluator:
    """
    A class to evaluate YOLOv8 models using Ultralytics framework.
    """

    def __init__(self, model_path: str, data_path: str):
        """
        Initialises the model evaluator with the path to the model and dataset.

        Args:
            model_path (str): The path to the trained model file.
            data_path (str): The path to the dataset configuration file.
        """
        self.model_path = model_path
        self.data_path = data_path
        self.model = YOLO(self.model_path)

    def evaluate(self) -> dict[str, Any]:
        """
        Evaluates the model using the provided dataset.

        Returns:
            Dict[str, Any]: The results from the model evaluation.
        """
        print(f"Evaluating model with data path: {self.data_path}")
        # The 'val' method is for evaluation; 'test' can be used if needed.
        return self.model.val(data=self.data_path)


def main():
    parser = argparse.ArgumentParser(description='Evaluates a YOLOv8 model.')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the trained model file.',
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the dataset configuration file.',
    )

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        model_path=args.model_path,
        data_path=args.data_path,
    )
    results = evaluator.evaluate()
    print(results)


if __name__ == '__main__':
    main()

"""example usage
python evaluate_yolov8.py \
    --model_path "../../models/pt/best_yolov8x.pt" \
    --data_path "dataset/data.yaml"
"""
