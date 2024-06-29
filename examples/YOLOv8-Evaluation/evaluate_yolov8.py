import argparse
from typing import Dict, Any

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

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluates the model using the provided dataset.

        Returns:
            Dict[str, Any]: The results from the model evaluation.
        """
        # The 'val' method is used for evaluation, 'test' could also be used as per the requirement.
        return self.model.val()

def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: The namespace containing command line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluates a YOLOv8 model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset configuration file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    evaluator = ModelEvaluator(model_path=args.model_path, data_path=args.data_path)
    results = evaluator.evaluate()
    print("Evaluation results:", results)

"""example usage
python evaluate_yolov8.py --model_path "../../models/pt/best_yolov8x.pt" --data_path "dataset/data.yaml"
"""
